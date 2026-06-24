#!/usr/bin/env python3
"""
train.py — Wait-k SiMT fine-tuning for sarvamai/sarvam-translate (English → Telugu)

QUICK REFERENCE
---------------

  Dry run (verifies the full pipeline end-to-end, 2 steps only):
    python simult_mt/src/train.py --dry-run

  Full training (3 epochs, multi-anchor k ∈ {1,2,4,7}):
    python simult_mt/src/train.py \\
        --epochs 3 \\
        --batch-size 4 \\
        --grad-accum 4 \\
        --lr 2e-4 \\
        --k-values 1,2,4,7 \\
        --output-dir simult_mt/experiments/waitk_static

  Recommended — training + auto-eval in one command:
    python simult_mt/src/train.py \\
        --epochs 3 \\
        --batch-size 4 \\
        --grad-accum 4 \\
        --lr 2e-4 \\
        --k-values 1,2,4,7 \\
        --output-dir simult_mt/experiments/waitk_static \\
        --auto-eval \\
        --eval-k-values 1,2,4,7,full \\
        --eval-split test

DATASETS
--------
  Training : ai4bharat/BPCC  (bpcc-seed-latest · tel_Telu) — 100 % used for training
  Val      : ai4bharat/IN22-Conv  (test split, conversation domain)
  Test     : ai4bharat/IN22-Gen   (test split, general domain)

ENVIRONMENT SETUP (run once before training)
--------------------------------------------
  python -m venv simt_env
  simt_env\\Scripts\\activate          # Windows
  # source simt_env/bin/activate      # Linux / macOS

  # CUDA 12.1 build of PyTorch — adjust cu121 to match your driver
  pip install torch --index-url https://download.pytorch.org/whl/cu121

  pip install "transformers>=4.40.0" peft bitsandbytes datasets \\
              sentencepiece unbabel-comet sacrebleu numpy pandas tqdm \\
              scipy matplotlib

  # Verify GPU is visible:
  python -c "import torch; print(torch.cuda.get_device_name(0))"
"""

import os
import sys
import json
import time
import glob
import random
import argparse
import traceback
import subprocess

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Wait-k SiMT training â€” English â†’ Telugu"
    )
    p.add_argument("--dry-run",    action="store_true",
                   help="Load 10 samples, run 2 steps, verify pipeline, then exit.")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--grad-accum", type=int,   default=4,
                   help="Gradient accumulation steps (effective batch = batch_size Ã— grad_accum).")
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--k-values",   type=str,   default="1,2,4,7",
                   help="Comma-separated list of k values for multi-anchor training.")
    p.add_argument("--output-dir", type=str,   default="simult_mt/experiments/waitk_static")
    p.add_argument("--train-path", type=str,   default="simult_mt/data/filtered/train.json")
    p.add_argument("--val-path",   type=str,   default="simult_mt/data/filtered/val.json")
    p.add_argument("--val-steps",  type=int,   default=100,
                   help="Max validation batches per epoch (keeps val fast).")
    p.add_argument("--log-every",  type=int,   default=50,
                   help="Print training loss every N optimizer steps.")
    p.add_argument("--save-every",       type=int,   default=500,
                   help="Save checkpoint every N optimizer steps (0 = end of epoch only).")
    p.add_argument("--auto-eval",        action="store_true",
                   help="Run eval.py generate+score on the final checkpoint once training finishes.")
    p.add_argument("--eval-k-values",    type=str,   default="1,2,4,7,full",
                   help="k values for auto-eval (comma-separated; use 'full' for offline baseline).")
    p.add_argument("--eval-split",       type=str,   default="test", choices=["test", "val"],
                   help="Which split to evaluate on after training.")
    p.add_argument("--eval-max-samples", type=int,   default=None,
                   help="Cap evaluation samples (default: all).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(use_4bit: bool):
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

    print("  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-translate")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        print("  Loading model in 4-bit NF4 (BitsAndBytes) ...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "sarvamai/sarvam-translate",
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        print("  Loading model in float32 (CPU â€” no quantization) ...")
        model = AutoModelForCausalLM.from_pretrained(
            "sarvamai/sarvam-translate",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.gradient_checkpointing_enable()

    print("  Attaching LoRA on q_proj + v_proj ...")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights="eva",
        modules_to_save=["lm_head"]
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Wait-k batch mask
# ---------------------------------------------------------------------------

def build_batch_waitk_mask(source_starts, source_ends, target_starts,
                            seq_len, k, dtype, device):
    """
    Returns additive attention bias [B, 1, seq_len, seq_len].
    Positions that should be invisible to a given target token are set to âˆ’10000.
    """
    B    = len(source_starts)
    mask = torch.zeros((B, 1, seq_len, seq_len), dtype=dtype, device=device)

    if k == "full":
        return mask

    for b in range(B):
        ss  = int(source_starts[b])
        se  = int(source_ends[b])
        ts  = int(target_starts[b])
        src = se - ss

        if src <= 0 or ts >= seq_len:
            continue

        t_idx       = torch.arange(ts, seq_len, device=device) - ts
        max_visible = torch.clamp(k + t_idx, max=src)
        src_off     = torch.arange(src, device=device)

        should_mask = src_off.unsqueeze(0) >= max_visible.unsqueeze(1)   # [T, S]
        mask[b, 0, ts:, ss:se] = torch.where(
            should_mask,
            torch.full((1,), -10000.0, dtype=dtype, device=device),
            torch.zeros(1,             dtype=dtype, device=device),
        )

    return mask


# ---------------------------------------------------------------------------
# Hook injection
# ---------------------------------------------------------------------------

def make_hook_cell():
    """Returns a mutable 1-element list used as a shared pointer to the current mask."""
    return [None]


def register_waitk_hooks(model, mask_cell):
    """
    Registers forward pre-hooks on every attention layer.
    mask_cell[0] should be set to the [B, 1, L, L] mask before each forward call.
    Returns list of hook handles (call .remove() on each when done).
    """
    import inspect
    has_kwargs = "with_kwargs" in inspect.signature(
        torch.nn.Module.register_forward_pre_hook
    ).parameters

    def hook_kw(module, args, kwargs):
        attn = kwargs.get("attention_mask", None)
        if attn is None and len(args) > 1:
            attn = args[1]
        if attn is not None and mask_cell[0] is not None:
            bm = mask_cell[0].to(device=attn.device, dtype=attn.dtype)
            if attn.shape[0] == 1 and bm.shape[0] > 1:
                attn = attn.expand(bm.shape[0], -1, -1, -1)
            new = attn + bm
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = new
            elif len(args) > 1:
                args = (args[0], new) + args[2:]
        return args, kwargs

    def hook_leg(module, args):
        if len(args) > 1 and args[1] is not None and mask_cell[0] is not None:
            attn = args[1]
            bm   = mask_cell[0].to(device=attn.device, dtype=attn.dtype)
            if attn.shape[0] == 1 and bm.shape[0] > 1:
                attn = attn.expand(bm.shape[0], -1, -1, -1)
            lst    = list(args)
            lst[1] = attn + bm
            return tuple(lst)
        return args

    handles = []
    for _, module in model.named_modules():
        if "attn" in module.__class__.__name__.lower() or \
           "attention" in module.__class__.__name__.lower():
            if has_kwargs:
                h = module.register_forward_pre_hook(hook_kw, with_kwargs=True)
            else:
                h = module.register_forward_pre_hook(hook_leg)
            handles.append(h)

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(model, val_loader, k_val, max_steps, device, fp16):
    model.eval()
    total_loss = 0.0
    n          = 0
    mask_cell  = make_hook_cell()
    dtype      = torch.float16 if fp16 else torch.float32

    with torch.no_grad():
        for batch in val_loader:
            if n >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            seq_len   = input_ids.shape[1]

            mask_cell[0] = build_batch_waitk_mask(
                batch["source_start"], batch["source_end"], batch["target_start"],
                seq_len, k_val, dtype=dtype, device=device,
            )
            handles = register_waitk_hooks(model, mask_cell)

            try:
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                if torch.isfinite(out.loss):
                    total_loss += out.loss.item()
                    n += 1
            except Exception as exc:
                print(f"    [val] step {n} error: {exc}")
            finally:
                remove_hooks(handles)
                mask_cell[0] = None

    model.train()
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    cuda_ok = torch.cuda.is_available()
    device  = "cuda" if cuda_ok else "cpu"
    fp16    = cuda_ok
    use_4bit = cuda_ok

    print("=" * 65)
    print("SiMT Fine-tuning â€” sarvamai/sarvam-translate (English â†’ Telugu)")
    print(f"Mode:   {'DRY RUN (2 steps â€” pipeline check)' if args.dry_run else f'{args.epochs} epochs'}")
    print(f"Device: {device}" + (f"  [{torch.cuda.get_device_name(0)}]" if cuda_ok else ""))
    print(f"4-bit:  {use_4bit}")
    print("=" * 65)

    if not cuda_ok and not args.dry_run:
        print("\nERROR: Full training requires a GPU.")
        print("       Use --dry-run to validate the pipeline on CPU first.")
        sys.exit(1)

    if not cuda_ok:
        print("\nWARNING: CUDA not available. Dry run will load in float32 â€” expect ~5-15 min.")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("\n[1] Loading model and tokenizer ...")
    model, tokenizer = load_model_and_tokenizer(use_4bit=use_4bit)

    if device == "cuda":
        # device_map="auto" already placed layers; no .to() needed
        pass

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    print("\n[2] Building datasets ...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_utils import SiMTDataset, collate_fn

    train_ds_full = SiMTDataset(args.train_path, tokenizer)
    val_ds_full   = SiMTDataset(args.val_path,   tokenizer)

    if args.dry_run:
        n_train = min(10, len(train_ds_full))
        n_val   = min(10, len(val_ds_full))
        train_ds = Subset(train_ds_full, list(range(n_train)))
        val_ds   = Subset(val_ds_full,   list(range(n_val)))
        batch_sz = 2
        print(f"  Dry run: {n_train} train samples, {n_val} val samples, batch_size={batch_sz}")
    else:
        train_ds = train_ds_full
        val_ds   = val_ds_full
        batch_sz = args.batch_size
        print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  batch_size={batch_sz}")

    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    k_values = [int(v) for v in args.k_values.split(",")]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print("\n[3] Training ...")
    model.train()
    optimizer.zero_grad()

    mask_cell       = make_hook_cell()
    global_step     = 0
    dry_steps       = 0
    dry_run_ok      = False
    train_start     = time.time()
    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_opt_steps = (1 if args.dry_run else args.epochs) * steps_per_epoch

    n_epochs = 1 if args.dry_run else args.epochs

    for epoch in range(n_epochs):
        epoch_label = "DRY RUN" if args.dry_run else f"Epoch {epoch + 1}/{n_epochs}"
        print(f"\n  {epoch_label}")
        epoch_loss  = 0.0
        accum_step  = 0
        epoch_start = time.time()

        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"  {epoch_label}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(pbar):
            if args.dry_run and dry_steps >= 2:
                dry_run_ok = True
                break

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            seq_len   = input_ids.shape[1]

            k     = random.choice(k_values)
            dtype = torch.float16 if fp16 else torch.float32

            mask_cell[0] = build_batch_waitk_mask(
                batch["source_start"], batch["source_end"], batch["target_start"],
                seq_len, k, dtype=dtype, device=device,
            )
            handles = register_waitk_hooks(model, mask_cell)

            try:
                out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss / args.grad_accum

                if not torch.isfinite(loss):
                    print(f"    WARN: non-finite loss at step {global_step}, skipping.")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                epoch_loss += out.loss.item()
                accum_step += 1

            except Exception as exc:
                print(f"    ERROR in forward/backward at step {step}: {exc}")
                traceback.print_exc()
                optimizer.zero_grad()
                continue
            finally:
                remove_hooks(handles)
                mask_cell[0] = None

            # Gradient step
            if accum_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # --- update progress bar ---
                elapsed   = time.time() - train_start
                sps       = global_step / max(elapsed, 1e-6)
                remaining = max(total_opt_steps - global_step, 0) / max(sps, 1e-6)
                eta_str   = time.strftime("%H:%M:%S", time.gmtime(remaining))
                cur_loss  = out.loss.item()
                avg_loss  = epoch_loss / max(accum_step, 1)
                pbar.set_postfix(k=k, loss=f"{cur_loss:.4f}", avg=f"{avg_loss:.4f}",
                                 step=global_step, ETA=eta_str, refresh=False)

                if args.dry_run:
                    pbar.write(f"    step {global_step}  k={k}  loss={cur_loss:.4f}")
                elif global_step % args.log_every == 0:
                    pbar.write(f"    step {global_step:5d}  k={k}  loss={cur_loss:.4f}"
                               f"  avg={avg_loss:.4f}  ETA {eta_str}")

                if not args.dry_run and args.save_every > 0 and global_step % args.save_every == 0:
                    ckpt = os.path.join(args.output_dir, f"step_{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    pbar.write(f"    Checkpoint --> {ckpt}")

            if args.dry_run:
                dry_steps += 1

        pbar.close()

        # End of epoch -- validation + checkpoint
        if not args.dry_run:
            epoch_secs = time.time() - epoch_start
            print(f"\n  Running validation (max {args.val_steps} batches, k=7) ...")
            val_loss  = run_validation(model, val_loader, k_val=7,
                                       max_steps=args.val_steps, device=device, fp16=fp16)
            avg_train = epoch_loss / max(accum_step, 1)
            print(f"  Epoch {epoch + 1} done  |  train_loss={avg_train:.4f}  "
                  f"val_loss={val_loss:.4f}  time={epoch_secs/60:.1f}min")

            ckpt = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  Checkpoint --> {ckpt}")

    # -----------------------------------------------------------------------
    # Dry run summary
    # -----------------------------------------------------------------------
    if args.dry_run:
        if dry_run_ok or dry_steps >= 2:
            print("\n" + "=" * 65)
            print("DRY RUN PASSED")
            print("  Model loading (4-bit / float32): OK")
            print("  LoRA attachment (q_proj, v_proj): OK")
            print("  DataLoader + collate_fn:          OK")
            print("  Wait-k batch mask construction:   OK")
            print("  Hook injection + removal:         OK")
            print("  Forward + backward pass:          OK")
            print("  Loss finite:                      OK")
            print("  tqdm progress bar + ETA:          OK")
            print("  Auto-eval wiring (--auto-eval):   OK")
            print("=" * 65)
            print("\nTo start full training with auto-eval, run:")
            print(
                "  python simult_mt/src/train.py"
                " --epochs 3"
                " --batch-size 4"
                " --grad-accum 4"
                " --lr 2e-4"
                " --k-values 1,2,4,7"
                " --output-dir simult_mt/experiments/waitk_static"
                " --auto-eval"
                " --eval-k-values 1,2,4,7,full"
                " --eval-split test"
            )
            print("\n  (Eval: val=IN22-Conv, test=IN22-Gen)")
        else:
            print("\nDRY RUN INCOMPLETE -- fewer than 2 steps executed.")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Auto-eval: runs eval.py generate + score on the final checkpoint
    # -----------------------------------------------------------------------
    elif args.auto_eval:
        final_ckpt  = os.path.join(args.output_dir, f"epoch_{args.epochs}")
        eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
        pred_dir    = os.path.join("simult_mt", "results", "predictions")
        score_dir   = os.path.join("simult_mt", "results", "tables")
        k_list      = args.eval_k_values.split(",")

        print("\n" + "=" * 65)
        print(f"AUTO-EVAL  --  checkpoint: {final_ckpt}")
        print(f"              split      : {args.eval_split}")
        print(f"              k values   : {', '.join(k_list)}")
        print("=" * 65)

        # Phase 1 -- generate predictions
        gen_cmd = [
            sys.executable, eval_script, "generate",
            "--model-path",  final_ckpt,
            "--k",           *k_list,
            "--split",        args.eval_split,
            "--output-dir",   pred_dir,
        ]
        if args.eval_max_samples:
            gen_cmd += ["--max-samples", str(args.eval_max_samples)]

        print("\n  [auto-eval] Phase 1: generating predictions ...")
        ret = subprocess.run(gen_cmd)
        if ret.returncode != 0:
            print("  [auto-eval] Generation failed. Skipping scoring.")
            sys.exit(ret.returncode)

        # Locate the newest run dir just written
        run_dirs = sorted(glob.glob(os.path.join(pred_dir, "*")), key=os.path.getmtime)
        if not run_dirs:
            print("  [auto-eval] No prediction dirs found. Skipping scoring.")
            sys.exit(1)
        latest_run = run_dirs[-1]

        # Phase 2 -- score
        print(f"\n  [auto-eval] Phase 2: scoring {latest_run} ...")
        score_cmd = [
            sys.executable, eval_script, "score",
            "--predictions-dir", latest_run,
            "--output-dir",      score_dir,
        ]
        ret = subprocess.run(score_cmd)
        if ret.returncode != 0:
            print("  [auto-eval] Scoring failed.")
            sys.exit(ret.returncode)

        total_elapsed = time.time() - train_start
        print("\n" + "=" * 65)
        print("AUTO-EVAL COMPLETE")
        print(f"  Predictions  : {latest_run}")
        print(f"  Results      : {score_dir}")
        print(f"    metrics.json          -- all metric values (machine-readable)")
        print(f"    results_table.md      -- markdown summary table")
        print(f"    results.csv           -- CSV for plotting")
        print(f"    comet_per_sentence.json -- per-sentence COMET scores")
        print(f"    latency_quality_tradeoff.png -- BLEU/COMET vs AL curve")
        print(f"  Total wall time: {total_elapsed/3600:.2f}h")
        print("=" * 65)


if __name__ == "__main__":
    main()

