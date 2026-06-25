#!/usr/bin/env python3
"""
LoRA wait-k fine-tuning for English -> Telugu SiMT (v2).

What is different from simult_mt/src/train.py:
  * Masking is an explicit 4D mask passed directly to the model forward
    (masking.build_batch_mask). No forward pre-hooks -> nothing to leak, and the
    same builder is reused at evaluation time.
  * Multi-anchor coverage is PER SAMPLE: every example in a batch is assigned its
    own wait-k value sampled from the anchor set, so a single optimizer step trains
    several latency regimes at once (denser than one k per whole batch).

Quick pipeline check (CPU, no GPU needed):
    python simt_v2/train.py --dry-run

Full fine-tune:
    python simt_v2/train.py \
        --train simult_mt/data/filtered/train.json \
        --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
        --k-values 1,2,4,7 --output-dir simt_v2/checkpoints
"""

import os
import sys
import random
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_batch_mask
from data import SiMTDataset, make_collate

BASE_MODEL = "sarvamai/sarvam-translate"


def parse_args():
    p = argparse.ArgumentParser(description="v2 wait-k LoRA fine-tuning (per-sample anchors)")
    p.add_argument("--train",      default="simult_mt/data/filtered/train.json")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--grad-accum", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--k-values",   default="1,2,4,7", help="comma-separated anchor k set")
    p.add_argument("--max-len",    type=int,   default=320)
    p.add_argument("--output-dir", default="simt_v2/checkpoints")
    p.add_argument("--log-every",  type=int,   default=50)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--dry-run",    action="store_true",
                   help="10 samples, 2 steps, CPU float32 — validates the pipeline")
    return p.parse_args()


def load_model(dry_run):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if not dry_run and torch.cuda.is_available():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto",
            attn_implementation="eager",   # eager consumes the explicit 4D float mask
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tok


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    anchors = [int(x) for x in args.k_values.split(",")]

    model, tok = load_model(args.dry_run)
    device = next(model.parameters()).device

    ds = SiMTDataset(args.train, tok, max_len=args.max_len)
    if args.dry_run:
        ds.rows = ds.rows[:10]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=not args.dry_run,
                        collate_fn=make_collate(tok.pad_token_id))

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()

    max_steps = 2 if args.dry_run else None
    step = 0
    for epoch in range(1 if args.dry_run else args.epochs):
        running = 0.0
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            pad_mask  = batch["pad_mask"].to(device)
            seq_len   = input_ids.shape[1]

            # one wait-k anchor PER SAMPLE
            ks = [random.choice(anchors) for _ in range(input_ids.shape[0])]
            mask4d = build_batch_mask(
                batch["source_start"], batch["source_end"], batch["target_start"],
                seq_len=seq_len, k=ks, pad_mask=pad_mask,
                dtype=torch.float32, device=device,
            )

            out = model(input_ids=input_ids, attention_mask=mask4d, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                optim.step()
                optim.zero_grad()
                step += 1

            if (i + 1) % args.log_every == 0:
                avg = running / (i + 1)
                print(f"epoch {epoch+1} step {i+1}/{len(loader)} "
                      f"loss {out.loss.item():.4f} avg {avg:.4f} ks {ks}")

            if max_steps and step >= max_steps:
                break

        if not args.dry_run:
            ckpt = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            print(f"saved checkpoint -> {ckpt}")

    if args.dry_run:
        assert torch.isfinite(out.loss), "loss is not finite"
        print("\n================ DRY RUN PASSED ================")
        print("  model load + LoRA          : OK")
        print("  dataset + collate          : OK")
        print("  per-sample 4D wait-k mask  : OK")
        print("  forward + backward + loss  : OK")
        print("===============================================")


if __name__ == "__main__":
    main()
