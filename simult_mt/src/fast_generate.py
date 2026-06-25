#!/usr/bin/env python3
"""
fast_generate.py — Faster inference for wait-k SiMT evaluation
================================================================
Key speedups over the original eval.py:
  1. bf16 instead of 4-bit quant (much faster compute on A6000)
  2. torch.compile on the model
  3. Batched generation for the 'full' baseline; single-sample loop for wait-k
     (batching wait-k requires per-sample hooks — see note in generate_batch)
  4. Each script invocation handles a single k-value so you can
     run k=2,4,7,full in parallel in separate screen windows.

Usage — pick a shared RUN_NAME for all k-windows, e.g. 20260625_v2:
  source ~/simt_env/bin/activate && cd ~/SimulMT
  python simult_mt/src/fast_generate.py --k 2    --run-name 20260625_v2
  python simult_mt/src/fast_generate.py --k 4    --run-name 20260625_v2
  python simult_mt/src/fast_generate.py --k 7    --run-name 20260625_v2
  python simult_mt/src/fast_generate.py --k full --run-name 20260625_v2

Then score with:
  python simult_mt/src/eval.py score --predictions-dir simult_mt/results/predictions/20260625_v2 --no-comet
"""

import os, sys, json, argparse, traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",     default="praneet3/sarvam-translate-waitk-simulmt")
    p.add_argument("--k",              required=True,
                   help="Single wait-k value (int or 'full')")
    p.add_argument("--split",          default="test", choices=["test", "val"])
    p.add_argument("--data-dir",       default="simult_mt/data/filtered")
    p.add_argument("--output-dir",     default="simult_mt/results/predictions")
    p.add_argument("--run-name",       default=None)
    p.add_argument("--max-samples",    type=int, default=100)
    p.add_argument("--batch-size",     type=int, default=8,
                   help="Batch size for 'full' baseline only. "
                        "Wait-k runs always use batch_size=1 for correctness.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--no-compile",     action="store_true")
    p.add_argument("--dtype",          default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    return p


def compute_latency_metrics(source_len, hyp_len, k):
    S, T = source_len, hyp_len
    if S == 0 or T == 0:
        return {"AP": None, "AL": None, "DAL": None}
    k_int = S if k == "full" else int(k)
    g = [min(k_int + (t - 1), S) for t in range(1, T + 1)]
    AP = sum(gt / S for gt in g) / T
    tau_S = next((t for t, gt in enumerate(g, 1) if gt == S), T)
    AL = sum(g[t-1] - (t - 1) * S / T for t in range(1, tau_S + 1)) / max(tau_S, 1)
    DAL = sum(max(g[t-1] - (t - 1) * S / T, 0) for t in range(1, T + 1)) / T
    return {"AP": round(AP, 4), "AL": round(AL, 4), "DAL": round(DAL, 4)}


def load_model(model_path, dtype_str="bf16"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[dtype_str]
    print(f"  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for decoder-only batched generation
    print(f"  Loading model in {dtype_str} (no quantization) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",   # required for wait-k additive mask hooks
    )
    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, source_text, tgt_lang="Telugu"):
    msgs = [
        {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
        {"role": "user",   "content": source_text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def compute_source_offsets(tokenizer, source_text, tgt_lang="Telugu"):
    sys_only = [{"role": "system", "content": f"Translate the text below to {tgt_lang}."}]
    with_src  = [
        {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
        {"role": "user",   "content": source_text},
    ]
    prompt_only_str = tokenizer.apply_chat_template(sys_only, tokenize=False, add_generation_prompt=False)
    prompt_src_str  = tokenizer.apply_chat_template(with_src, tokenize=False, add_generation_prompt=False)
    prompt_full_str = tokenizer.apply_chat_template(with_src, tokenize=False, add_generation_prompt=True)
    enc = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    return enc(prompt_only_str), enc(prompt_src_str), enc(prompt_full_str)


def generate_batch(model, tokenizer, sources, k, max_new_tokens, device):
    """
    Generate translations for a list of source sentences under a wait-k policy.

    For k='full': sources are batched together for throughput.
    For wait-k (int): sources are processed one-at-a-time.

    WHY single-sample for wait-k?
    ==============================
    WaitKMaskController holds a single (source_start, source_end, target_start)
    context and broadcasts it across all heads/layers via a forward pre-hook.
    In a left-padded batch, every sample has different absolute token positions,
    so one shared context would mask the wrong columns for all but one sample.

    The correct fix is batch_size=1 for wait-k.  The 'full' path below retains
    batched throughput since it needs no masking at all.

    The previous implementation tried to hook Linear projection layers
    (q_proj / k_proj / v_proj) which never receive an attention_mask, so the
    constraint was silently dropped on every call — producing identical output
    across all k values.  We now use WaitKMaskController.register_hooks() which
    correctly targets the Attention *class* modules and uses waitk_bias() to
    handle both the prefill pass (q_len == kv_len) and every KV-cached decode
    step (q_len == 1, kv_len grows), matching the training-time masking exactly.
    """
    import torch
    from masking import WaitKMaskController

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # ------------------------------------------------------------------
    # Full-attention baseline: batch for speed, no masking needed
    # ------------------------------------------------------------------
    if k == "full":
        B = len(sources)
        prompts = [format_prompt(tokenizer, s) for s in sources]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        input_ids      = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        pad_len        = input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        results = []
        for i in range(B):
            gen = out[i, pad_len:].tolist()
            while gen and gen[-1] in (eos_id, pad_id):
                gen = gen[:-1]
            results.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
        return results

    # ------------------------------------------------------------------
    # Wait-k: one sample at a time with correct WaitKMaskController hooks
    # ------------------------------------------------------------------
    ctrl = WaitKMaskController(model)
    results = []

    for src in sources:
        prompt_str = format_prompt(tokenizer, src)
        enc_one    = tokenizer(
            prompt_str, return_tensors="pt", add_special_tokens=False
        )
        ids_one  = enc_one.input_ids.to(device)      # [1, L]
        mask_one = enc_one.attention_mask.to(device)  # [1, L]
        prompt_len = ids_one.shape[1]

        # Compute offsets for this sample (no padding offset needed: batch=1)
        ss, se, ts = compute_source_offsets(tokenizer, src)

        ctrl.set_context(
            source_start=ss,
            source_end=se,
            target_start=ts,
            seq_len=prompt_len + max_new_tokens,
            k=int(k),
        )
        ctrl.register_hooks()   # hooks fire on every forward inside generate()

        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids_one,
                    attention_mask=mask_one,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
        finally:
            ctrl.remove_hooks()  # always clean up, even on error

        gen = out[0, prompt_len:].tolist()
        while gen and gen[-1] in (eos_id, pad_id):
            gen = gen[:-1]
        results.append(tokenizer.decode(gen, skip_special_tokens=True).strip())

    return results


def main():
    import torch
    args   = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k      = "full" if args.k == "full" else int(args.k)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, run_name)
    os.makedirs(out_root, exist_ok=True)

    manifest_path = os.path.join(out_root, "manifest.json")
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            json.dump({
                "run_name": run_name, "model_path": args.model_path,
                "split": args.split, "n_samples": args.max_samples,
                "k_values": ["2","4","7","full"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    split_path = os.path.join(args.data_dir, f"{args.split}.json")
    samples = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples.")

    model, tokenizer = load_model(args.model_path, args.dtype)

    if not args.no_compile and hasattr(torch, "compile"):
        print("  Applying torch.compile (mode=reduce-overhead) ...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile OK.")
        except Exception as e:
            print(f"  torch.compile failed ({e}), continuing without.")

    k_tag    = f"k{k}" if k != "full" else "full"
    pred_dir = os.path.join(out_root, k_tag)
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, "predictions.jsonl")

    # For wait-k runs, always use batch_size=1 (see generate_batch docstring).
    # For 'full', use the requested batch_size for throughput.
    bs = args.batch_size if k == "full" else 1

    print(f"\nGenerating k={k}  ->  {pred_path}")
    print(f"effective_batch_size={bs}  max_new_tokens={args.max_new_tokens}\n")
    if k != "full":
        print("  (Wait-k uses batch_size=1 to guarantee correct per-sample masking)\n")

    n = len(samples)

    with open(pred_path, "w", encoding="utf-8") as fout:
        for start in range(0, n, bs):
            batch   = samples[start:start + bs]
            sources = [s["source"] for s in batch]
            end     = min(start + len(batch), n)
            print(f"  [{start+1}–{end}/{n}] ...", flush=True)
            try:
                hyps = generate_batch(model, tokenizer, sources, k, args.max_new_tokens, device)
            except Exception as e:
                print(f"  ERROR on batch starting {start}: {e}")
                traceback.print_exc()
                hyps = [""] * len(batch)

            for sample, hyp in zip(batch, hyps):
                src_len = len(tokenizer.encode(sample["source"], add_special_tokens=False))
                ref_len = len(tokenizer.encode(sample["target"], add_special_tokens=False))
                hyp_len = len(tokenizer.encode(hyp,             add_special_tokens=False))
                lat     = compute_latency_metrics(src_len, hyp_len, k)
                fout.write(json.dumps({
                    "id": sample["id"], "source": sample["source"],
                    "reference": sample["target"], "hypothesis": hyp,
                    "k": str(k), "src_len": src_len, "ref_len": ref_len, "hyp_len": hyp_len,
                    "AP": lat["AP"], "AL": lat["AL"], "DAL": lat["DAL"],
                }, ensure_ascii=False) + "\n")

    print(f"\nDone! -> {pred_path}")
    print(f"\nScore when all k done:")
    print(f"  python simult_mt/src/eval.py score --predictions-dir {out_root} --no-comet")


if __name__ == "__main__":
    main()
