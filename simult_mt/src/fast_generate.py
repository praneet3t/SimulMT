#!/usr/bin/env python3
"""
fast_generate.py — Faster inference for wait-k SiMT evaluation
================================================================
Key speedups over the original eval.py:
  1. bf16 instead of 4-bit quant (much faster compute on A6000)
  2. torch.compile on the model
  3. Batched generation (multiple samples at once per k-value)
  4. Each script invocation handles a single k-value so you can
     run k=1,2,4,7,full in parallel in separate screen windows.

Usage — pick a shared RUN_NAME for all k-windows, e.g. 20260625_fast:
  source ~/simt_env/bin/activate && cd ~/SimulMT
  python simult_mt/src/fast_generate.py --k 1    --run-name 20260625_fast
  python simult_mt/src/fast_generate.py --k 2    --run-name 20260625_fast
  python simult_mt/src/fast_generate.py --k 4    --run-name 20260625_fast
  python simult_mt/src/fast_generate.py --k 7    --run-name 20260625_fast
  python simult_mt/src/fast_generate.py --k full --run-name 20260625_fast

Then score with:
  python simult_mt/src/eval.py score --predictions-dir simult_mt/results/predictions/20260625_fast --no-comet
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
    p.add_argument("--batch-size",     type=int, default=8)
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
    import torch, inspect
    from masking import WaitKMaskController

    B       = len(sources)
    prompts = [format_prompt(tokenizer, s) for s in sources]
    enc     = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids      = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    pad_len        = input_ids.shape[1]

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # Compute per-sample offsets adjusted for left-padding
    offsets = []
    for i, src in enumerate(sources):
        ss0, se0, ts0 = compute_source_offsets(tokenizer, src)
        actual_len    = int(attention_mask[i].sum().item())
        left_pad      = pad_len - actual_len
        offsets.append((ss0 + left_pad, se0 + left_pad, ts0 + left_pad))

    if k == "full":
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

    # Wait-k: batch-aware hook via WaitKMaskController.build_batch_mask
    ctrl = WaitKMaskController(model)
    source_starts = [o[0] for o in offsets]
    source_ends   = [o[1] for o in offsets]
    target_starts = [o[2] for o in offsets]

    def waitk_hook_fn(module, args, kwargs):
        if "attention_mask" in kwargs:
            amask = kwargs["attention_mask"]
        elif len(args) > 1:
            amask = args[1]
        else:
            return
        if amask is None or amask.ndim != 4:
            return
        bsz, _, q_len, kv_len = amask.shape
        bias = ctrl.build_batch_mask(
            source_starts[:bsz], source_ends[:bsz], target_starts[:bsz],
            kv_len, int(k), dtype=amask.dtype, device=amask.device,
        )  # [B, 1, kv_len, kv_len]
        bias = bias[:, :, -q_len:, :]  # [B, 1, q_len, kv_len]
        new_mask = amask + bias
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = new_mask
        else:
            lst = list(args); lst[1] = new_mask
            return tuple(lst), kwargs

    has_kwargs = "with_kwargs" in inspect.signature(
        torch.nn.Module.register_forward_pre_hook
    ).parameters
    handles = []
    for name, module in model.named_modules():
        nm = name.lower()
        if ("attn" in nm or "attention" in nm) and not list(module.children()):
            if has_kwargs:
                h = module.register_forward_pre_hook(waitk_hook_fn, with_kwargs=True)
            else:
                h = module.register_forward_pre_hook(
                    lambda m, a: waitk_hook_fn(m, a, {})
                )
            handles.append(h)

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
    finally:
        for h in handles:
            h.remove()

    results = []
    for i in range(B):
        gen = out[i, pad_len:].tolist()
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
                "k_values": ["1","2","4","7","full"],
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

    print(f"\nGenerating k={k}  ->  {pred_path}")
    print(f"batch_size={args.batch_size}  max_new_tokens={args.max_new_tokens}\n")

    bs = args.batch_size
    n  = len(samples)

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
