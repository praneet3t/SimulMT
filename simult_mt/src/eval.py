#!/usr/bin/env python3
"""
eval.py Ã¢â‚¬â€ Complete evaluation for English Ã¢â€ â€™ Telugu SiMT
========================================================

Two-phase design so you never re-run the model unnecessarily.

  PHASE 1 Ã¢â‚¬â€ Generate predictions (needs GPU + fine-tuned model):
    python simult_mt/src/eval.py generate \\
        --model-path simult_mt/experiments/waitk_static/epoch_3 \\
        --k 1 2 4 7 full \\
        --split test \\
        --output-dir simult_mt/results/predictions

  PHASE 2 Ã¢â‚¬â€ Score from saved predictions (CPU, fast, repeatable):
    python simult_mt/src/eval.py score \\
        --predictions-dir simult_mt/results/predictions \\
        --output-dir simult_mt/results/tables

  Re-score any time with no model needed:
    python simult_mt/src/eval.py score --predictions-dir ... --output-dir ...

  Compare multiple runs:
    python simult_mt/src/eval.py compare \\
        --dirs simult_mt/results/predictions/run1 simult_mt/results/predictions/run2 \\
        --output-dir simult_mt/results/tables
"""

import os
import sys
import json
import argparse
import random
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="SiMT evaluation Ã¢â‚¬â€ generate predictions + score metrics"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- generate ------------------------------------------------------------
    gen = sub.add_parser("generate", help="Run model on test/val set, save predictions")
    gen.add_argument("--model-path",    required=True,
                     help="Path to fine-tuned LoRA checkpoint (or 'sarvamai/sarvam-translate' for baseline)")
    gen.add_argument("--k",             nargs="+", default=["1", "2", "4", "7", "full"],
                     help="Wait-k values to evaluate (use 'full' for full-attention baseline)")
    gen.add_argument("--split",         default="test",
                     choices=["test", "val"],
                     help="Which split to evaluate on")
    gen.add_argument("--data-dir",      default="simult_mt/data/filtered")
    gen.add_argument("--output-dir",    default="simult_mt/results/predictions")
    gen.add_argument("--batch-size",    type=int, default=1,
                     help="Generation batch size (1 recommended for correctness)")
    gen.add_argument("--max-new-tokens",type=int, default=120)
    gen.add_argument("--max-samples",   type=int, default=None,
                     help="Cap number of test samples (default: use all)")
    gen.add_argument("--run-name",      default=None,
                     help="Tag for this run (default: auto timestamp)")

    # -- score ---------------------------------------------------------------
    scr = sub.add_parser("score", help="Load saved predictions, compute all metrics")
    scr.add_argument("--predictions-dir", required=True)
    scr.add_argument("--output-dir",      default="simult_mt/results/tables")
    scr.add_argument("--no-comet",        action="store_true",
                     help="Skip COMET (slow, requires model download)")

    # -- compare -------------------------------------------------------------
    cmp = sub.add_parser("compare", help="Compare multiple prediction directories")
    cmp.add_argument("--dirs",      nargs="+", required=True)
    cmp.add_argument("--labels",    nargs="+", default=None,
                     help="Human-readable labels for each dir (default: dir names)")
    cmp.add_argument("--output-dir",default="simult_mt/results/tables")

    return p


# ---------------------------------------------------------------------------
# Latency metrics (computed analytically from wait-k formula)
# ---------------------------------------------------------------------------

def compute_latency_metrics(source_len: int, hyp_len: int, k) -> dict:
    """
    Compute Average Proportion (AP), Average Lagging (AL), and
    Differentiable AL (DAL) for wait-k policy.

    For wait-k, the number of source tokens read when writing target token t
    (1-indexed) is:
        g(t) = min(k + (t - 1), source_len)

    References:
        Ma et al. (2019) Ã¢â‚¬â€ "STACL: Simultaneous Translation with Implicit
        Anticipation and Controllable Latency using Prefix-to-Prefix Framework"
        
        Arivazhagan et al. (2020) Ã¢â‚¬â€ "Monotonic Infinite Lookback Attention"
    """
    S = source_len
    T = hyp_len

    if S == 0 or T == 0:
        return {"AP": None, "AL": None, "DAL": None}

    # k = "full" means full-attention baseline (reads all source before writing)
    if k == "full":
        k_int = S
    else:
        k_int = int(k)

    # g(t) for t = 1 ... T
    g = [min(k_int + (t - 1), S) for t in range(1, T + 1)]

    # Average Proportion
    AP = sum(gt / S for gt in g) / T

    # Average Lagging (Ma et al. 2019)
    # AL = (1 / Ãâ€ž(S)) * ÃŽÂ£_{t=1}^{Ãâ€ž(S)} [g(t) - (t-1) * S/T]
    # where Ãâ€ž(S) = first t where g(t) = S
    tau_S = next((t for t, gt in enumerate(g, 1) if gt == S), T)
    if tau_S == 0:
        AL = 0.0
    else:
        al_sum = sum(g[t-1] - (t - 1) * S / T for t in range(1, tau_S + 1))
        AL = al_sum / tau_S

    # Differentiable AL (Cherry & Foster 2019 variant)
    # Same as AL but sums to T (no early stopping at Ãâ€ž(S))
    dal_terms = [g[t-1] - (t - 1) * S / T for t in range(1, T + 1)]
    DAL = sum(max(d, 0) for d in dal_terms) / T

    return {"AP": round(AP, 4), "AL": round(AL, 4), "DAL": round(DAL, 4)}


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def score_sacrebleu(hypotheses: list[str], references: list[str]) -> dict:
    import sacrebleu
    bleu   = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf   = sacrebleu.corpus_chrf(hypotheses, [references])
    ter    = sacrebleu.corpus_ter(hypotheses,  [references])
    return {
        "BLEU":        round(bleu.score, 2),
        "BLEU_bp":     round(bleu.bp,    4),
        "chrF":        round(chrf.score, 2),
        "TER":         round(ter.score,  2),
    }


def score_comet(sources: list[str], hypotheses: list[str],
                references: list[str]) -> dict:
    """
    Compute COMET score using Unbabel/wmt22-comet-da.
    Returns corpus-level score and list of sentence-level scores.
    """
    from comet import download_model, load_from_checkpoint

    print("    Loading COMET model (wmt22-comet-da) ...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    data = [{"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)]

    output = comet_model.predict(data, batch_size=32, gpus=0)
    return {
        "COMET_corpus":   round(float(output.system_score), 4),
        "COMET_per_sent": [round(float(x), 4) for x in output.scores],
    }


def score_length_stats(hypotheses: list[str], references: list[str]) -> dict:
    """Basic length ratio and coverage stats."""
    hyp_lens = [len(h.split()) for h in hypotheses]
    ref_lens  = [len(r.split()) for r in references]
    empty     = sum(1 for h in hypotheses if not h.strip())

    avg_hyp = sum(hyp_lens) / max(len(hyp_lens), 1)
    avg_ref = sum(ref_lens)  / max(len(ref_lens),  1)

    return {
        "avg_hyp_words":  round(avg_hyp, 2),
        "avg_ref_words":  round(avg_ref, 2),
        "length_ratio":   round(avg_hyp / max(avg_ref, 1), 4),
        "empty_outputs":  empty,
        "n_sentences":    len(hypotheses),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_model_for_eval(model_path: str):
    """
    Load the fine-tuned model + tokenizer for inference.
    If model_path points to a LoRA checkpoint, load with PEFT.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cuda_ok = torch.cuda.is_available()

    # Check if this is a PEFT checkpoint
    peft_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft = os.path.exists(peft_config_path)

    if cuda_ok:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path if not is_peft else _get_base_model_name(model_path),
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path if not is_peft else _get_base_model_name(model_path),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    if is_peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()   # merge LoRA into base for faster inference
        print("    LoRA merged into base model for inference.")
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def _get_base_model_name(peft_checkpoint: str) -> str:
    cfg_path = os.path.join(peft_checkpoint, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("base_model_name_or_path", "sarvamai/sarvam-translate")


def format_prompt(tokenizer, source_text: str, tgt_lang: str = "Telugu") -> str:
    """Format English source into Gemma 3's chat template."""
    msgs = [
        {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
        {"role": "user",   "content": source_text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def compute_source_offsets(tokenizer, source_text: str, tgt_lang: str = "Telugu") -> tuple[int, int, int]:
    """Return (source_start, source_end, target_start) for the wait-k mask."""
    sys_only = [{"role": "system", "content": f"Translate the text below to {tgt_lang}."}]
    with_src  = [
        {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
        {"role": "user",   "content": source_text},
    ]

    prompt_only_str = tokenizer.apply_chat_template(sys_only,     tokenize=False, add_generation_prompt=False)
    prompt_src_str  = tokenizer.apply_chat_template(with_src,     tokenize=False, add_generation_prompt=False)
    prompt_sep_str  = tokenizer.apply_chat_template(with_src,     tokenize=False, add_generation_prompt=True)

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    return len(enc(prompt_only_str)), len(enc(prompt_src_str)), len(enc(prompt_sep_str))


def generate_one(model, tokenizer, source_text: str, k,
                 max_new_tokens: int = 120, device: str = "cuda") -> str:
    """
    Generate a Telugu translation with wait-k attention masking.
    Single-sample autoregressive generation.
    """
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import build_batch_waitk_mask, register_waitk_hooks, remove_hooks

    prompt_str = format_prompt(tokenizer, source_text)
    source_start, source_end, target_start = compute_source_offsets(tokenizer, source_text)

    input_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(device)

    eos_id    = tokenizer.eos_token_id
    generated = input_ids
    dtype     = torch.float16 if device != "cpu" else torch.float32

    mask_cell = [None]

    with torch.no_grad():
        for step in range(max_new_tokens):
            seq_len      = generated.shape[1]
            target_step  = seq_len - target_start   # how many target tokens written so far

            mask_cell[0] = build_batch_waitk_mask(
                source_starts=[source_start],
                source_ends=[source_end],
                target_starts=[target_start],
                seq_len=seq_len,
                k=k if k != "full" else seq_len,
                dtype=dtype,
                device=device,
            )
            handles = register_waitk_hooks(model, mask_cell)

            attn_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
            try:
                out = model(input_ids=generated, attention_mask=attn_mask)
            finally:
                remove_hooks(handles)
                mask_cell[0] = None

            next_token = out.logits[:, -1, :].argmax(dim=-1)
            if next_token.item() == eos_id:
                break
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    gen_tokens = generated[0, target_start:].tolist()
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Phase 1: Generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    import torch

    cuda_ok = torch.cuda.is_available()
    device  = "cuda" if cuda_ok else "cpu"

    if not cuda_ok:
        print("WARNING: CUDA not available. Generation will be very slow on CPU.")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, run_name)
    os.makedirs(out_root, exist_ok=True)

    # Load test data
    split_path = os.path.join(args.data_dir, f"{args.split}.json")
    samples = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Evaluating on {len(samples)} samples from {args.split} split.")

    # Save manifest
    manifest = {
        "run_name":   run_name,
        "model_path": args.model_path,
        "split":      args.split,
        "n_samples":  len(samples),
        "k_values":   args.k,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Load model once
    print(f"\nLoading model from: {args.model_path}")
    model, tokenizer = load_model_for_eval(args.model_path)

    k_values_parsed = []
    for kv in args.k:
        if kv == "full":
            k_values_parsed.append("full")
        else:
            k_values_parsed.append(int(kv))

    for k in k_values_parsed:
        k_tag = f"k{k}" if k != "full" else "full"
        pred_dir = os.path.join(out_root, k_tag)
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(pred_dir, "predictions.jsonl")

        print(f"\n  Generating with k={k}  Ã¢â€ â€™  {pred_path}")

        with open(pred_path, "w", encoding="utf-8") as fout:
            for i, sample in enumerate(samples):
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(samples)} ...")
                try:
                    hyp = generate_one(
                        model, tokenizer,
                        source_text=sample["source"],
                        k=k,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )
                except Exception as e:
                    print(f"    WARN: sample {i} failed: {e}")
                    hyp = ""

                src_len = len(tokenizer.encode(sample["source"], add_special_tokens=False))
                ref_len = len(tokenizer.encode(sample["target"], add_special_tokens=False))
                hyp_len = len(tokenizer.encode(hyp,             add_special_tokens=False))

                lat = compute_latency_metrics(src_len, hyp_len, k)

                record = {
                    "id":         sample["id"],
                    "source":     sample["source"],
                    "reference":  sample["target"],
                    "hypothesis": hyp,
                    "k":          str(k),
                    "src_len":    src_len,
                    "ref_len":    ref_len,
                    "hyp_len":    hyp_len,
                    "AP":         lat["AP"],
                    "AL":         lat["AL"],
                    "DAL":        lat["DAL"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"    Done  Ã¢â€ â€™  {pred_path}")

    print(f"\nAll predictions saved to: {out_root}")
    print(f"Run name: {run_name}")
    print(f"\nTo score:\n  python simult_mt/src/eval.py score --predictions-dir {out_root}")


# ---------------------------------------------------------------------------
# Phase 2: Score
# ---------------------------------------------------------------------------

def load_predictions(pred_dir: str) -> list[dict]:
    pred_path = os.path.join(pred_dir, "predictions.jsonl")
    records = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_all_metrics(records: list[dict], run_comet: bool = True) -> dict:
    sources    = [r["source"]    for r in records]
    references = [r["reference"] for r in records]
    hypotheses = [r["hypothesis"] for r in records]

    # --- latency (pre-computed per sentence, average here) ---
    ap_vals  = [r["AP"]  for r in records if r["AP"]  is not None]
    al_vals  = [r["AL"]  for r in records if r["AL"]  is not None]
    dal_vals = [r["DAL"] for r in records if r["DAL"] is not None]

    latency = {
        "AP_mean":  round(sum(ap_vals)  / max(len(ap_vals),  1), 4),
        "AL_mean":  round(sum(al_vals)  / max(len(al_vals),  1), 4),
        "DAL_mean": round(sum(dal_vals) / max(len(dal_vals), 1), 4),
    }

    # --- quality ---
    quality = score_sacrebleu(hypotheses, references)
    quality.update(score_length_stats(hypotheses, references))

    # --- COMET ---
    comet_results = {}
    if run_comet:
        try:
            comet_results = score_comet(sources, hypotheses, references)
        except Exception as e:
            print(f"    COMET failed: {e}")
            comet_results = {"COMET_corpus": None, "COMET_per_sent": []}

    return {**quality, **latency, **comet_results}


def cmd_score(args):
    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(args.predictions_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Find all k subdirectories
    k_dirs = sorted([
        d for d in Path(args.predictions_dir).iterdir()
        if d.is_dir() and (d / "predictions.jsonl").exists()
    ])

    if not k_dirs:
        print(f"No prediction directories found in {args.predictions_dir}")
        sys.exit(1)

    all_results = {}
    per_sentence_comet = {}

    for k_dir in k_dirs:
        k_tag = k_dir.name
        print(f"\n  Scoring {k_tag} ...")
        records = load_predictions(str(k_dir))
        metrics = compute_all_metrics(records, run_comet=not args.no_comet)

        # Store per-sentence COMET for later analysis
        if "COMET_per_sent" in metrics and metrics["COMET_per_sent"]:
            per_sentence_comet[k_tag] = metrics.pop("COMET_per_sent")
        else:
            metrics.pop("COMET_per_sent", None)

        all_results[k_tag] = metrics

        print(f"    BLEU={metrics.get('BLEU', 'N/A')}  "
              f"chrF={metrics.get('chrF', 'N/A')}  "
              f"TER={metrics.get('TER', 'N/A')}  "
              f"COMET={metrics.get('COMET_corpus', 'N/A')}  "
              f"AL={metrics.get('AL_mean', 'N/A')}  "
              f"AP={metrics.get('AP_mean', 'N/A')}")

    # Save full results JSON
    results_path = os.path.join(args.output_dir, "metrics.json")
    full_output = {
        "manifest": manifest,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "results": all_results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results Ã¢â€ â€™ {results_path}")

    # Save per-sentence COMET scores separately
    if per_sentence_comet:
        comet_path = os.path.join(args.output_dir, "comet_per_sentence.json")
        with open(comet_path, "w") as f:
            json.dump(per_sentence_comet, f, indent=2)
        print(f"  Per-sentence COMET Ã¢â€ â€™ {comet_path}")

    # Save markdown summary table
    _save_markdown_table(all_results, manifest, args.output_dir)

    # Save CSV for plotting
    _save_csv(all_results, args.output_dir)

    # Plot latency-quality tradeoff
    _plot_latency_quality(all_results, args.output_dir)


def _save_markdown_table(results: dict, manifest: dict, output_dir: str):
    """Save a clean markdown summary table."""
    cols = ["k", "BLEU", "chrF", "TER", "COMET_corpus",
            "AP_mean", "AL_mean", "DAL_mean",
            "length_ratio", "empty_outputs"]

    rows = []
    for k_tag, m in sorted(results.items()):
        row = [k_tag] + [str(m.get(c, "Ã¢â‚¬â€")) for c in cols[1:]]
        rows.append(row)

    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join([":---:"] * len(cols)) + "|"
    body   = "\n".join("| " + " | ".join(r) + " |" for r in rows)

    model_info = manifest.get("model_path", "unknown")
    split_info = manifest.get("split", "unknown")
    n_samples  = manifest.get("n_samples", "?")

    table_md = f"""# Evaluation Results

**Model:** `{model_info}`  
**Split:** {split_info} ({n_samples} samples)  
**Scored at:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Quality Ãƒâ€” Latency Summary

{header}
{sep}
{body}

## Column Definitions

| Column | Description |
|:---|:---|
| BLEU | SacreBLEU corpus score (tokenize=13a) |
| chrF | Character n-gram F-score (n=6, ÃŽÂ²=2) |
| TER | Translation Edit Rate (lower = better) |
| COMET_corpus | Unbabel/wmt22-comet-da neural metric (higher = better) |
| AP | Average Proportion Ã¢â‚¬â€ fraction of source read per target token |
| AL | Average Lagging (Ma et al. 2019) Ã¢â‚¬â€ lag behind ideal simultaneous |
| DAL | Differentiable AL (Cherry & Foster 2019 variant) |
| length_ratio | avg hypothesis words / avg reference words |
| empty_outputs | number of empty/failed translations |

## Latency Metric Definitions

For wait-k policy with source length S and hypothesis length T:
- **g(t)** = min(k + t Ã¢Ë†â€™ 1, S)  Ã¢â€ Â  source tokens read when writing target token t
- **AP** = (1/T) ÃŽÂ£ g(t)/S
- **AL** = (1/Ã â€ž(S)) ÃŽÂ£_{{t=1}}^{{Ã â€ž(S)}} [g(t) Ã¢Ë†â€™ (tÃ¢Ë†â€™1)Ã‚Â·S/T]  where Ã â€ž(S) = first t where g(t) = S
- **DAL** = (1/T) ÃŽÂ£ max(g(t) Ã¢Ë†â€™ (tÃ¢Ë†â€™1)Ã‚Â·S/T, 0)
"""
    md_path = os.path.join(output_dir, "results_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(table_md)
    print(f"  Markdown table Ã¢â€ â€™ {md_path}")


def _save_csv(results: dict, output_dir: str):
    import csv
    csv_path = os.path.join(output_dir, "results.csv")
    if not results:
        return
    fields = ["k"] + list(next(iter(results.values())).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for k_tag, m in sorted(results.items()):
            writer.writerow({"k": k_tag, **m})
    print(f"  CSV Ã¢â€ â€™ {csv_path}")


def _plot_latency_quality(results: dict, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k_tags = sorted(results.keys())
        al_vals   = [results[k].get("AL_mean")      for k in k_tags]
        bleu_vals = [results[k].get("BLEU")          for k in k_tags]
        comet_vals= [results[k].get("COMET_corpus")  for k in k_tags]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("LatencyÃ¢â‚¬â€œQuality Tradeoff (English Ã¢â€ â€™ Telugu SiMT)", fontsize=14)

        for ax, y_vals, y_label in [
            (axes[0], bleu_vals, "SacreBLEU"),
            (axes[1], comet_vals, "COMET"),
        ]:
            valid = [(al, y) for al, y in zip(al_vals, y_vals) if al is not None and y is not None]
            if not valid:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "o-", color="#2563eb", linewidth=2, markersize=8)
            for k_tag, x, y in zip(k_tags, xs, ys):
                ax.annotate(k_tag, (x, y), textcoords="offset points",
                            xytext=(6, 4), fontsize=10)
            ax.set_xlabel("Average Lagging (AL) Ã¢â€ â€œ", fontsize=12)
            ax.set_ylabel(f"{y_label} Ã¢â€ â€˜", fontsize=12)
            ax.set_title(f"{y_label} vs. Latency", fontsize=12)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "latency_quality_tradeoff.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot Ã¢â€ â€™ {plot_path}")
    except Exception as e:
        print(f"  Could not generate plot: {e}")


# ---------------------------------------------------------------------------
# Phase 3: Compare multiple runs
# ---------------------------------------------------------------------------

def cmd_compare(args):
    os.makedirs(args.output_dir, exist_ok=True)
    labels = args.labels or [os.path.basename(d.rstrip("/\\")) for d in args.dirs]

    all_runs = {}
    for label, d in zip(labels, args.dirs):
        results_path = os.path.join(d, "metrics.json")  # from previous score run
        if not os.path.exists(results_path):
            # Try scoring on the fly
            print(f"  No metrics.json found in {d}, skipping.")
            continue
        with open(results_path) as f:
            data = json.load(f)
        all_runs[label] = data.get("results", {})

    if not all_runs:
        print("No scored runs found.")
        sys.exit(1)

    # Flatten into comparison table
    rows = []
    for run_label, run_results in all_runs.items():
        for k_tag, metrics in sorted(run_results.items()):
            rows.append({
                "run":   run_label,
                "k":     k_tag,
                "BLEU":  metrics.get("BLEU"),
                "chrF":  metrics.get("chrF"),
                "COMET": metrics.get("COMET_corpus"),
                "AL":    metrics.get("AL_mean"),
                "AP":    metrics.get("AP_mean"),
            })

    cmp_path = os.path.join(args.output_dir, "comparison.json")
    with open(cmp_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Comparison saved Ã¢â€ â€™ {cmp_path}")

    # Print table
    print("\nComparison:")
    print(f"{'Run':<25} {'k':<6} {'BLEU':>6} {'chrF':>6} {'COMET':>7} {'AL':>6} {'AP':>6}")
    print("-" * 70)
    for r in rows:
        print(f"{r['run']:<25} {r['k']:<6} "
              f"{str(r['BLEU']):>6} {str(r['chrF']):>6} "
              f"{str(r['COMET']):>7} {str(r['AL']):>6} {str(r['AP']):>6}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()


