#!/usr/bin/env python3
"""
Evaluation for English -> Telugu SiMT (v2).

Generation enforces wait-k with the SAME explicit 4D mask builder used in training
(masking.build_window_mask) via a small greedy decode loop -- no forward hooks, and
no reliance on model.generate() honouring a custom mask. A single `generate` call
writes predictions AND scores them, saving everything under the run directory:

    <run>/manifest.json
    <run>/{k1,k2,k4,k7,full}/predictions.jsonl
    <run>/tables/metrics.json | results_table.md | results.csv | tradeoff.png

Usage:
    python simt_v2/eval.py generate \
        --model-path praneet3/sarvam-translate-waitk-simulmt \
        --k 1 2 4 7 full --split test --max-samples 100

    python simt_v2/eval.py score --run-dir simt_v2/results/<TIMESTAMP>
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masking import build_window_mask
from data import compute_offsets, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Latency metrics (analytic, from the wait-k schedule g(t) = min(k+t-1, S))
# ---------------------------------------------------------------------------

def latency_metrics(S, T, k):
    if S == 0 or T == 0:
        return {"AP": None, "AL": None, "DAL": None}
    k_int = S if k == "full" else int(k)
    g = [min(k_int + (t - 1), S) for t in range(1, T + 1)]
    AP = sum(gt / S for gt in g) / T
    tau = next((t for t, gt in enumerate(g, 1) if gt == S), T)
    AL = sum(g[t - 1] - (t - 1) * S / T for t in range(1, tau + 1)) / max(tau, 1)
    DAL = sum(max(g[t - 1] - (t - 1) * S / T, 0) for t in range(1, T + 1)) / T
    return {"AP": round(AP, 4), "AL": round(AL, 4), "DAL": round(DAL, 4)}


# ---------------------------------------------------------------------------
# Model loading + generation
# ---------------------------------------------------------------------------

def load_model(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    base = model_path
    if is_peft:
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            base = json.load(f).get("base_model_name_or_path", "sarvamai/sarvam-translate")

    if torch.cuda.is_available():
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, device_map="auto", attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager")

    if is_peft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path).merge_and_unload()

    model.eval()
    return model, tok


def generate_one(model, tok, source, k, max_new_tokens, device, use_cache=True):
    import torch

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": source}],
        tokenize=False, add_generation_prompt=True)
    input_ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)
    P = input_ids.shape[1]
    ss, se, ts = compute_offsets(tok, source)
    eos = tok.eos_token_id

    generated, past, cur = [], None, input_ids
    with torch.no_grad():
        for step in range(max_new_tokens):
            if use_cache and past is not None:
                q_pos  = torch.tensor([P + step - 1], device=device)
                mask   = build_window_mask(q_pos, P + step, ss, se, ts, k, device=device)
                out    = model(input_ids=cur, attention_mask=mask,
                               past_key_values=past, use_cache=True)
            else:
                cur_len = cur.shape[1]
                q_pos   = torch.arange(cur_len, device=device)
                mask    = build_window_mask(q_pos, cur_len, ss, se, ts, k, device=device)
                out     = model(input_ids=cur, attention_mask=mask, use_cache=use_cache)

            past = out.past_key_values if use_cache else None
            nxt  = int(out.logits[0, -1, :].argmax())
            if nxt == eos:
                break
            generated.append(nxt)
            cur = torch.tensor([[nxt]], device=device) if use_cache else \
                  torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)

    return tok.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — generation will be slow.")

    run = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run)
    os.makedirs(run_dir, exist_ok=True)

    samples = [json.loads(l) for l in open(
        os.path.join(args.data_dir, f"{args.split}.json"), encoding="utf-8") if l.strip()]
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"Evaluating {len(samples)} {args.split} samples.")

    json.dump({"run": run, "model_path": args.model_path, "split": args.split,
               "n_samples": len(samples), "k_values": args.k,
               "timestamp": datetime.now(timezone.utc).isoformat()},
              open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)

    print(f"Loading model: {args.model_path}")
    model, tok = load_model(args.model_path)

    ks = [kv if kv == "full" else int(kv) for kv in args.k]
    for k in ks:
        tag = "full" if k == "full" else f"k{k}"
        kdir = os.path.join(run_dir, tag)
        os.makedirs(kdir, exist_ok=True)
        print(f"\n  generating {tag} ...")
        with open(os.path.join(kdir, "predictions.jsonl"), "w", encoding="utf-8") as fo:
            for i, s in enumerate(samples):
                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(samples)}")
                try:
                    hyp = generate_one(model, tok, s["source"], k,
                                       args.max_new_tokens, device, use_cache=not args.no_cache)
                except Exception:
                    traceback.print_exc()
                    hyp = ""
                S = len(tok.encode(s["source"], add_special_tokens=False))
                T = len(tok.encode(hyp, add_special_tokens=False))
                rec = {"id": s["id"], "source": s["source"], "reference": s["target"],
                       "hypothesis": hyp, "k": str(k), "src_len": S, "hyp_len": T,
                       **latency_metrics(S, T, k)}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nPredictions -> {run_dir}")
    if not args.no_score:
        score_run(run_dir, run_comet=not args.no_comet_gen)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def _corpus_quality(hyps, refs):
    import sacrebleu
    empty = sum(1 for h in hyps if not h.strip())
    return {
        "BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrF": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
        "TER":  round(sacrebleu.corpus_ter(hyps, [refs]).score, 2),
        "empty_outputs": empty,
        "n": len(hyps),
    }


def _comet(srcs, hyps, refs):
    from comet import download_model, load_from_checkpoint
    m = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return round(float(m.predict(data, batch_size=32, gpus=0).system_score), 4)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def score_run(run_dir, run_comet=False):
    from pathlib import Path
    tables = os.path.join(run_dir, "tables")
    os.makedirs(tables, exist_ok=True)

    results = {}
    for kdir in sorted(Path(run_dir).iterdir()):
        pred = kdir / "predictions.jsonl"
        if not pred.exists():
            continue
        recs = [json.loads(l) for l in open(pred, encoding="utf-8") if l.strip()]
        hyps = [r["hypothesis"] for r in recs]
        refs = [r["reference"] for r in recs]
        srcs = [r["source"] for r in recs]
        m = _corpus_quality(hyps, refs)
        m["AP"] = _mean([r["AP"] for r in recs])
        m["AL"] = _mean([r["AL"] for r in recs])
        m["DAL"] = _mean([r["DAL"] for r in recs])
        if run_comet:
            try:
                m["COMET"] = _comet(srcs, hyps, refs)
            except Exception as e:
                print(f"    COMET failed: {e}")
                m["COMET"] = None
        results[kdir.name] = m
        print(f"  {kdir.name}: BLEU={m['BLEU']} chrF={m['chrF']} TER={m['TER']} "
              f"AL={m['AL']} empty={m['empty_outputs']}")

    json.dump(results, open(os.path.join(tables, "metrics.json"), "w"),
              indent=2, ensure_ascii=False)
    _write_table(results, tables)
    _write_csv(results, tables)
    _plot(results, tables)
    print(f"\nTables + plot -> {tables}")


def _write_table(results, tables):
    cols = ["BLEU", "chrF", "TER", "COMET", "AP", "AL", "DAL", "empty_outputs"]
    lines = ["# Evaluation Results (v2)\n",
             "| k | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for k in sorted(results):
        lines.append("| " + k + " | " +
                     " | ".join(str(results[k].get(c, "-")) for c in cols) + " |")
    open(os.path.join(tables, "results_table.md"), "w").write("\n".join(lines) + "\n")


def _write_csv(results, tables):
    import csv
    if not results:
        return
    fields = ["k"] + sorted({c for m in results.values() for c in m})
    with open(os.path.join(tables, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(results):
            w.writerow({"k": k, **results[k]})


def _plot(results, tables):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = sorted(results)
        al = [results[k].get("AL") for k in ks]
        bleu = [results[k].get("BLEU") for k in ks]
        pts = [(x, y, k) for x, y, k in zip(al, bleu, ks) if x is not None and y is not None]
        if not pts:
            return
        xs, ys, labels = zip(*pts)
        plt.figure(figsize=(6, 5))
        plt.plot(xs, ys, "o-", color="#2563eb")
        for x, y, k in pts:
            plt.annotate(k, (x, y), textcoords="offset points", xytext=(6, 4))
        plt.xlabel("Average Lagging (AL)  -> lower = lower latency")
        plt.ylabel("SacreBLEU  -> higher = better")
        plt.title("Latency-Quality Tradeoff (En->Te SiMT, v2)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(tables, "tradeoff.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  plot skipped: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="v2 SiMT evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--model-path", required=True)
    g.add_argument("--k", nargs="+", default=["1", "2", "4", "7", "full"])
    g.add_argument("--split", default="test", choices=["test", "val"])
    g.add_argument("--data-dir", default="simult_mt/data/filtered")
    g.add_argument("--output-dir", default="simt_v2/results")
    g.add_argument("--max-new-tokens", type=int, default=200)
    g.add_argument("--max-samples", type=int, default=100, help="<= 0 for all")
    g.add_argument("--run-name", default=None)
    g.add_argument("--no-cache", action="store_true",
                   help="recompute the full mask each step (slower, bulletproof)")
    g.add_argument("--no-score", action="store_true")
    g.add_argument("--no-comet-gen", action="store_true")

    s = sub.add_parser("score")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--comet", action="store_true")

    args = p.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    else:
        score_run(args.run_dir, run_comet=args.comet)


if __name__ == "__main__":
    main()
