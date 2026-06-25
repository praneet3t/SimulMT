# simt_v2 — a cleaner wait-k fine-tuning + evaluation design

This is an alternative implementation of the English → Telugu wait-k SiMT pipeline.
It targets the same model (`sarvamai/sarvam-translate`, Gemma 3 4B) and the same data,
but changes three things that made the original brittle.

## What is different and why

1. **Explicit 4D masks instead of forward hooks.**
   The original injects the wait-k bias by registering forward pre-hooks on every
   attention module and mutating the `attention_mask` argument in flight. That is
   exactly where the evaluation broke: the hook's mask shape did not match the live
   attention shape during cached decoding, and `model.generate()` ignored custom
   masks entirely. Here, [`masking.py`](masking.py) builds one `[B, 1, L, L]` additive
   mask (causal + wait-k + padding) and passes it straight to the model forward. No
   hooks, nothing to leak, and no dependence on `generate()` respecting the mask.

2. **The same masking code runs at train and eval time.**
   `build_batch_mask` (training) and `build_window_mask` (incremental decoding) share
   the identical wait-k rule, so the training policy and the inference policy cannot
   drift apart — the failure mode behind the original "all k look the same" bug.

3. **Per-sample multi-anchor coverage.**
   The original samples one `k` per *batch*. Here every *example* in a batch is
   assigned its own `k` from the anchor set `{1, 2, 4, 7}`, so each optimizer step
   trains several latency regimes at once. This is denser supervision at the same
   compute, and is the standard multi-path idea (Elbayad et al. 2020, *Efficient
   Wait-k Models for Simultaneous Machine Translation*).

Both implementations require `attn_implementation="eager"`, which is what lets an
explicit 4D additive mask reach the attention scores unchanged.

## Files

| File | Role |
|---|---|
| `masking.py` | `build_batch_mask` (training) + `build_window_mask` (decoding); `python masking.py` runs a self-test |
| `data.py`    | `SiMTDataset` + collate; computes `source_start / source_end / target_start` offsets |
| `train.py`   | LoRA 4-bit fine-tuning with per-sample wait-k anchors |
| `eval.py`    | greedy wait-k decoding + scoring; one command writes predictions, tables, and the tradeoff plot |

## Running

Uses the same environment and data as the root project (see the main
[`README.md`](../README.md)). Run on a machine with a CUDA GPU and the ML stack.

```bash
# 0. sanity-check the masking math (no GPU)
python simt_v2/masking.py

# 1. validate the training pipeline (CPU, 10 samples, 2 steps)
python simt_v2/train.py --dry-run

# 2. fine-tune
python simt_v2/train.py \
    --train simult_mt/data/filtered/train.json \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
    --k-values 1,2,4,7 --output-dir simt_v2/checkpoints

# 3. evaluate — a single call generates AND scores, saving everything under the run dir
python simt_v2/eval.py generate \
    --model-path simt_v2/checkpoints/epoch_3 \
    --k 1 2 4 7 full --split test --max-samples 100 \
    --output-dir simt_v2/results
# (or --model-path praneet3/sarvam-translate-waitk-simulmt to eval the published model)

# re-score without the GPU; add --comet for the neural metric
python simt_v2/eval.py score --run-dir simt_v2/results/<TIMESTAMP>
```

### Outputs

```
simt_v2/results/<TIMESTAMP>/
├── manifest.json
├── k1/ k2/ k4/ k7/ full/predictions.jsonl   # source, reference, hypothesis, AL/AP/DAL
└── tables/
    ├── metrics.json          # BLEU / chrF / TER / (COMET) / AP / AL / DAL per k
    ├── results_table.md
    ├── results.csv
    └── tradeoff.png          # BLEU vs. Average Lagging
```

A healthy run has non-empty hypotheses and a curve where BLEU and AL both rise with
`k` (`k=1` lowest latency/quality, `full` the offline upper bound). If you ever
suspect the mask is being dropped, re-run `generate` with `--no-cache`, which
recomputes the full mask every step (slower, but leaves no room for cache-shape
ambiguity).
