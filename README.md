# English → Telugu Simultaneous Machine Translation (wait-k)

Simultaneous machine translation (SiMT) produces the target translation while the
source sentence is still arriving, rather than waiting for the full sentence. This
repository fine-tunes **[`sarvamai/sarvam-translate`](https://huggingface.co/sarvamai/sarvam-translate)**
(Gemma 3 4B) for English → Telugu under a **wait-k** policy, using LoRA and an
attention-masking controller that constrains how much of the source each target
token may attend to.

**Fine-tuned model:** [`praneet3/sarvam-translate-waitk-simulmt`](https://huggingface.co/praneet3/sarvam-translate-waitk-simulmt)

English → Telugu is a deliberately hard direction. English is SVO and Telugu is SOV,
so the verb — the most information-dense token — arrives last in Telugu but the model
often has to commit to it early. Telugu is also agglutinative and expands by ~1.76×
in token count over the English source, which compounds the latency penalty.

---

## Repository layout

```
simult_mt/
├── src/
│   ├── data_pipeline.py     # download BPCC + IN22, filter, split, write JSON
│   ├── data_utils.py        # SiMTDataset, collate_fn, token-offset computation
│   ├── masking.py           # WaitKMaskController — wait-k attention bias + hooks
│   ├── train.py             # LoRA multi-anchor wait-k fine-tuning (+ dry run)
│   ├── eval.py              # generate predictions + score + compare
│   └── setup_and_verify.py  # one-time CUDA / model-load check
├── data/filtered/           # train.json / val.json / test.json (+ stats)
└── results/
    ├── predictions/         # per-run model outputs (one folder per run)
    ├── tables/              # metrics, CSV, markdown, tradeoff plot
    └── plots/               # architecture + preprocessing figures

simt_v2/                     # alternative fine-tuning + eval design (see its README)
masking_logic_check.py       # standard-library proof that the wait-k mask is correct
```

---

## Method

### Data

Training data is `ai4bharat/BPCC` (`bpcc-seed-latest`, `tel_Telu`), passed through a
five-stage token-based filter (source/target length, length ratio, Telugu-script
validity, source-side dedup) using the `sarvam-translate` tokenizer. Evaluation uses
the held-out **IN22** benchmark (`IN22-Conv` for validation, `IN22-Gen` for test),
which has no overlap with BPCC. Each example is a JSON line: `{"id", "source", "target"}`.

### Sequence layout and wait-k masking

The decoder-only model sees one flattened sequence built from Gemma 3's chat template:

```
<bos><start_of_turn>system
Translate the text below to Telugu.<end_of_turn>
<start_of_turn>user
{English source}<end_of_turn>
<start_of_turn>model
{Telugu target}<end_of_turn><eos>
```

For each example we record three token offsets — `source_start`, `source_end`,
`target_start` — by tokenizing progressively longer prefixes of the template.

Under wait-k, a target token at step `t` (0-indexed from `target_start`) may attend to
source positions `[source_start, source_start + min(k + t, S))`, where `S` is the
source length. Forbidden source positions get an additive bias of `-10000.0` before
the softmax. Prompt rows, source rows, and target self-attention stay causal and
unmasked. The constraint is applied by `WaitKMaskController` via forward pre-hooks on
the attention modules — the same mechanism at training and inference, so the two stay
consistent.

### Multi-anchor training

A model trained at a single fixed `k` is brittle at other latencies. For each batch we
sample one `k` from `{1, 2, 4, 7}` and apply the matching mask, so over training the
model sees every operating point. The base model is loaded in 4-bit NF4; LoRA
(`r=16`, `α=32`, dropout `0.05`) is attached to the attention projections. Loss is
cross-entropy on target tokens only (prompt and source labels set to `-100`).

### Metrics

- **Quality:** SacreBLEU, chrF (well suited to morphologically rich Telugu), TER, and
  optional neural COMET (`Unbabel/wmt22-comet-da`).
- **Latency** (computed analytically from the wait-k schedule, with
  `g(t) = min(k + t − 1, S)`): Average Proportion (AP), Average Lagging (AL, Ma et al.
  2019), and Differentiable AL (DAL, Cherry & Foster 2019).

A correct run shows quality and latency both rising with `k`: `k=1` is the
lowest-latency / lowest-quality point and `full` (offline, full attention) is the
upper bound.

---

## A note on the evaluation fix

Earlier evaluation runs were broken in two ways, both now fixed in
[`masking.py`](simult_mt/src/masking.py) and [`eval.py`](simult_mt/src/eval.py):

1. **Every k produced identical output.** The wait-k mask was handed to
   `model.generate()`, which rebuilds its own causal mask and ignored it — so all `k`
   silently ran at full attention and there was no latency–quality tradeoff.
2. **Empty outputs.** A later attempt built a fixed-size mask that did not match the
   live attention shape during cached decoding, and dropped eager attention (SDPA's
   causal fast-path discards the mask entirely).

The controller now derives the wait-k bias from the *live* attention-mask shape on
every forward pass, so it is correct for both the prefill step and each KV-cached
decode step, and evaluation forces `attn_implementation="eager"`. The masking math is
verified against the training-time convention by `masking_logic_check.py` (standard
library only, no GPU):

```bash
python masking_logic_check.py
```

---

## Running from scratch

> **Environment.** Fine-tuning and generation need a CUDA GPU plus `torch`,
> `transformers`, `peft`, and `bitsandbytes`. In this project those run inside a
> GPU container, not on the host — run the GPU steps wherever the model and CUDA live.

### 0. Setup

```bash
python -m venv simt_env && source simt_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40.0" peft bitsandbytes datasets sentencepiece \
            unbabel-comet sacrebleu numpy pandas tqdm scipy matplotlib

huggingface-cli login          # sarvam-translate + BPCC/IN22 are gated
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 1. Data (skip if `simult_mt/data/filtered/test.json` already exists)

```bash
python simult_mt/src/data_pipeline.py
```

### 2. Training (optional — the fine-tuned model is already on the Hub)

```bash
# fast pipeline check: 10 samples, 2 optimizer steps, CPU-friendly
python simult_mt/src/train.py --dry-run

# full LoRA fine-tune
python simult_mt/src/train.py \
    --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 \
    --k-values 1,2,4,7 \
    --output-dir simult_mt/experiments/waitk_static
```

### 3. Evaluation

A single `generate` call now also scores the run and writes all tables and the plot
into `<run_dir>/tables/`, so one command produces every output.

```bash
# quick check on 100 test samples
python simult_mt/src/eval.py generate \
    --model-path praneet3/sarvam-translate-waitk-simulmt \
    --k 1 2 4 7 full --split test --max-samples 100 \
    --output-dir simult_mt/results/predictions

# full test set: --max-samples 0   (add --no-comet-gen to skip neural COMET)
```

**Confirm the run is healthy** — hypotheses must be non-empty *and* `k1` must differ
from `full`:

```bash
RUN=simult_mt/results/predictions/<TIMESTAMP>
python - <<'PY'
import json, os
run = os.environ["RUN"]
hyps = lambda k: [json.loads(l)["hypothesis"] for l in open(f"{run}/{k}/predictions.jsonl", encoding="utf-8") if l.strip()]
k1, full = hyps("k1"), hyps("full")
print("k1 non-empty:", sum(bool(h.strip()) for h in k1), "/", len(k1))
print("k1 identical to full:", sum(a == b for a, b in zip(k1, full)), "/", len(k1), "(should be low)")
PY
```

Re-score any run without touching the GPU:

```bash
python simult_mt/src/eval.py score \
    --predictions-dir simult_mt/results/predictions/<TIMESTAMP> \
    --output-dir simult_mt/results/tables          # add --no-comet to skip COMET
```

### Outputs

Each run directory contains:

```
results/predictions/<TIMESTAMP>/
├── manifest.json                 # model, split, k values, timestamp
├── k1/ k2/ k4/ k7/ full/
│   └── predictions.jsonl         # source, reference, hypothesis, AL/AP/DAL per line
└── tables/
    ├── metrics.json              # all metrics per k (machine-readable)
    ├── results_table.md          # human-readable summary
    ├── results.csv               # for plotting
    ├── comet_per_sentence.json   # per-sentence COMET (if computed)
    └── latency_quality_tradeoff.png
```

---

## Alternative design (`simt_v2/`)

`simt_v2/` is a cleaner reimplementation of the training and evaluation: it passes an
explicit 4D causal + wait-k mask directly to the model (no forward hooks), assigns a
**per-sample** wait-k anchor within each batch for denser multi-anchor coverage, and
shares one masking module between training and evaluation so they cannot drift. See
[`simt_v2/README.md`](simt_v2/README.md).
