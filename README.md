# English-to-Telugu Simultaneous Machine Translation (SiMT)

This repository implements a Simultaneous Machine Translation (SiMT) pipeline using a wait-$k$ static policy applied to **Gemma 3 (sarvamai/sarvam-translate)** via parameter-efficient fine-tuning (LoRA) and a custom attention masking controller.

---

## 1. Problem Formulation & Translation Direction

Simultaneous Machine Translation (SiMT) requires the model to start generating target tokens before the entire source sentence has arrived. We implement the **English $\to$ Telugu** translation direction, which is highly challenging due to structural divergence:

*   **Word Order Mismatch (SVO $\to$ SOV):** English is Subject-Object-Verb (SVO), while Telugu is Subject-Object-Verb (SOV). Since the verb is the most information-dense token and Telugu puts it last, the model must commit to a Telugu verb before it has seen the English verb. This makes translation highly dependent on the model's anticipation capabilities.
*   **Target Length Expansion:** Telugu is morphologically rich and agglutinative, expanding by an average of **1.76Ã—** in token length relative to the English source. This expansion compounds the latency penalty.

---

## 2. Preprocessing & Data Funnel

We train on the `ai4bharat/BPCC` (Bharat Parallel Corpus Collection) dataset, specifically the human-verified `bpcc-seed-latest` split (`tel_Telu`). 

To produce high-quality training pairs, we apply a strict 5-stage token-based preprocessing funnel (using the `sarvamai/sarvam-translate` SentencePiece tokenizer):

| Stage | Clean Pairs Remaining | Removed | Filtering Rule |
| :--- | :---: | :---: | :--- |
| **0. Raw Corpus** | 98,117 | â€” | Raw seed dataset download |
| **1. Source Length** | 98,017 | 100 | English source tokens must be in $[4, 60]$. Prevents short noise and extremely long sequences. |
| **2. Target Length** | 97,520 | 497 | Telugu target tokens must be in $[5, 80]$. Accommodates Telugu's natural expansion. |
| **3. Length Ratio** | 97,520 | 0 | Token ratio $\frac{\text{Telugu}}{\text{English}}$ must be in $[0.5, 5.0]$. Catches misaligned sentences. |
| **4. Script Validity** | 96,291 | 1,229 | $\ge 80\%$ of non-whitespace characters must fall in the Telugu Unicode block `[U+0C00â€“U+0C7F]`. |
| **5. Deduplication** | **95,074** | 1,217 | Deduplicates identical English source sentences (keeps first occurrence). |

### Dataset Splits
From the final clean pool (95,074 pairs), we shuffle using seed 42 and partition:
*   **Test Set:** 1,000 pairs (held out)
*   **Validation Set:** 2,000 pairs
*   **Train Set:** 92,074 pairs (all remaining clean pairs; zero source-side overlap with val/test)

---

## 3. Architecture & Wait-k Masking

![SiMT Wait-k Architecture](simult_mt/results/plots/architecture.png)

### Sequence Layout
The decoder-only model processes a flattened sequence formatted via Gemma 3's chat template:
```
<bos><start_of_turn>system
Translate the text below to Telugu.<end_of_turn>
<start_of_turn>user
{English source sentence}<end_of_turn>
<start_of_turn>model
{Telugu target translation}<end_of_turn><eos>
```

### Wait-k Masking Math
Under a wait-$k$ static policy, the model reads the first $k$ source tokens, then alternates between reading one source token and generating one target token. 

For a sequence with:
*   $\text{source\_start} = ss$
*   $\text{source\_end} = se$ (where source length $S = se - ss$)
*   $\text{target\_start} = ts$

A target token generated at step $t$ (where $t = i - ts$ for token index $i \ge ts$) is only allowed to attend to source tokens up to index:
$$\text{max\_visible}(t) = \min(k + t, S)$$

Any attention query at target index $i$ to source index $j$ where $(j - ss) \ge \text{max\_visible}(t)$ is blocked by adding a bias of $-10,000.0$ to the attention logit before softmax. Self-attention on the prompt, the English source, and generated Telugu target tokens remains causal and unmasked.

### PyTorch Pre-Hook Implementation
The masking is dynamically enforced during forward passes using PyTorch **forward pre-hooks** registered on all attention modules (specifically matching `"attn"` or `"attention"` classes). These hooks modify the `attention_mask` argument on the fly:

```python
# Batch-vectorized computation of wait-k mask
t_idx       = torch.arange(ts, seq_len, device=device) - ts      # [T]
max_visible = torch.clamp(k + t_idx, max=src_len)                # [T]
src_off     = torch.arange(src_len, device=device)               # [S]

# Create should_mask matrix of shape [T, S]
should_mask = src_off.unsqueeze(0) >= max_visible.unsqueeze(1)
mask[batch, 0, ts:, ss:se] = torch.where(should_mask, -10000.0, 0.0)
```

Pre-hooks are attached per-batch during training/inference and cleanly unregistered to prevent state leakage or memory buildup.

---

## 4. Multi-Anchor Training

Training a model with a single fixed $k$ makes it brittle to other latency regimes. We use **Multi-Anchor Training**:
*   For each batch, a wait-k step $k$ is sampled uniformly from $\{1, 2, 4, 7\}$.
*   The corresponding wait-k attention mask is applied to that batch.
*   **LoRA Adaptation:** To fit on a single GPU, the model is loaded in 4-bit NormalFloat (NF4) quantization. LoRA is attached to `q_proj` and `v_proj` layers across all attention blocks (rank $r=16$, $\alpha=32$, dropout $0.05$).
*   **Loss Function:** Standard Cross-Entropy computed strictly on target (Telugu) tokens. Prompt and English source tokens are ignored by masking their target label with $-100$.

---

## 5. Evaluation Protocol

We use a decoupled, two-phase evaluation framework:

### Phase 1: Generation (GPU-based)
Runs the model on the test split for $k \in \{1, 2, 4, 7, \text{full}\}$ (where `full` represents standard offline generation). For each sample, the script autoregressively generates the translation under the wait-$k$ attention constraints. All outputs are saved to `results/predictions/{run_name}/k*/predictions.jsonl`, with pre-computed sentence-level latency metrics:
1.  **Average Proportion (AP):** Measures the average fraction of the source sentence read when generating each target token:
    $$\text{AP} = \frac{1}{T} \sum_{t=1}^T \frac{g(t)}{S}$$
2.  **Average Lagging (AL):** Measures the token lag behind an ideal simultaneous translator (Ma et al. 2019):
    $$\text{AL} = \frac{1}{\tau_s} \sum_{t=1}^{\tau_s} \left( g(t) - \frac{t-1}{S/T} \right)$$
    where $\tau_s$ is the first step where $g(t) = S$.
3.  **Differentiable AL (DAL):** A smoother variation without early stopping (Cherry & Foster 2019):
    $$\text{DAL} = \frac{1}{T} \sum_{t=1}^T \max\left( g(t) - \frac{t-1}{S/T}, 0 \right)$$

### Phase 2: Scoring (CPU-based)
Loads the saved predictions and computes corpus-level quality metrics:
*   **SacreBLEU:** Standard n-gram precision.
*   **chrF:** Character-level F-score (highly correlated with human judgments for morphologically rich languages like Telugu).
*   **TER:** Translation Edit Rate (lower is better).
*   **COMET:** Neural metric using `wmt22-comet-da` (trained on human direct assessments).
*   **Tradeoff Curves:** Generates BLEU vs. AL and COMET vs. AL plots to map the quality-latency frontier.

---

## 6. Reproducibility Guide

### Hardware Requirements
| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB (4-bit quant) | 24 GB |
| System RAM | 32 GB | 64 GB |
| Disk (model + data + ckpts) | 40 GB | 80 GB |
| CUDA | 12.1+ | 12.4 |

Training 3 epochs on 92k pairs with batch 4, grad-accum 4 on an A100-40G takes approximately **4–6 hours**.

---

### Step 0 — Environment Setup

```powershell
# Create and activate virtual environment
python -m venv simt_env
simt_env\Scripts\activate           # Windows
# source simt_env/bin/activate      # Linux / macOS

# PyTorch with CUDA 12.1 (adjust cu121 to match your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# All project dependencies
pip install "transformers>=4.40.0" peft bitsandbytes datasets sentencepiece `
            unbabel-comet sacrebleu numpy pandas tqdm scipy matplotlib

# Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

> **HuggingFace auth:** `sarvamai/sarvam-translate` is a gated model.
> Run `huggingface-cli login` and accept the model's usage agreement at
> https://huggingface.co/sarvamai/sarvam-translate before running any script.

---

### Step 1 — Data Preprocessing

Downloads `ai4bharat/BPCC` (bpcc-seed-latest, tel_Telu), applies the 5-stage
filtering funnel, and writes `train.json`, `val.json`, `test.json` to
`simult_mt/data/filtered/`.

```powershell
python simult_mt/src/data_pipeline.py
```

Expected output after the funnel: **~95,074** clean pairs split into
92,074 / 2,000 / 1,000.

---

### Step 2 — Pipeline Verification (Dry Run)

Loads 10 samples, runs exactly 2 optimizer steps, validates the entire
training pipeline (model loading, LoRA, masking, forward/backward, loss),
then exits.  
**No GPU required** (runs on CPU in float32, takes 5–15 min).

```powershell
python simult_mt/src/train.py --dry-run
```

Expected final output:
```
================================================================
DRY RUN PASSED
  Model loading (4-bit / float32): OK
  LoRA attachment (q_proj, v_proj): OK
  DataLoader + collate_fn:          OK
  Wait-k batch mask construction:   OK
  Hook injection + removal:         OK
  Forward + backward pass:          OK
  Loss finite:                      OK
  tqdm progress bar + ETA:          OK
  Auto-eval wiring (--auto-eval):   OK
================================================================
```

---

### Step 3 — Full LoRA Fine-tuning

```powershell
python simult_mt/src/train.py `
    --epochs 3 `
    --batch-size 4 `
    --grad-accum 4 `
    --lr 2e-4 `
    --k-values 1,2,4,7 `
    --output-dir simult_mt/experiments/waitk_static `
    --log-every 50 `
    --save-every 500
```

The progress bar shows live `loss`, `avg`, `k`, `step`, and `ETA` per batch.
Epoch-end validation loss is printed and checkpoints are saved to
`simult_mt/experiments/waitk_static/epoch_{N}/`.

**Recommended — training + auto-eval in one command:**

```powershell
python simult_mt/src/train.py `
    --epochs 3 `
    --batch-size 4 `
    --grad-accum 4 `
    --lr 2e-4 `
    --k-values 1,2,4,7 `
    --output-dir simult_mt/experiments/waitk_static `
    --auto-eval `
    --eval-k-values 1,2,4,7,full `
    --eval-split test
```

`--auto-eval` triggers Phase 1 (generation) and Phase 2 (scoring) automatically
on the final checkpoint as soon as training completes.

---

### Step 4 — Stand-alone Evaluation (optional, if not using --auto-eval)

**Phase 1 — Generate predictions** (GPU required):

```powershell
python simult_mt/src/eval.py generate `
    --model-path simult_mt/experiments/waitk_static/epoch_3 `
    --k 1 2 4 7 full `
    --split test `
    --output-dir simult_mt/results/predictions
```

**Phase 2 — Score from saved predictions** (CPU only, fast, repeatable):

```powershell
python simult_mt/src/eval.py score `
    --predictions-dir simult_mt/results/predictions/<run_name> `
    --output-dir simult_mt/results/tables
```

**Ablation comparison across multiple runs:**

```powershell
python simult_mt/src/eval.py compare `
    --dirs simult_mt/results/predictions/run_k1only `
           simult_mt/results/predictions/run_multianchor `
    --labels "k=1 only" "Multi-anchor k={1,2,4,7}" `
    --output-dir simult_mt/results/tables/ablation
```

---

### Step 5 — Evaluation Outputs

After scoring, the following files are written to `simult_mt/results/tables/`:

| File | Contents |
|---|---|
| `metrics.json` | All BLEU / chrF / TER / COMET / AL / DAL / AP values per k — **machine-readable JSON for ablation scripts** |
| `results_table.md` | Human-readable markdown table |
| `results.csv` | CSV for plotting with pandas/matplotlib |
| `comet_per_sentence.json` | Per-sentence COMET scores for significance testing |
| `latency_quality_tradeoff.png` | BLEU vs. AL and COMET vs. AL frontier plot |

---

### Quick Command Reference

| Action | Command |
|---|---|
| Environment setup | `pip install ...` (see Step 0) |
| Data preprocessing | `python simult_mt/src/data_pipeline.py` |
| Dry run | `python simult_mt/src/train.py --dry-run` |
| Full training | `python simult_mt/src/train.py --epochs 3 --batch-size 4 --grad-accum 4 --lr 2e-4 --k-values 1,2,4,7 --output-dir simult_mt/experiments/waitk_static` |
| Training + auto-eval | Add `--auto-eval` to the training command |
| Generate predictions only | `python simult_mt/src/eval.py generate --model-path ... --k 1 2 4 7 full` |
| Score existing predictions | `python simult_mt/src/eval.py score --predictions-dir ...` |
| Ablation comparison | `python simult_mt/src/eval.py compare --dirs run1 run2 --labels ...` |

