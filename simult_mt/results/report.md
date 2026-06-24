# Simultaneous Machine Translation â€” English â†’ Telugu

**Model:** `sarvamai/sarvam-translate` (Gemma 3 4B Instruction-Tuned)  
**Data:** AI4Bharat BPCC Â· Seed-Latest split  
**Approach:** Wait-k static policy + LoRA fine-tuning

---

## What This Is

Standard machine translation reads the full source sentence before producing any output. Simultaneous Machine Translation (SiMT) works under a stricter constraint â€” the model must start generating the target while the source is still arriving, token by token.

We use the **wait-k** policy: read the first `k` source tokens, then alternate one source read and one target write. The tradeoff is unavoidable â€” more waiting means better quality, less waiting means lower latency.

**Why English â†’ Telugu is the hard direction.**  
Telugu is SOV (Subject-Object-Verb). English is SVO. This matters for SiMT because the verb is the most information-dense token in both languages, and in Telugu it comes last. When the model is translating simultaneously, it often has to commit to a Telugu verb before it has seen the English verb â€” because the Telugu sentence ends before the English one does. This is the genuinely difficult direction for wait-k policy research. Telugu also expands relative to English: on average, a Telugu translation uses ~1.6Ã— more tokens than its English source. That expansion compounds the latency problem.

---

## Dataset

**Source:** `ai4bharat/BPCC`, config `bpcc-seed-latest`, split `tel_Telu`

BPCC (Bharat Parallel Corpus Collection) is a large-scale multilingual parallel corpus maintained by AI4Bharat. The seed portion is a higher-quality curated subset with human-verified sentence pairs across 22 Indian languages.

| Field | Content |
|:------|:--------|
| `src` | English (Latin script) â†’ **source side** |
| `tgt` | Telugu (Telugu script) â†’ **target side** |
| Raw pairs downloaded | 98,117 |

---

## Preprocessing

Five sequential filters applied using the `sarvamai/sarvam-translate` tokenizer (SentencePiece). Each filter stage feeds the next. All tokenization is done in a single pass over the raw corpus before any filtering begins.

### Filtering Funnel

| Stage | Pairs Remaining | Removed | Rule |
|:---|:---:|:---:|:---|
| Raw corpus | 98,117 | â€” | Initial BPCC download |
| Rule 1 â€” English source length [4, 60] | 98,017 | 100 | Dropped 100 very long English sentences (max was 135 tokens) |
| Rule 2 â€” Telugu target length [5, 80] | 97,520 | 497 | Dropped 497 very long Telugu translations (max was 221 tokens) |
| Rule 3 â€” Ratio tel/eng [0.5, 5.0] | 97,520 | **0** | Nothing removed â€” actual ratio range is [0.68, 4.36], entirely within bounds |
| Rule 4 â€” Script validity | 96,291 | 1,229 | Sentences with too many numerals/Latin chars mixed in |
| Rule 5 â€” Deduplication (English source) | **95,074** | 1,217 | Duplicate English source sentences removed |

Full funnel also saved to `data/filtered/filtering_funnel.txt`.

### Filter Logic

**Rule 1 â€” English source length [4, 60]**  
Lower bound: anything below 4 tokens is noise (a word, punctuation, a stray number). Upper bound: very long English sentences don't just cost memory â€” they stretch the wait-k window. If a source is 80 tokens long, a model with k=4 is operating at ~5% of full source context when it writes its first Telugu token. Capping at 60 keeps the effective k-to-length ratio in a range where the model can actually learn useful patterns.

**Rule 2 â€” Telugu target length [5, 80]**  
The target cap is looser because Telugu genuinely expands. A 40-token English sentence often becomes a 60-token Telugu one. Capping Telugu at 60 (the old value, incorrectly borrowed from a wrong-direction pipeline) would have silently dropped valid long translations.

**Rule 3 â€” Ratio tel/eng [0.5, 5.0]**  
We compute `tel_tokens / eng_tokens` â€” target over source, which is the expansion factor. The distribution (plotted in `data/filtered/ratio_distribution.png`) is centered around 1.6, with the 5th percentile near 0.7 and the 95th near 3.5. The bounds [0.5, 5.0] are deliberately wider than the bulk of the data:

- Below 0.5: Telugu is suspiciously shorter than English â€” almost always a misaligned or truncated pair.
- Above 5.0: Telugu is more than 5Ã— longer than English â€” usually a multi-sentence Telugu translation paired with a single English sentence, or garbage on the Telugu side.

The old pipeline computed `eng/tel` (source/target), which had backwards semantics for this direction. That's corrected here.

**Rule 4 â€” Telugu script [â‰¥80% in U+0C00â€“U+0C7F]**  
The Telugu Unicode block is U+0C00 to U+0C7F. Any string labeled as Telugu that has less than 80% of its non-whitespace characters in this range is wrong â€” Romanized, transliterated, or mislabeled. Five example pairs removed by this filter are printed at pipeline runtime so you can spot-check that it's catching bad data, not valid entries.

**Rule 5 â€” Deduplication on English source only**  
Remove pairs where the English source sentence has appeared before (keep first occurrence). We do **not** deduplicate on the Telugu side â€” the same English sentence translated differently into Telugu is valid training data and shouldn't be discarded. Source dedup prevents the model from memorising repeated inputs and from the same source sentence leaking into multiple splits.

---

## Data Splits

From the final clean pool, shuffled with seed 42:

| Split | Pairs | Purpose |
|:---|:---:|:---|
| Test | 1,000 | Set aside first, not touched until final evaluation |
| Validation | 2,000 | In-training loss monitoring |
| **Train** | **92,074** | All remaining clean pairs |

The old pipeline capped training at 50,000 and discarded ~22,000 clean pairs for no reason. All 95,074 clean pairs are now used.

Zero overlap verified on the **English source side** across all three splits.

---

## Analysis

Token length statistics on the training split (92,074 pairs, measured post-filter):

| Side | Min | Max | Mean | Median |
|:---|:---:|:---:|:---:|:---:|
| English (source) | 5 | 60 | 20.47 | 19.0 |
| Telugu (target) | 8 | 80 | 35.29 | 33.0 |
| Ratio tel/eng | 0.68 | 4.36 | **1.76** | **1.72** |

Also notable from the raw-corpus distribution (before any filtering):
- English p5=11, p90=32, p95=36
- Telugu p5=19, p90=55, p95=62
- Ratio p5=1.24, p95=2.40 â€” tightly clustered, hence the ratio filter removed **zero pairs**

**Expansion factor:** Telugu uses on average **1.76Ã— more tokens** than the corresponding English source (median 1.72). This is `tel/eng` â€” target over source.

**Why the expansion matters for wait-k design:**  
At the median (English=19, Telugu=33), k=4 means the model writes its first Telugu token having read only 4/19 â‰ˆ 21% of the English source. The model is always writing ahead of what it has fully seen. The tighter the ratio distribution (p5=1.24, p95=2.40 from the raw data), the more predictable this expansion is â€” which is useful: a model that learns the Telugu expansion pattern can plan ahead even with limited source context.

### Sample Pairs (Train)

| English | Telugu |
|:---|:---|
| The more money you put in the account, the more interest you will make. | à°®à±€à°°à± à°–à°¾à°¤à°¾à°²à±‹ à°Žà°‚à°¤ à°Žà°•à±à°•à±à°µ à°¡à°¬à±à°¬à±à°²à± à°µà±‡à°¸à±à°¤à±‡ à°®à±€à°•à± à°…à°‚à°¤ à°Žà°•à±à°•à±à°µ à°µà°¡à±à°¡à±€ à°µà°¸à±à°¤à±à°‚à°¦à°¿. |
| The soils of the arid region are generally sandy to sandy-loam in texture. | à°ˆ à°¶à±à°·à±à°• à°ªà±à°°à°¾à°‚à°¤à°‚ à°¯à±Šà°•à±à°• à°®à°Ÿà±à°Ÿà°¿ à°¸à°¾à°§à°¾à°°à°£à°‚à°—à°¾ à°‡à°¸à±à°•à°®à°¯à°‚ à°¨à±à°‚à°¡à°¿ à°‡à°¸à±à°•-à°’à°‚à°¡à±à°°à± à°‰à°ªà°°à°¿à°¤à°²à°‚ à°•à°²à°¿à°—à°¿ à°‰à°‚à°Ÿà±à°‚à°¦à°¿. |
| In the individual events two athletes of each nation participated. | à°µà±à°¯à°•à±à°¤à°¿à°—à°¤ à°ˆà°µà±†à°‚à°Ÿà±à°²à°²à±‹ à°ªà±à°°à°¤à°¿ à°¦à±‡à°¶à°‚ à°¨à±à°‚à°¡à°¿ à°‡à°¦à±à°¦à°°à± à°…à°¥à±à°²à±†à°Ÿà±à°²à± à°ªà°¾à°²à±à°—à±Šà°¨à±à°¨à°¾à°°à±. |
| The head of government is an indirectly elected Chief Minister who is vested with most of the executive powers. | à°ªà±à°°à°­à±à°¤à±à°µ à°…à°§à°¿à°ªà°¤à°¿ à°ªà°°à±‹à°•à±à°·à°‚à°—à°¾ à°Žà°¨à±à°¨à±à°•à±‹à°¬à°¡à°¿à°¨ à°®à±à°–à±à°¯à°®à°‚à°¤à±à°°à°¿, à°†à°¯à°¨ à°…à°¨à±‡à°• à°•à°¾à°°à±à°¯à°¨à°¿à°°à±à°µà°¾à°¹à°• à°…à°§à°¿à°•à°¾à°°à°¾à°²à± à°•à°²à°¿à°—à°¿ à°‰à°‚à°Ÿà°¾à°¡à±. |

---

## Methodology

### Model

`sarvamai/sarvam-translate` is a 4-billion parameter decoder-only language model from Google DeepMind, instruction-tuned on a multilingual corpus. It uses grouped-query attention (GQA) with a mix of local sliding-window and global attention layers, and a 256k-token context window. The model has 42 transformer layers, hidden dim 2560, and a vocabulary optimised for multilingual text including Indic scripts.

Loaded in **4-bit NF4 quantization** (BitsAndBytesConfig) to fit single-GPU training:
- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.float16`
- `bnb_4bit_use_double_quant=True`

**LoRA** targets only `q_proj` and `v_proj` inside each self-attention layer â€” no MLP, no embeddings. Config: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`. This gives ~3â€“5M trainable parameters out of 4B total.

### Input Format

Everything is flattened into one token sequence:

```
[system prompt] [English source] [separator] [Telugu target] [EOS]
```

Using Gemma 3's built-in chat template via `tokenizer.apply_chat_template()`:

```
<bos><start_of_turn>system
Translate the text below to Telugu.<end_of_turn>
<start_of_turn>user
{English sentence}<end_of_turn>
<start_of_turn>model
{Telugu translation}<end_of_turn><eos>
```

Token offsets `source_start`, `source_end`, `target_start` are computed per sample at dataset time by tokenizing progressively longer prefixes. These offsets are passed through the DataLoader and consumed by the masking module at training time.

### Wait-k Attention Masking

Under normal causal attention every token sees all previous tokens. Wait-k restricts what the **Telugu target** can see from the **English source**:

```
Telugu token t=0  â†’  attends to English source[0 : k]
Telugu token t=1  â†’  attends to English source[0 : k+1]
Telugu token t=2  â†’  attends to English source[0 : k+2]
```

This is enforced by adding an additive bias of `âˆ’10000.0` to forbidden attention positions. The bias is added to the 4D attention mask `[B, 1, seq_len, seq_len]` via **forward pre-hooks** injected into every attention layer. Hooks are registered per-batch, applied during the forward pass, and removed immediately after to avoid any state leaking across batches.

The mask is constructed efficiently without loops over individual positions:

```python
t_idx       = arange(ts, seq_len) - ts          # [T]
max_visible = clamp(k + t_idx, max=src_len)      # [T]
src_off     = arange(src_len)                    # [S]
should_mask = src_off[None, :] >= max_visible[:, None]   # [T, S]
```

**Unit test (17-token synthetic sequence, k=2, 5 prompt + 6 English + 6 Telugu):**
- Row 11 (t=0): sees positions 5,6 only âœ“
- Row 12 (t=1): sees positions 5,6,7 only âœ“  
- Row 13 (t=2): sees positions 5,6,7,8 only âœ“
- Prompt rows 0â€“4: unchanged âœ“
- English source rows 5â€“10: unchanged âœ“

### Training â€” Multi-Anchor Wait-k

Training with a single fixed `k` produces a model that is brittle to unseen k values at inference. Instead, each batch randomly samples `k` from `{1, 2, 4, 7}`. Over training, the model sees all four operating points and learns representations that generalise across the quality-latency curve.

Loss is standard cross-entropy on target tokens only (all prompt + source positions have label `âˆ’100`). Gradient checkpointing is enabled. Gradient clipping at `max_norm=1.0`. Effective batch size = `batch_size Ã— grad_accum = 4 Ã— 4 = 16`.

### Evaluation

| Metric | What it measures | Target |
|:---|:---|:---:|
| COMET (primary) | Neural quality score, correlates with human judgments | â‰¥ 0.55 |
| SacreBLEU | n-gram precision, standard for comparability | â€” |
| Average Proportion (AP) | How much source is consumed per target token; lower = more simultaneous | â€” |

---

## Architecture

![System Architecture](c:\Users\apran\Videos\Cin\LIBRARY\SimulMT\simult_mt\results\plots\architecture.png)

---

## Evaluation

### What gets saved

After generation, every prediction is saved to a JSONL file so metrics can be recomputed without re-running the model:

```
results/predictions/{run_name}/
â”œâ”€â”€ manifest.json              â† model path, split, timestamp, k values
â”œâ”€â”€ k1/predictions.jsonl       â† one line per sample: source, reference, hypothesis, AL, AP, DAL
â”œâ”€â”€ k2/predictions.jsonl
â”œâ”€â”€ k4/predictions.jsonl
â”œâ”€â”€ k7/predictions.jsonl
â””â”€â”€ full/predictions.jsonl     â† full-attention baseline

results/tables/
â”œâ”€â”€ metrics.json               â† all metric values (machine-readable)
â”œâ”€â”€ results_table.md           â† formatted summary table
â”œâ”€â”€ results.csv                â† same, for plotting
â”œâ”€â”€ comet_per_sentence.json    â† sentence-level COMET scores per k
â””â”€â”€ latency_quality_tradeoff.png
```

### Metrics

**Quality:**

| Metric | Tool | Notes |
|:---|:---|:---|
| BLEU | SacreBLEU (tokenize=13a) | Corpus-level n-gram precision with brevity penalty |
| chrF | SacreBLEU | Character n-gram F-score â€” better for morphologically rich languages like Telugu |
| TER | SacreBLEU | Translation Edit Rate â€” edit distance normalised by ref length; lower = better |
| COMET | Unbabel/wmt22-comet-da | Neural metric trained on human judgments; best correlation with human evals |

**Latency** (computed analytically from wait-k, no timing hardware needed):

For source length S, hypothesis length T, and wait-k policy:  
`g(t)` = source tokens read when writing target token `t` = `min(k + t âˆ’ 1, S)`

| Metric | Formula | What it means |
|:---|:---|:---|
| AP | (1/T) Î£ g(t)/S | Average fraction of source consumed per target token. k=1 â†’ ~0.55, full â†’ ~1.0 |
| AL | (1/Ï„) Î£ [g(t) âˆ’ (tâˆ’1)Â·S/T] | Lag behind ideal simultaneous (Ma et al. 2019). Lower = more simultaneous |
| DAL | (1/T) Î£ max(g(t) âˆ’ (tâˆ’1)Â·S/T, 0) | AL variant without early stopping (Cherry & Foster 2019) |

### Expected result shape

The tradeoff curve looks like this across k values:

| k | Latency (AL) | Quality (BLEU) | Notes |
|:---:|:---:|:---:|:---|
| 1 | lowest | lowest | Commits to Telugu output after just 1 English word |
| 2 | â†“ | â†“ | |
| 4 | â†‘ | â†‘ | Practical sweet spot for most SiMT systems |
| 7 | â†‘â†‘ | â†‘â†‘ | Near-offline quality with modest latency |
| full | highest | highest | Full-attention baseline â€” upper bound on quality |

A good fine-tuned model should show the curve bending right (more quality gain per unit latency) compared to a baseline model that hasn't been trained with wait-k constraints.

### Running evaluation

```powershell
# Phase 1 â€” generate predictions (needs GPU + fine-tuned model)
python simult_mt/src/eval.py generate `
    --model-path simult_mt/experiments/waitk_static/epoch_3 `
    --k 1 2 4 7 full `
    --split test `
    --output-dir simult_mt/results/predictions

# Phase 2 â€” score all metrics from saved files (CPU, fast, repeatable)
python simult_mt/src/eval.py score `
    --predictions-dir simult_mt/results/predictions/{run_name} `
    --output-dir simult_mt/results/tables

# Re-score without re-running the model (e.g., to add COMET after skipping it)
python simult_mt/src/eval.py score `
    --predictions-dir simult_mt/results/predictions/{run_name} `
    --output-dir simult_mt/results/tables

# Compare two fine-tuning runs
python simult_mt/src/eval.py compare `
    --dirs simult_mt/results/predictions/run1 simult_mt/results/predictions/run2 `
    --labels "baseline" "waitk_finetuned" `
    --output-dir simult_mt/results/tables
```

---

## How to Run

### 1 â€” Environment setup (once)

```powershell
python -m venv simt_env
simt_env\Scripts\activate

# Install PyTorch â€” match cu121 to your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install "transformers>=4.40.0" peft bitsandbytes datasets `
            sentencepiece unbabel-comet sacrebleu numpy pandas `
            tqdm scipy matplotlib

# Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 2 â€” Re-run data pipeline (if needed)

Regenerates `train.json`, `val.json`, `test.json`, `stats.json`, `ratio_distribution.png`, `filtering_funnel.txt`:

```powershell
cd c:\Users\apran\Videos\Cin\LIBRARY\SimulMT
python simult_mt/src/data_pipeline.py
```

### 3 â€” Dry run (first, always â€” proves the pipeline works)

Loads 10 samples, runs 2 optimizer steps, prints loss, exits with PASS/FAIL:

```powershell
python simult_mt/src/train.py --dry-run
```

Expected output ends with:
```
DRY RUN PASSED
  Model loading (4-bit / float32): OK
  LoRA attachment (q_proj, v_proj): OK
  DataLoader + collate_fn:          OK
  Wait-k batch mask construction:   OK
  Hook injection + removal:         OK
  Forward + backward pass:          OK
  Loss finite:                      OK
```

### 4 â€” Full training

```powershell
python simult_mt/src/train.py `
    --epochs 3 `
    --batch-size 4 `
    --grad-accum 4 `
    --lr 2e-4 `
    --k-values 1,2,4,7 `
    --output-dir simult_mt/experiments/waitk_static
```

Checkpoints saved to `simult_mt/experiments/waitk_static/epoch_1/`, `epoch_2/`, `epoch_3/`.  
Intermediate checkpoints every 500 steps if `--save-every 500` is passed.

---

## Project Structure

```
simult_mt/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_pipeline.py      # Download BPCC â†’ filter â†’ split â†’ stats
â”‚   â”œâ”€â”€ data_utils.py         # SiMTDataset, collate_fn, DataLoader
â”‚   â”œâ”€â”€ masking.py            # WaitKMaskController (single + batched masks)
â”‚   â”œâ”€â”€ train.py              # LoRA training loop + dry-run mode
â”‚   â””â”€â”€ setup_and_verify.py   # One-time env check (CUDA, model load)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                  # train.eng, train.tel (downloaded text)
â”‚   â””â”€â”€ filtered/
â”‚       â”œâ”€â”€ train.json        # ~69k pairs (all clean data)
â”‚       â”œâ”€â”€ val.json          # 2,000 pairs
â”‚       â”œâ”€â”€ test.json         # 1,000 pairs (held out until final eval)
â”‚       â”œâ”€â”€ stats.json        # Token length stats on train split
â”‚       â”œâ”€â”€ filtering_funnel.txt
â”‚       â””â”€â”€ ratio_distribution.png
â”œâ”€â”€ experiments/
â”‚   â”œâ”€â”€ baseline/             # Full-attention (k=âˆž) baseline
â”‚   â”œâ”€â”€ waitk_static/         # Wait-k checkpoints
â”‚   â””â”€â”€ adaptive_gate/        # Reserved
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ tables/
â”‚   â””â”€â”€ plots/
â””â”€â”€ configs/
```

---

## Status

| Component | Status |
|:---|:---:|
| Environment setup | Done |
| Data download (BPCC) | Done |
| Preprocessing pipeline (5 rules, corrected direction) | Done |
| Splits â€” zero overlap, all clean data used (92,074 train) | Done |
| Token length & ratio analysis | Done |
| Wait-k mask controller (single + batched) | Done |
| SiMTDataset + DataLoader | Done |
| train.py â€” dry run + full training | Done |
| eval.py â€” generate + score + compare | Done |
| Architecture diagram | Done |
| Dry run on GPU | Pending (GPU needed) |
| Full 3-epoch training | Pending |
| COMET / SacreBLEU / AL evaluation | Pending |

