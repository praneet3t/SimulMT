# Evaluation Results

**Model:** `sarvamai/sarvam-translate`  
**Split:** test (100 samples)  
**Scored at:** 2026-06-25 17:27

## Quality Ãƒâ€” Latency Summary

| k | BLEU | chrF++ | TER | COMET_corpus | AP_mean | AL_mean | DAL_mean | length_ratio | empty_outputs |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | 10.95 | 45.16 | 82.06 | 0.8673 | 1.0 | 31.31 | 15.9559 | 1.0257 | 0 |
| k2 | 2.49 | 21.72 | 101.87 | 0.678 | 0.5923 | 1.5244 | 3.7663 | 0.7622 | 0 |
| k4 | 2.79 | 23.3 | 102.36 | 0.6923 | 0.6576 | 3.7758 | 5.1344 | 0.8131 | 0 |
| k7 | 3.91 | 30.16 | 99.79 | 0.7473 | 0.7584 | 8.3673 | 7.6002 | 0.9154 | 0 |

## Column Definitions

| Column | Description |
|:---|:---|
| BLEU | SacreBLEU corpus score (tokenize=13a) |
| chrF++ | chrF with word_order=2 (character + word n-grams, higher = better) |
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
- **AL** = (1/Ã â€ž(S)) ÃŽÂ£_{t=1}^{Ã â€ž(S)} [g(t) Ã¢Ë†â€™ (tÃ¢Ë†â€™1)Ã‚Â·S/T]  where Ã â€ž(S) = first t where g(t) = S
- **DAL** = (1/T) ÃŽÂ£ max(g(t) Ã¢Ë†â€™ (tÃ¢Ë†â€™1)Ã‚Â·S/T, 0)
