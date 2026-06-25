# Evaluation Results

**Model:** `simult_mt/experiments/waitk_static/epoch_1`  
**Split:** test (1024 samples)  
**Scored at:** 2026-06-25 05:05

## Quality Ãƒâ€” Latency Summary

| k | BLEU | chrF | TER | COMET_corpus | AP_mean | AL_mean | DAL_mean | length_ratio | empty_outputs |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | 17.56 | 55.27 | 70.64 | 0.8746 | 1.0 | 33.1016 | 16.8472 | 0.9378 | 0 |
| k1 | 17.56 | 55.27 | 70.64 | 0.8746 | 0.715 | 7.1933 | 7.0264 | 0.9378 | 0 |
| k2 | 17.56 | 55.27 | 70.64 | 0.8746 | 0.7366 | 7.9894 | 7.5936 | 0.9378 | 0 |
| k4 | 17.56 | 55.27 | 70.64 | 0.8746 | 0.7763 | 9.5816 | 8.6609 | 0.9378 | 0 |
| k7 | 17.56 | 55.27 | 70.64 | 0.8746 | 0.8275 | 11.9672 | 10.0927 | 0.9378 | 0 |

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
- **AL** = (1/Ã â€ž(S)) ÃŽÂ£_{t=1}^{Ã â€ž(S)} [g(t) Ã¢Ë†â€™ (tÃ¢Ë†â€™1)Ã‚Â·S/T]  where Ã â€ž(S) = first t where g(t) = S
- **DAL** = (1/T) ÃŽÂ£ max(g(t) Ã¢Ë†â€™ (tÃ¢Ë†â€™1)Ã‚Â·S/T, 0)
