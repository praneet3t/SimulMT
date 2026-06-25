# Evaluation Results

**Model:** `praneet3/sarvam-translate-waitk-simulmt`  
**Split:** test (100 samples)  
**Scored at:** 2026-06-25 10:52

## Quality Ãƒâ€” Latency Summary

| k | BLEU | chrF | TER | COMET_corpus | AP_mean | AL_mean | DAL_mean | length_ratio | empty_outputs |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| full | 8.22 | 35.85 | 114.41 | Ã¢â‚¬â€ | 1.0 | 31.31 | 15.9926 | 1.3101 | 0 |
| k1 | 8.22 | 35.85 | 114.41 | Ã¢â‚¬â€ | 0.6759 | 5.4612 | 5.3651 | 1.3101 | 0 |
| k2 | 8.22 | 35.85 | 114.41 | Ã¢â‚¬â€ | 0.7006 | 6.2948 | 5.9739 | 1.3101 | 0 |
| k4 | 8.22 | 35.85 | 114.41 | Ã¢â‚¬â€ | 0.7466 | 7.9625 | 7.1556 | 1.3101 | 0 |
| k7 | 8.22 | 35.85 | 114.41 | Ã¢â‚¬â€ | 0.8066 | 10.4684 | 8.7825 | 1.3101 | 0 |

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
