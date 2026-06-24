import os
import sys
import json
import random
import traceback
import statistics
from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_telugu_unicode_ratio(text):
    """Return True if â‰¥80% of non-whitespace characters are in U+0C00â€“U+0C7F."""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False
    tel_count = sum(1 for c in non_space if "\u0c00" <= c <= "\u0c7f")
    return (tel_count / len(non_space)) >= 0.8


def percentile(sorted_data, p):
    """p-th percentile of a pre-sorted list."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def print_distribution(label, values, extra_percentiles=False):
    s = sorted(values)
    print(f"\n  {label}")
    print(f"    min:    {min(values)}")
    print(f"    max:    {max(values)}")
    print(f"    mean:   {statistics.mean(values):.3f}")
    print(f"    median: {statistics.median(values):.3f}")
    if extra_percentiles:
        print(f"    p5:     {percentile(s, 5):.3f}")
        print(f"    p10:    {percentile(s, 10):.3f}")
        print(f"    p90:    {percentile(s, 90):.3f}")
        print(f"    p95:    {percentile(s, 95):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw_dir      = os.path.join("simult_mt", "data", "raw")
    filtered_dir = os.path.join("simult_mt", "data", "filtered")
    os.makedirs(raw_dir,      exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1 â€” Download
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 â€” Downloading data")
    print("=" * 60)

    pairs = []

    try:
        print("Loading ai4bharat/BPCC  (bpcc-seed-latest Â· tel_Telu split)...")
        dataset = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest", split="tel_Telu")
        print(f"Loaded {len(dataset)} pairs.")
        for row in dataset:
            pairs.append({
                "source": row["src"].strip(),   # English
                "target": row["tgt"].strip(),   # Telugu
            })
    except Exception as e:
        print(f"BPCC failed: {e}")
        traceback.print_exc()

        try:
            print("\nFallback 1 â€” Helsinki-NLP/opus-100 (en-te)...")
            dd = load_dataset("Helsinki-NLP/opus-100", "en-te")
            for split in dd.keys():
                for row in dd[split]:
                    pairs.append({"source": row["translation"]["en"].strip(),
                                  "target": row["translation"]["te"].strip()})
            print(f"Fallback 1 loaded {len(pairs)} pairs.")
        except Exception as e2:
            print(f"Fallback 1 failed: {e2}")
            traceback.print_exc()

            try:
                print("\nFallback 2 â€” ai4bharat/IN22-Gen...")
                dataset = load_dataset("ai4bharat/IN22-Gen", split="test")
                for row in dataset:
                    te = row.get("sentence_te") or row.get("te")
                    en = row.get("sentence_en") or row.get("en")
                    if te and en:
                        pairs.append({"source": en.strip(), "target": te.strip()})
                print(f"Fallback 2 loaded {len(pairs)} pairs.")
            except Exception as e3:
                print(f"Fallback 2 failed: {e3}")
                traceback.print_exc()
                raise RuntimeError("All download sources failed.")

    raw_count = len(pairs)
    print(f"\nRaw corpus: {raw_count} pairs")

    # Save raw text files
    with open(os.path.join(raw_dir, "train.eng"), "w", encoding="utf-8") as fe, \
         open(os.path.join(raw_dir, "train.tel"), "w", encoding="utf-8") as ft:
        for p in pairs:
            fe.write(p["source"].replace("\n", " ").strip() + "\n")
            ft.write(p["target"].replace("\n", " ").strip() + "\n")
    print(f"Raw files saved to {raw_dir}")

    # -----------------------------------------------------------------------
    # STEP 2 â€” Tokenise everything once (expensive, do it once only)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 â€” Tokenising all pairs")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-translate")

    tokenized = []
    for i, p in enumerate(pairs):
        if i % 10000 == 0 and i > 0:
            print(f"  {i:,}/{raw_count:,} ...")
        eng_len = len(tokenizer.encode(p["source"], add_special_tokens=False))
        tel_len = len(tokenizer.encode(p["target"], add_special_tokens=False))
        tokenized.append({**p, "eng_len": eng_len, "tel_len": tel_len})

    print(f"  Done. All {raw_count:,} pairs tokenised.")

    # -----------------------------------------------------------------------
    # STEP 3 â€” Print distributions on the raw corpus
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 â€” Raw-corpus distributions")
    print("=" * 60)

    eng_raw = [t["eng_len"] for t in tokenized]
    tel_raw = [t["tel_len"] for t in tokenized]

    print_distribution("English source token lengths:", eng_raw, extra_percentiles=True)
    print_distribution("Telugu target token lengths:",  tel_raw, extra_percentiles=True)

    # Ratio: tel_tokens / eng_tokens  (target / source)
    ratios_raw = [t["tel_len"] / t["eng_len"] for t in tokenized if t["eng_len"] > 0]
    ratios_sorted = sorted(ratios_raw)

    ratio_mean   = statistics.mean(ratios_raw)
    ratio_median = statistics.median(ratios_raw)
    ratio_p5     = percentile(ratios_sorted, 5)
    ratio_p95    = percentile(ratios_sorted, 95)

    print(f"\n  Ratio  tel_tokens / eng_tokens  (raw corpus):")
    print(f"    mean:   {ratio_mean:.3f}")
    print(f"    median: {ratio_median:.3f}")
    print(f"    p5:     {ratio_p5:.3f}")
    print(f"    p95:    {ratio_p95:.3f}")
    print(f"    min:    {min(ratios_raw):.3f}")
    print(f"    max:    {max(ratios_raw):.3f}")

    # How many pairs each bound would remove (before any other filters)
    RATIO_LOW  = 0.5
    RATIO_HIGH = 5.0
    below_low  = sum(1 for r in ratios_raw if r < RATIO_LOW)
    above_high = sum(1 for r in ratios_raw if r > RATIO_HIGH)
    print(f"\n  Ratio bounds chosen:  [{RATIO_LOW}, {RATIO_HIGH}]")
    print(f"    Pairs below {RATIO_LOW}: {below_low:,}")
    print(f"    Pairs above {RATIO_HIGH}: {above_high:,}")

    # Save histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(ratios_raw, bins=30, color="#2563eb", edgecolor="white", alpha=0.85)
        ax.axvline(ratio_mean,   color="#dc2626", linestyle="--", lw=1.8,
                   label=f"Mean = {ratio_mean:.2f}")
        ax.axvline(ratio_median, color="#16a34a", linestyle="--", lw=1.8,
                   label=f"Median = {ratio_median:.2f}")
        ax.axvline(RATIO_LOW,  color="#f97316", linestyle=":", lw=1.8,
                   label=f"Lower bound = {RATIO_LOW}")
        ax.axvline(RATIO_HIGH, color="#f97316", linestyle=":", lw=1.8,
                   label=f"Upper bound = {RATIO_HIGH}")
        ax.set_xlabel("tel_tokens / eng_tokens", fontsize=13)
        ax.set_ylabel("Pairs", fontsize=13)
        ax.set_title("Token-length ratio: Telugu target / English source", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        png_path = os.path.join(filtered_dir, "ratio_distribution.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"\n  Histogram saved â†’ {png_path}")
    except Exception as e:
        print(f"\n  Could not save histogram: {e}")

    # -----------------------------------------------------------------------
    # STEP 4 â€” Apply filters
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 â€” Applying filters")
    print("=" * 60)

    # Rule 1 â€” English source length: [4, 60]
    ENG_MIN, ENG_MAX = 4, 60
    before = len(tokenized)
    filtered_r1 = [t for t in tokenized if ENG_MIN <= t["eng_len"] <= ENG_MAX]
    removed_r1  = before - len(filtered_r1)
    print(f"\nRule 1  English source length [{ENG_MIN}, {ENG_MAX}]:")
    print(f"  Removed below {ENG_MIN}: {sum(1 for t in tokenized if t['eng_len'] < ENG_MIN):,}")
    print(f"  Removed above {ENG_MAX}: {sum(1 for t in tokenized if t['eng_len'] > ENG_MAX):,}")
    print(f"  Remaining: {len(filtered_r1):,}  (removed: {removed_r1:,})")

    # Rule 2 â€” Telugu target length: [5, 80]
    TEL_MIN, TEL_MAX = 5, 80
    before = len(filtered_r1)
    filtered_r2 = [t for t in filtered_r1 if TEL_MIN <= t["tel_len"] <= TEL_MAX]
    removed_r2  = before - len(filtered_r2)
    print(f"\nRule 2  Telugu target length [{TEL_MIN}, {TEL_MAX}]:")
    print(f"  Removed below {TEL_MIN}: {sum(1 for t in filtered_r1 if t['tel_len'] < TEL_MIN):,}")
    print(f"  Removed above {TEL_MAX}: {sum(1 for t in filtered_r1 if t['tel_len'] > TEL_MAX):,}")
    print(f"  Remaining: {len(filtered_r2):,}  (removed: {removed_r2:,})")

    # Rule 3 â€” Ratio filter: [0.5, 5.0]  (tel / eng)
    before = len(filtered_r2)
    filtered_r3 = [t for t in filtered_r2
                   if t["eng_len"] > 0 and RATIO_LOW <= (t["tel_len"] / t["eng_len"]) <= RATIO_HIGH]
    removed_r3 = before - len(filtered_r3)
    print(f"\nRule 3  Ratio tel/eng [{RATIO_LOW}, {RATIO_HIGH}]:")
    print(f"  Remaining: {len(filtered_r3):,}  (removed: {removed_r3:,})")

    # Rule 4 â€” Telugu script validity
    before = len(filtered_r3)
    filtered_r4 = []
    removed_by_script = []
    for t in filtered_r3:
        if check_telugu_unicode_ratio(t["target"]):
            filtered_r4.append(t)
        else:
            removed_by_script.append(t)
    removed_r4 = before - len(filtered_r4)
    print(f"\nRule 4  Telugu script â‰¥80% U+0C00â€“U+0C7F:")
    print(f"  Remaining: {len(filtered_r4):,}  (removed: {removed_r4:,})")

    print("\n  5 examples removed by script filter:")
    for i, ex in enumerate(removed_by_script[:5], 1):
        print(f"  [{i}] source: {ex['source'][:90]}")
        print(f"       target: {ex['target'][:90]}")

    # Rule 5 â€” Deduplicate on English source only (keep first occurrence)
    before = len(filtered_r4)
    seen_source = set()
    filtered_r5 = []
    for t in filtered_r4:
        key = t["source"].strip()
        if key not in seen_source:
            seen_source.add(key)
            filtered_r5.append(t)
    removed_r5 = before - len(filtered_r5)
    print(f"\nRule 5  Deduplication on English source only:")
    print(f"  Remaining: {len(filtered_r5):,}  (removed: {removed_r5:,})")

    final_count = len(filtered_r5)

    # -----------------------------------------------------------------------
    # STEP 5 â€” Print and save funnel
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FILTERING FUNNEL â€” English â†’ Telugu")
    print("=" * 60)

    funnel = [
        f"Raw corpus:                {raw_count:>7,}",
        f"After English len filter:  {len(filtered_r1):>7,}  (removed: {removed_r1:,},  bounds: [{ENG_MIN}, {ENG_MAX}])",
        f"After Telugu len filter:   {len(filtered_r2):>7,}  (removed: {removed_r2:,},  bounds: [{TEL_MIN}, {TEL_MAX}])",
        f"After ratio filter:        {len(filtered_r3):>7,}  (removed: {removed_r3:,},  bounds: [{RATIO_LOW}, {RATIO_HIGH}])",
        f"After script filter:       {len(filtered_r4):>7,}  (removed: {removed_r4:,})",
        f"After deduplication:       {len(filtered_r5):>7,}  (removed: {removed_r5:,})",
        f"Final clean pairs:         {final_count:>7,}",
    ]
    for line in funnel:
        print(f"  {line}")

    funnel_path = os.path.join(filtered_dir, "filtering_funnel.txt")
    with open(funnel_path, "w", encoding="utf-8") as f:
        f.write("FILTERING FUNNEL â€” English â†’ Telugu\n")
        f.write("=" * 55 + "\n")
        for line in funnel:
            f.write(line + "\n")
    print(f"\n  Funnel saved â†’ {funnel_path}")

    # -----------------------------------------------------------------------
    # STEP 6 â€” Split: test=1000, val=2000, train=everything else
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 â€” Splitting")
    print("=" * 60)

    if final_count < 3000:
        raise RuntimeError(f"Only {final_count} pairs after filtering â€” need at least 3,000.")

    random.seed(42)
    random.shuffle(filtered_r5)

    test_pairs  = filtered_r5[:1000]
    val_pairs   = filtered_r5[1000:3000]
    train_pairs = filtered_r5[3000:]

    print(f"  Test:  {len(test_pairs):,}")
    print(f"  Val:   {len(val_pairs):,}")
    print(f"  Train: {len(train_pairs):,}  (all remaining clean pairs)")

    def write_jsonl(pairs, filename, prefix):
        path = os.path.join(filtered_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for idx, p in enumerate(pairs, 1):
                f.write(json.dumps({
                    "id":     f"{prefix}_{idx:05d}",
                    "source": p["source"],
                    "target": p["target"],
                }, ensure_ascii=False) + "\n")
        print(f"  Saved {len(pairs):,} lines â†’ {path}")

    write_jsonl(train_pairs, "train.json", "train")
    write_jsonl(val_pairs,   "val.json",   "val")
    write_jsonl(test_pairs,  "test.json",  "test")

    # -----------------------------------------------------------------------
    # STEP 7 â€” Verify zero overlap on source side across splits
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 â€” Overlap verification (English source)")
    print("=" * 60)

    train_src = {p["source"].strip() for p in train_pairs}
    val_src   = {p["source"].strip() for p in val_pairs}
    test_src  = {p["source"].strip() for p in test_pairs}

    tv = train_src & val_src
    tt = train_src & test_src
    vt = val_src   & test_src

    print(f"  Train âˆ© Val:  {len(tv)}")
    print(f"  Train âˆ© Test: {len(tt)}")
    print(f"  Val   âˆ© Test: {len(vt)}")
    assert len(tv) == 0 and len(tt) == 0 and len(vt) == 0, "Overlap found!"
    print("  Zero overlap confirmed.")

    # -----------------------------------------------------------------------
    # STEP 8 â€” Token stats on train split + save stats.json
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8 â€” Token statistics on train split")
    print("=" * 60)

    tr_eng = [p["eng_len"] for p in train_pairs]
    tr_tel = [p["tel_len"] for p in train_pairs]
    tr_ratio = [p["tel_len"] / p["eng_len"] for p in train_pairs if p["eng_len"] > 0]

    print_distribution("English source:", tr_eng)
    print_distribution("Telugu target:",  tr_tel)
    print_distribution("Ratio (tel/eng):", tr_ratio)

    stats = {
        "english": {
            "min":    int(min(tr_eng)),
            "max":    int(max(tr_eng)),
            "mean":   round(statistics.mean(tr_eng), 3),
            "median": float(statistics.median(tr_eng)),
        },
        "telugu": {
            "min":    int(min(tr_tel)),
            "max":    int(max(tr_tel)),
            "mean":   round(statistics.mean(tr_tel), 3),
            "median": float(statistics.median(tr_tel)),
        },
        "expansion_ratio_tel_over_eng": {
            "mean":   round(statistics.mean(tr_ratio), 3),
            "median": round(float(statistics.median(tr_ratio)), 3),
        },
        "split_sizes": {
            "train": len(train_pairs),
            "val":   len(val_pairs),
            "test":  len(test_pairs),
        },
        "filter_bounds": {
            "eng_min": ENG_MIN, "eng_max": ENG_MAX,
            "tel_min": TEL_MIN, "tel_max": TEL_MAX,
            "ratio_min": RATIO_LOW, "ratio_max": RATIO_HIGH,
        },
    }

    stats_path = os.path.join(filtered_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Stats saved â†’ {stats_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  {len(train_pairs):,} train  |  {len(val_pairs):,} val  |  {len(test_pairs):,} test")
    print("=" * 60)


if __name__ == "__main__":
    main()


