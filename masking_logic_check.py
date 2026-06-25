"""
Pure-Python verification of the wait-k masking logic (no torch needed).

Mirrors two implementations and proves they agree:
  REF  = build_mask() convention used in training (masking.py / train.py
         build_batch_waitk_mask): for absolute row i >= ts, t = i - ts,
         max_visible = min(k+t, S); block source col j where (j-ss) >= max_visible.
  GEN  = waitk_bias(q_len, kv_len): the new dynamic per-forward bias used during
         cached generation. Query rows occupy abs positions [kv-q, kv).

We check:
  1. GEN over a single full forward (q=kv=L) == REF full [L,L] mask.
  2. GEN incremental decode (q=1, kv=i+1) row == REF row i, on the source span.
  3. Prefill (rows < ts) is never masked.
  4. Different k produce different masks (real latency/quality tradeoff exists).
"""

BLOCK = -10000.0


def ref_full_mask(ss, se, ts, L, k):
    """Reference [L,L] additive mask (training convention)."""
    S = se - ss
    M = [[0.0] * L for _ in range(L)]
    if k == "full":
        return M
    for i in range(ts, L):
        t = i - ts
        max_visible = min(k + t, S)
        for j in range(ss, se):
            if (j - ss) >= max_visible:
                M[i][j] = BLOCK
    return M


def gen_bias(ss, se, ts, q_len, kv_len, k):
    """Mirror of WaitKMaskController.waitk_bias -> [q_len, kv_len] additive bias."""
    S = se - ss
    bias = [[0.0] * kv_len for _ in range(q_len)]
    if k == "full" or S <= 0:
        return bias
    for qi in range(q_len):
        abs_pos = (kv_len - q_len) + qi           # absolute position of this query row
        if abs_pos < ts:                          # prompt / source row -> untouched
            continue
        visible = min(k + (abs_pos - ts), S)
        for j in range(ss, se):
            if (j - ss) >= visible:
                bias[qi][j] = BLOCK
    return bias


def check(ss, se, ts, L, k):
    ref = ref_full_mask(ss, se, ts, L, k)

    # (1) single full forward q=kv=L equals reference everywhere
    full = gen_bias(ss, se, ts, L, L, k)
    assert full == ref, f"full-forward mismatch ss={ss} se={se} ts={ts} L={L} k={k}"

    # (2) incremental decode: simulate generating tokens at positions ts..L-1.
    #     At the step whose newest token sits at absolute position p, the forward
    #     has q_len=1, kv_len=p+1. That single row must equal ref row p on source.
    for p in range(ts, L):
        row = gen_bias(ss, se, ts, 1, p + 1, k)[0]
        for j in range(ss, se):
            assert row[j] == ref[p][j], (
                f"decode row mismatch p={p} j={j} k={k}: {row[j]} vs {ref[p][j]}")

    # (3) prefill rows (< ts) never masked
    for i in range(ts):
        assert all(v == 0.0 for v in ref[i]), f"prompt row {i} masked!"


def visible_counts(ss, se, ts, L, k):
    """How many source tokens each target row can see (for the tradeoff sanity check)."""
    ref = ref_full_mask(ss, se, ts, L, k)
    S = se - ss
    return [sum(1 for j in range(ss, se) if ref[i][j] == 0.0) for i in range(ts, L)]


# ---- geometries: (source_start, source_end, target_start, total_len) ----
geoms = [
    (5, 11, 11, 17),     # the unit-test geometry from masking.py
    (8, 28, 28, 60),     # realistic: 20 src tokens, 32 target tokens
    (10, 15, 15, 40),    # short source
    (3, 50, 50, 120),    # long source
]
ks = [1, 2, 4, 7, "full"]

print("=== Equivalence: dynamic gen bias  ==  training reference ===")
for g in geoms:
    for k in ks:
        check(*g, k)
    print(f"  PASS  ss,se,ts,L={g}  for k in {ks}")

print("\n=== k=2 on unit-test geometry (rows = target steps t=0,1,2,...) ===")
ss, se, ts, L = geoms[0]
ref = ref_full_mask(ss, se, ts, L, 2)
print("     src cols:", list(range(ss, se)))
for i in range(ts, ts + 4):
    seen = [j for j in range(ss, se) if ref[i][j] == 0.0]
    print(f"  t={i-ts}: row {i} sees source positions {seen}")
assert [j for j in range(ss, se) if ref[ts][j] == 0.0] == [5, 6], "t=0 should see [5,6]"
assert [j for j in range(ss, se) if ref[ts+1][j] == 0.0] == [5, 6, 7], "t=1 should see [5,6,7]"
print("  matches technical_writeup.md unit test (t=0 -> {5,6}, t=1 -> {5,6,7}).")

print("\n=== Latency tradeoff exists: different k -> different visibility ===")
ss, se, ts, L = geoms[1]   # 20 source tokens, 32 target
print(f"  geometry: {se-ss} source tokens, {L-ts} target tokens")
prev = None
for k in [1, 2, 4, 7]:
    vc = visible_counts(ss, se, ts, L, k)
    # average proportion of source visible across target steps
    ap = sum(v / (se - ss) for v in vc) / len(vc)
    print(f"  k={k}: first-token sees {vc[0]}/{se-ss} src, mean source-visible AP={ap:.3f}")
    assert prev is None or vc[0] > prev, "larger k must see >= more at first token"
    prev = vc[0]
# full sees everything
vc_full = visible_counts(ss, se, ts, L, "full")
assert all(v == (se - ss) for v in vc_full), "full must see all source"
print(f"  full: every target row sees all {se-ss} src tokens (offline upper bound)")

print("\nALL CHECKS PASSED")
