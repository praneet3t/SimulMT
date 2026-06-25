"""
Wait-k attention masking for English -> Telugu SiMT (v2).

Design difference from simult_mt/src/masking.py: this version builds an explicit
4D additive attention mask (causal + wait-k + padding) and passes it straight to
the model's forward as `attention_mask`. There are no forward pre-hooks. The exact
same builder is used for training and for evaluation, so the two policies cannot
drift apart.

A target token at step t (0-indexed from target_start) may attend to source
positions [source_start, source_start + min(k + t, S)), where S is the source
length. Blocked positions get an additive bias of NEG before the softmax.
"""

import torch

NEG = -10000.0  # additive bias applied to blocked attention positions


def _per_sample_k(k, batch_size):
    """Normalise k into a length-B list of ints (or return None for 'full')."""
    if k == "full":
        return None
    if isinstance(k, int):
        return [k] * batch_size
    return [int(x) for x in k]  # list / tensor of per-sample k


def build_batch_mask(source_starts, source_ends, target_starts, seq_len, k,
                     pad_mask=None, dtype=torch.float32, device="cpu"):
    """
    Build a [B, 1, seq_len, seq_len] additive attention mask combining:
      * causal masking  - a query at position i cannot attend to key j > i
      * wait-k masking  - a target row constrains how much source it sees
      * padding masking - keys at padded positions are blocked (if pad_mask given)

    k : int (shared), per-sample list/tensor of ints, or "full" (causal only).
    pad_mask : optional [B, seq_len] with 1 for real tokens and 0 for padding.
    """
    B = len(source_starts)
    pos = torch.arange(seq_len, device=device)
    causal = torch.where(
        pos.unsqueeze(0) <= pos.unsqueeze(1),                 # key <= query -> visible
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), NEG, dtype=dtype, device=device),
    )                                                         # [seq_len, seq_len]
    mask = causal.unsqueeze(0).repeat(B, 1, 1)                # [B, L, L]

    ks = _per_sample_k(k, B)
    if ks is not None:
        for b in range(B):
            ss, se, ts = int(source_starts[b]), int(source_ends[b]), int(target_starts[b])
            S = se - ss
            if S <= 0 or ts >= seq_len:
                continue
            rows    = torch.arange(ts, seq_len, device=device)        # [T]
            visible = torch.clamp(ks[b] + (rows - ts), max=S)         # [T]
            src_off = torch.arange(S, device=device)                  # [S]
            block   = src_off.unsqueeze(0) >= visible.unsqueeze(1)    # [T, S]
            seg = mask[b, ts:, ss:se]
            mask[b, ts:, ss:se] = torch.where(
                block, torch.full((), NEG, dtype=dtype, device=device), seg)

    if pad_mask is not None:
        keep = pad_mask.to(device=device).bool()                      # [B, L]
        mask = mask.masked_fill((~keep).unsqueeze(1), NEG)            # block pad key columns

    return mask.unsqueeze(1)                                          # [B, 1, L, L]


def build_window_mask(q_positions, kv_len, source_start, source_end, target_start, k,
                      dtype=torch.float32, device="cpu"):
    """
    Build a [1, 1, Q, kv_len] causal + wait-k mask for query rows at absolute
    positions `q_positions` (LongTensor [Q]) attending over keys [0, kv_len).

    Used during evaluation/decoding, where Q == 1 for each newly generated token
    and kv_len grows by one per step (KV cache). Single-sample (batch size 1).
    """
    q    = q_positions.view(-1, 1)                                    # [Q, 1]
    kcol = torch.arange(kv_len, device=device).view(1, -1)           # [1, KV]
    mask = torch.where(
        kcol <= q,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), NEG, dtype=dtype, device=device),
    )                                                                # [Q, KV]

    if k != "full":
        S = source_end - source_start
        if S > 0:
            visible   = torch.clamp(int(k) + (q - target_start), max=S)   # [Q, 1]
            is_target = q >= target_start                                 # [Q, 1]
            src_off   = torch.arange(source_start, source_end, device=device).view(1, -1) - source_start
            block     = is_target & (src_off >= visible)                  # [Q, S]
            seg = mask[:, source_start:source_end]
            mask[:, source_start:source_end] = torch.where(
                block, torch.full((), NEG, dtype=dtype, device=device), seg)

    return mask.view(1, 1, *mask.shape)                              # [1, 1, Q, KV]


# ---------------------------------------------------------------------------
# Self-test (needs torch only)
# ---------------------------------------------------------------------------

def _selftest():
    # 5 prompt | 6 source [5..10] | 6 target [11..16]
    ss, se, ts, L, k = 5, 11, 11, 17, 2
    m = build_batch_mask([ss], [se], [ts], L, k)[0, 0]
    # t=0 (row 11) sees source 5,6 only
    assert m[11, 5] == 0 and m[11, 6] == 0 and m[11, 7] == NEG and m[11, 10] == NEG
    # t=1 (row 12) sees 5,6,7
    assert m[12, 7] == 0 and m[12, 8] == NEG
    # causal: row 11 cannot see future key 12
    assert m[11, 12] == NEG
    # incremental window for the row-11 query (kv up to 12) matches
    w = build_window_mask(torch.tensor([11]), 12, ss, se, ts, k)[0, 0, 0]
    assert w[5] == 0 and w[6] == 0 and w[7] == NEG
    # padding blocks a key column
    pad = torch.ones(1, L); pad[0, 6] = 0
    mp = build_batch_mask([ss], [se], [ts], L, k, pad_mask=pad)[0, 0]
    assert (mp[:, 6] == NEG).all()
    print("masking selftest: PASS")


if __name__ == "__main__":
    _selftest()
