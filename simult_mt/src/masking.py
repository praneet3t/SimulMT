import os
import sys
import torch


class WaitKMaskController:
    """
    Injects wait-k attention constraints into every attention layer of a
    decoder-only model via forward pre-hooks.

    Sequence layout (flat, English â†’ Telugu):
        [prompt tokens] [English source tokens] [separator] [Telugu target tokens] [EOS]

    Under wait-k, Telugu target token at output step t (0-indexed) may only
    attend to English source positions [source_start, source_start + min(k+t, S)),
    where S = source_end - source_start.
    All other source positions are masked with âˆ’10000.0.
    Prompt rows and source rows are left unchanged.
    """

    def __init__(self, model):
        self.model            = model
        self.hook_handles     = []
        self.source_start     = None
        self.source_end       = None
        self.target_start     = None
        self.current_k        = None
        self.current_seq_len  = None

    # ------------------------------------------------------------------
    # Single-sample interface (inference / unit tests)
    # ------------------------------------------------------------------

    def set_context(self, source_start, source_end, target_start, seq_len, k):
        """
        source_start : int  â€” first English source token index
        source_end   : int  â€” one past last English source token (exclusive)
        target_start : int  â€” first Telugu target token index
        seq_len      : int  â€” total flat sequence length
        k            : int or "full"
        """
        self.source_start    = source_start
        self.source_end      = source_end
        self.target_start    = target_start
        self.current_seq_len = seq_len
        self.current_k       = k

    def build_mask(self):
        """
        Build a single-sample additive attention bias of shape (seq_len, seq_len).
        0.0 = attend normally, âˆ’10000.0 = masked.
        """
        seq_len = self.current_seq_len
        mask    = torch.zeros((seq_len, seq_len), dtype=torch.float32)

        if self.current_k == "full":
            return mask

        k             = self.current_k
        ss            = self.source_start
        se            = self.source_end
        ts            = self.target_start
        source_length = se - ss

        for i in range(ts, seq_len):
            t           = i - ts
            max_visible = min(k + t, source_length)
            for j in range(ss, se):
                if (j - ss) >= max_visible:
                    mask[i][j] = -10000.0

        return mask

    # ------------------------------------------------------------------
    # Batched interface (training)
    # ------------------------------------------------------------------

    def build_batch_mask(self, source_starts, source_ends, target_starts,
                         seq_len, k, dtype=torch.float32, device="cpu"):
        """
        Build a per-sample wait-k additive bias for an entire batch.

        source_starts : list[int] or LongTensor, shape [B]
        source_ends   : list[int] or LongTensor, shape [B]
        target_starts : list[int] or LongTensor, shape [B]
        seq_len       : int
        k             : int or "full"

        Returns tensor of shape [B, 1, seq_len, seq_len].
        """
        B    = len(source_starts)
        mask = torch.zeros((B, 1, seq_len, seq_len), dtype=dtype, device=device)

        if k == "full":
            return mask

        for b in range(B):
            ss  = int(source_starts[b])
            se  = int(source_ends[b])
            ts  = int(target_starts[b])
            src_len = se - ss

            if src_len <= 0 or ts >= seq_len:
                continue

            # Vectorised: avoid inner j loop
            t_idx       = torch.arange(ts, seq_len, device=device) - ts          # [T]
            max_visible = torch.clamp(k + t_idx, max=src_len)                    # [T]
            src_off     = torch.arange(src_len, device=device)                   # [S]

            # should_mask[i, j] = True if src_off[j] >= max_visible[i]
            should_mask = src_off.unsqueeze(0) >= max_visible.unsqueeze(1)        # [T, S]
            mask[b, 0, ts:, ss:se] = torch.where(
                should_mask,
                torch.full((1,), -10000.0, dtype=dtype, device=device),
                torch.zeros(1, dtype=dtype, device=device),
            )

        return mask

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self, batch_mask=None):
        """
        Register a forward pre-hook on every attention layer.

        batch_mask : optional pre-built [B, 1, L, L] tensor for batched training.
                     If None, build_mask() is called inside the hook (single-sample).
        """
        self.remove_hooks()

        import inspect
        has_with_kwargs = "with_kwargs" in inspect.signature(
            torch.nn.Module.register_forward_pre_hook
        ).parameters

        _batch_mask = [batch_mask]   # mutable cell so hook can read latest value

        def hook_with_kwargs(module, args, kwargs):
            attn_mask = kwargs.get("attention_mask", None)
            if attn_mask is None and len(args) > 1:
                attn_mask = args[1]

            if attn_mask is not None:
                if _batch_mask[0] is not None:
                    wk = _batch_mask[0].to(device=attn_mask.device, dtype=attn_mask.dtype)
                else:
                    wk = self.build_mask().to(device=attn_mask.device, dtype=attn_mask.dtype)
                    if attn_mask.ndim == 4:
                        wk = wk.unsqueeze(0).unsqueeze(1)
                    elif attn_mask.ndim == 3:
                        wk = wk.unsqueeze(0)

                # Broadcast batch dim if model repeated mask across heads
                if attn_mask.shape[0] == 1 and wk.shape[0] > 1:
                    attn_mask = attn_mask.expand(wk.shape[0], -1, -1, -1)

                new_mask = attn_mask + wk
                if "attention_mask" in kwargs:
                    kwargs["attention_mask"] = new_mask
                elif len(args) > 1:
                    args = (args[0], new_mask) + args[2:]

            return args, kwargs

        def hook_legacy(module, args):
            if len(args) > 1 and args[1] is not None:
                attn_mask = args[1]
                if _batch_mask[0] is not None:
                    wk = _batch_mask[0].to(device=attn_mask.device, dtype=attn_mask.dtype)
                else:
                    wk = self.build_mask().to(device=attn_mask.device, dtype=attn_mask.dtype)
                    if attn_mask.ndim == 4:
                        wk = wk.unsqueeze(0).unsqueeze(1)
                    elif attn_mask.ndim == 3:
                        wk = wk.unsqueeze(0)

                if attn_mask.shape[0] == 1 and wk.shape[0] > 1:
                    attn_mask = attn_mask.expand(wk.shape[0], -1, -1, -1)

                new_args = list(args)
                new_args[1] = attn_mask + wk
                return tuple(new_args)
            return args

        for name, module in self.model.named_modules():
            if "attn" in name.lower() or "Attention" in module.__class__.__name__:
                if has_with_kwargs:
                    h = module.register_forward_pre_hook(hook_with_kwargs, with_kwargs=True)
                else:
                    h = module.register_forward_pre_hook(hook_legacy)
                self.hook_handles.append(h)

        return _batch_mask   # caller can update _batch_mask[0] without re-registering hooks

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_tests():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Loading tokenizer (optional â€” skip if offline)...")
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("sarvamai/sarvam-translate")
        print("  Tokenizer loaded OK.")
    except Exception as e:
        print(f"  Skipped: {e}")

    # Synthetic sequence: 5 prompt + 6 English source + 6 Telugu target tokens
    # Layout: [0-4] prompt | [5-10] English source | [11-16] Telugu target
    source_start = 5
    source_end   = 11
    target_start = 11
    seq_len      = 17
    k            = 2

    class _DummyModel:
        def named_modules(self):
            return []

    ctrl = WaitKMaskController(_DummyModel())
    ctrl.set_context(source_start, source_end, target_start, seq_len, k)
    mask = ctrl.build_mask()

    print(f"\n17Ã—17 wait-k mask  (k={k})")
    print("     " + "".join(f"{j:>8}" for j in range(seq_len)))
    for i, row in enumerate(mask):
        vals = "".join(f"{v:>8.0f}" for v in row)
        print(f"r{i:02d}: {vals}")

    print("\n=== Verification ===")
    checks = []

    # t=0: target row 11 sees source[5,6] only â†’ [7,8,9,10] masked
    r11 = mask[11]
    c1  = r11[5].item() == 0.0 and r11[6].item() == 0.0 and \
          all(r11[j].item() == -10000.0 for j in range(7, 11))
    checks.append(("Row 11 (t=0): sees source[5,6] only", c1))

    # t=1: target row 12 sees source[5,6,7] â†’ [8,9,10] masked
    r12 = mask[12]
    c2  = all(r12[j].item() == 0.0 for j in [5, 6, 7]) and \
          all(r12[j].item() == -10000.0 for j in range(8, 11))
    checks.append(("Row 12 (t=1): sees source[5,6,7] only", c2))

    # t=2: target row 13 sees source[5,6,7,8] â†’ [9,10] masked
    r13 = mask[13]
    c3  = all(r13[j].item() == 0.0 for j in [5, 6, 7, 8]) and \
          all(r13[j].item() == -10000.0 for j in range(9, 11))
    checks.append(("Row 13 (t=2): sees source[5,6,7,8] only", c3))

    # Prompt rows 0-4 unchanged
    c4 = all(torch.all(mask[r] == 0.0).item() for r in range(5))
    checks.append(("Prompt rows 0-4: all zero", c4))

    # English source rows 5-10 unchanged
    c5 = all(torch.all(mask[r] == 0.0).item() for r in range(5, 11))
    checks.append(("English source rows 5-10: all zero", c5))

    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {label}")
        if not passed:
            all_pass = False

    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    # Also smoke-test build_batch_mask
    ctrl2 = WaitKMaskController(_DummyModel())
    bm = ctrl2.build_batch_mask(
        source_starts=[5, 5],
        source_ends=[11, 11],
        target_starts=[11, 11],
        seq_len=17,
        k=2,
    )
    assert bm.shape == (2, 1, 17, 17), f"Unexpected batch mask shape: {bm.shape}"
    print("\nbuild_batch_mask shape check: PASS")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()


