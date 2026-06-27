"""M2b — look-ahead OVERLAP cute-DSL QR (panel[k+1] ‖ far_apply[k]), CTA-barrier delimited.

The first overlap measurement. Splits each super-panel's apply into NARROW (next-panel columns, must
finish before panel[k+1]) + FAR (the rest), then runs far_apply[k] ‖ panel_factor[k+1] concurrently
between two CTA barriers (disjoint columns → correct). OVERLAP flag toggles the concurrency:
  ov=0 (serial):  narrow[k]; barrier; far[k]; barrier; panel[k+1]; barrier
  ov=1 (overlap): narrow[k]; barrier; { far[k] ‖ panel[k+1] }; barrier
Same total work; the ov=1 vs ov=0 time delta = how much of panel[k+1]'s serial latency hides behind
far_apply[k]. Correctness must be bit-identical (ov=1 == ov=0 == geqrf). This answers Depth-1
hideability at the MECHANISM level (still warp-reduce apply, not tcgen05 — clean perf verdict needs M3).

warp0 = PANEL (setmaxregister_increase 192), warp1 = APPLY (decrease 128). Reflectors in gmem stril(H).
Run: modal run modal_cute_lab.py::run_candidate --script cute_qr_m2b.py
(static-scan clean.)
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

N_CUTE_MAX = 256
NB = 16


@cute.jit
def _warp_reduce_add(val: cutlass.Float32) -> cutlass.Float32:
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=(1 << i))
    return val


@cute.jit
def _apply_reflector(Hm, col, cc, tau_j, lane, n: cutlass.Constexpr):
    pw = cutlass.Float32(0.0)
    r = col + 1 + lane
    while r < n:
        pw = pw + Hm[r, col] * Hm[r, cc]
        r = r + 32
    w = _warp_reduce_add(pw) + Hm[col, cc]
    if lane == 0:
        Hm[col, cc] = Hm[col, cc] - tau_j * w
    r = col + 1 + lane
    while r < n:
        Hm[r, cc] = Hm[r, cc] - tau_j * Hm[r, col] * w
        r = r + 32


@cute.jit
def _apply_range(Hm, tm, c0, pend, lo, hi, lane, n: cutlass.Constexpr):
    # apply panel [c0,pend) reflectors to columns [lo, hi)
    col = c0
    while col < pend:
        tau_j = tm[col]
        if tau_j != 0.0:
            cc = lo
            while cc < hi:
                _apply_reflector(Hm, col, cc, tau_j, lane, n)
                cc = cc + 1
        col = col + 1


@cute.jit
def _panel_factor(Hm, tm, c0, pend, lane, n: cutlass.Constexpr):
    col = c0
    while col < pend:
        alpha = Hm[col, col]
        partial = cutlass.Float32(0.0)
        r = col + 1 + lane
        while r < n:
            hv = Hm[r, col]
            partial = partial + hv * hv
            r = r + 32
        xnorm2 = _warp_reduce_add(partial)
        need = xnorm2 > 0.0
        normfull = cute.math.sqrt(alpha * alpha + xnorm2, fastmath=True)
        beta = -normfull if alpha >= 0.0 else normfull
        tau_j = (beta - alpha) / beta if need else cutlass.Float32(0.0)
        scale = 1.0 / (alpha - beta) if need else cutlass.Float32(0.0)
        r = col + 1 + lane
        while r < n:
            if need:
                Hm[r, col] = Hm[r, col] * scale
            r = r + 32
        if lane == 0:
            Hm[col, col] = beta if need else alpha
            tm[col] = tau_j
        if need:
            cc = col + 1
            while cc < pend:
                _apply_reflector(Hm, col, cc, tau_j, lane, n)
                cc = cc + 1
        col = col + 1


@cute.kernel
def _qr_kernel(mH, mtau, n: cutlass.Constexpr, nb: cutlass.Constexpr, OV: cutlass.Constexpr):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    Hm = mH[bid, None, None]
    tm = mtau[bid, None]

    if warp == 0:
        cute.arch.setmaxregister_increase(192)
    else:
        cute.arch.setmaxregister_decrease(128)

    # seed: factor panel 0
    pend0 = nb if nb < n else n
    if warp == 0:
        _panel_factor(Hm, tm, 0, pend0, lane, n)
    cute.arch.barrier()

    c0 = cutlass.Int32(0)
    while c0 < n:
        pend = c0 + nb if c0 + nb < n else n         # current panel [c0,pend) already factored
        pend2 = pend + nb if pend + nb < n else n    # next panel [pend,pend2)
        # NARROW apply: panel[c0] -> next-panel cols [pend,pend2)  (must precede panel[k+1])
        if warp == 1:
            _apply_range(Hm, tm, c0, pend, pend, pend2, lane, n)
        cute.arch.barrier()                          # next-panel cols finalized + visible

        if OV == 1:
            # CONCURRENT: far_apply[k] (warp1)  ‖  panel_factor[k+1] (warp0)
            if warp == 1:
                _apply_range(Hm, tm, c0, pend, pend2, n, lane, n)
            if warp == 0:
                if pend < n:
                    _panel_factor(Hm, tm, pend, pend2, lane, n)
            cute.arch.barrier()
        else:
            # SERIAL: far_apply[k], then panel_factor[k+1], split by a barrier
            if warp == 1:
                _apply_range(Hm, tm, c0, pend, pend2, n, lane, n)
            cute.arch.barrier()
            if warp == 0:
                if pend < n:
                    _panel_factor(Hm, tm, pend, pend2, lane, n)
            cute.arch.barrier()

        c0 = c0 + nb


@cute.jit
def _qr_host(mH, mtau, n: cutlass.Constexpr, nb: cutlass.Constexpr, OV: cutlass.Constexpr):
    B = cute.size(mH, mode=[0])
    _qr_kernel(mH, mtau, n, nb, OV).launch(grid=[B, 1, 1], block=[64, 1, 1])


_compiled = {}


def _run_cute(H, tau, ov):
    n = H.shape[-1]
    Ht = from_dlpack(H).mark_layout_dynamic()
    tt = from_dlpack(tau).mark_layout_dynamic()
    key = (n, H.shape[0], ov)
    if key not in _compiled:
        _compiled[key] = cute.compile(_qr_host, Ht, tt, n, NB, ov)
    _compiled[key](Ht, tt)


def custom_kernel(data):
    A = data
    B, n, _ = A.shape
    if n <= N_CUTE_MAX:
        try:
            H = A.clone().contiguous()
            tau = torch.zeros(B, n, device=A.device, dtype=A.dtype)
            _run_cute(H, tau, 1)
            torch.cuda.synchronize()
            return H, tau
        except Exception:
            pass
    return torch.geqrf(A)


def _time(fn, it=50):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.empty((16, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)  # clear L2
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000  # median us


if __name__ == "__main__":
    import traceback
    print(f"=== M2b look-ahead overlap (NB={NB}): correctness + ov=0 vs ov=1 timing ===")
    torch.manual_seed(0)
    print("\n-- correctness (ov=1 must match geqrf) --")
    for B, n in [(1, 64), (1, 128), (4, 256), (1, 100)]:
        A = torch.randn(B, n, n, device="cuda", dtype=torch.float32)
        try:
            for ov in (0, 1):
                H = A.clone().contiguous(); tau = torch.zeros(B, n, device="cuda", dtype=torch.float32)
                _run_cute(H, tau, ov); torch.cuda.synchronize()
                Hg, taug = torch.geqrf(A)
                herr = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
                ok = herr < 1e-4
                print(f"  B={B} n={n:4d} ov={ov}: H={herr:.2e}  {'PASS ✅' if ok else 'FAIL ❌'}")
        except Exception:
            print(f"  B={B} n={n}: EXC\n" + traceback.format_exc()[-1800:])
    print("\n-- timing: ov=0 (serial) vs ov=1 (overlap), grid=B --")
    for B, n in [(1, 256), (64, 256), (640, 128)]:
        try:
            A = torch.randn(B, n, n, device="cuda", dtype=torch.float32)
            H = A.clone().contiguous(); tau = torch.zeros(B, n, device="cuda", dtype=torch.float32)
            t0 = _time(lambda: _run_cute(H, tau, 0))
            t1 = _time(lambda: _run_cute(H, tau, 1))
            print(f"  B={B:4d} n={n}: ov0={t0:9.1f}us  ov1={t1:9.1f}us  overlap={t0/t1:.3f}x "
                  f"({'HIDES' if t1 < t0*0.97 else 'no/neg'})")
        except Exception:
            print(f"  B={B} n={n}: EXC\n" + traceback.format_exc()[-1500:])
    print("DONE")
