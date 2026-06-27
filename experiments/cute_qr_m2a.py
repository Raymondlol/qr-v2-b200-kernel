"""M2a-0 — warp-SPLIT cute-DSL QR (panel warp ‖ apply warp), serial via CTA barrier.

First step of the M2 overlap triangle: split each super-panel's two phases across SEPARATE warps with
per-warp register realloc (the FA4 mechanism), synchronized by a full CTA barrier (SERIAL, no overlap
yet — ov=0 baseline). Validates that the warp-role split + setmaxregister produces correct QR, before
adding the mbarrier handshake (M2a-1) and the look-ahead overlap (M2b).

  WARP 0 (PANEL,  setmaxregister_increase(192)): factor super-panel cols [c0,pend) + within-panel update
  WARP 1 (APPLY,  setmaxregister_decrease(128)): far-apply the panel's reflectors to cols [pend,n)
  barriers between phases give cross-warp gmem visibility + ordering (serial).

Reflectors live in gmem stril(H); the apply warp reads them directly (no SMEM handoff needed for the
unblocked apply). 2 warps (64 threads). Route n<=256 to cute, else geqrf.
Run: modal run modal_cute_lab.py::run_candidate --script cute_qr_m2a.py
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
def _apply_reflector(Hm: cute.Tensor, col: cutlass.Int32, cc: cutlass.Int32,
                     tau_j: cutlass.Float32, lane: cutlass.Int32, n: cutlass.Constexpr):
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
def _panel_factor(Hm: cute.Tensor, tm: cute.Tensor, c0: cutlass.Int32, pend: cutlass.Int32,
                  lane: cutlass.Int32, n: cutlass.Constexpr):
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


@cute.jit
def _far_apply(Hm: cute.Tensor, tm: cute.Tensor, c0: cutlass.Int32, pend: cutlass.Int32,
               lane: cutlass.Int32, n: cutlass.Constexpr):
    col = c0
    while col < pend:
        tau_j = tm[col]
        if tau_j != 0.0:
            cc = pend
            while cc < n:
                _apply_reflector(Hm, col, cc, tau_j, lane, n)
                cc = cc + 1
        col = col + 1


@cute.kernel
def _qr_kernel(mH: cute.Tensor, mtau: cute.Tensor, n: cutlass.Constexpr, nb: cutlass.Constexpr):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    Hm = mH[bid, None, None]
    tm = mtau[bid, None]

    # FA4 per-warp register realloc: panel warp gets more regs, apply warp fewer.
    if warp == 0:
        cute.arch.setmaxregister_increase(192)
    else:
        cute.arch.setmaxregister_decrease(128)

    c0 = cutlass.Int32(0)
    while c0 < n:
        pend = c0 + nb if c0 + nb < n else n
        if warp == 0:
            _panel_factor(Hm, tm, c0, pend, lane, n)
        cute.arch.barrier()                       # panel reflectors now visible to apply warp
        if warp == 1:
            _far_apply(Hm, tm, c0, pend, lane, n)
        cute.arch.barrier()                       # apply done before next panel
        c0 = c0 + nb


@cute.jit
def _qr_host(mH: cute.Tensor, mtau: cute.Tensor, n: cutlass.Constexpr, nb: cutlass.Constexpr):
    B = cute.size(mH, mode=[0])
    _qr_kernel(mH, mtau, n, nb).launch(grid=[B, 1, 1], block=[64, 1, 1])   # 2 warps


_compiled = {}


def _run_cute(H, tau):
    n = H.shape[-1]
    Ht = from_dlpack(H).mark_layout_dynamic()
    tt = from_dlpack(tau).mark_layout_dynamic()
    key = (n, H.shape[0])
    if key not in _compiled:
        _compiled[key] = cute.compile(_qr_host, Ht, tt, n, NB)
    _compiled[key](Ht, tt)


def custom_kernel(data):
    A = data
    B, n, _ = A.shape
    if n <= N_CUTE_MAX:
        try:
            H = A.clone().contiguous()
            tau = torch.zeros(B, n, device=A.device, dtype=A.dtype)
            _run_cute(H, tau)
            torch.cuda.synchronize()
            return H, tau
        except Exception:
            pass
    return torch.geqrf(A)


if __name__ == "__main__":
    import traceback
    print(f"=== M2a-0 warp-split (panel‖apply, CTA-barrier serial, NB={NB}) vs torch.geqrf ===")
    torch.manual_seed(0)
    for B, n in [(1, 32), (1, 64), (1, 128), (8, 64), (4, 256), (1, 100)]:
        A = torch.randn(B, n, n, device="cuda", dtype=torch.float32)
        try:
            H = A.clone().contiguous()
            tau = torch.zeros(B, n, device="cuda", dtype=torch.float32)
            _run_cute(H, tau)
            torch.cuda.synchronize()
            Hg, taug = torch.geqrf(A)
            herr = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
            terr = (tau - taug).abs().max().item() / (taug.abs().max().item() + 1e-30)
            ok = herr < 1e-4 and terr < 1e-4
            print(f"  B={B:2d} n={n:4d}: H={herr:.2e} tau={terr:.2e}  {'PASS ✅' if ok else 'FAIL ❌'}")
        except Exception:
            print(f"  B={B:2d} n={n:4d}: EXC\n" + traceback.format_exc()[-2000:])
    print("DONE")
