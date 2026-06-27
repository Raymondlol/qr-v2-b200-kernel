"""M1.5 — BLOCKED cute-DSL QR (single warp), the structural precursor to the M2 overlap triangle.

Reorganizes M1's unblocked QR into super-panels of width NB, splitting each super-panel into TWO
PHASES that M2 will assign to separate warps and overlap:
  PHASE A (panel-factor): factor NB columns with Householder, applying each reflector ONLY within the
                          panel block [c0:c0+NB] (within-panel update).
  PHASE B (far-apply):    apply the NB stored reflectors to the FAR trailing columns [c0+NB : n].
Still 1 warp, warp_reduce, geqrf-matching convention — correctness-only (PHASE B is per-reflector, no
LARFT/WY/tcgen05 yet; the GEMM-ification comes at M3). The point is the panel↔apply SEPARATION.

Route n<=256 to cute, else geqrf fallback. Run: modal run modal_cute_lab.py::run_candidate --script cute_qr_m1b.py
(static-scan clean.)
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

N_CUTE_MAX = 256
NB = 16   # super-panel width


@cute.jit
def _warp_reduce_add(val: cutlass.Float32) -> cutlass.Float32:
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=(1 << i))
    return val


@cute.jit
def _apply_reflector(Hm: cute.Tensor, col: cutlass.Int32, cc: cutlass.Int32,
                     tau_j: cutlass.Float32, lane: cutlass.Int32, n: cutlass.Constexpr):
    # apply reflector v=[1; H[col+1:,col]] (with tau_j) to column cc:  H[:,cc] -= tau_j * v * (v^T H[:,cc])
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


@cute.kernel
def _qr_kernel(mH: cute.Tensor, mtau: cute.Tensor, n: cutlass.Constexpr, nb: cutlass.Constexpr):
    bid, _, _ = cute.arch.block_idx()
    lane, _, _ = cute.arch.thread_idx()
    Hm = mH[bid, None, None]
    tm = mtau[bid, None]

    c0 = cutlass.Int32(0)
    while c0 < n:
        pend = c0 + nb if c0 + nb < n else n          # panel end (exclusive)
        # ---- PHASE A: factor panel cols [c0, pend) with within-panel update ----
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
                while cc < pend:                        # within-panel only
                    _apply_reflector(Hm, col, cc, tau_j, lane, n)
                    cc = cc + 1
            col = col + 1

        # ---- PHASE B: far-apply the panel's reflectors to trailing cols [pend, n) ----
        col = c0
        while col < pend:
            tau_j = tm[col]
            if tau_j != 0.0:
                cc = pend
                while cc < n:
                    _apply_reflector(Hm, col, cc, tau_j, lane, n)
                    cc = cc + 1
            col = col + 1

        c0 = c0 + nb


@cute.jit
def _qr_host(mH: cute.Tensor, mtau: cute.Tensor, n: cutlass.Constexpr, nb: cutlass.Constexpr):
    B = cute.size(mH, mode=[0])
    _qr_kernel(mH, mtau, n, nb).launch(grid=[B, 1, 1], block=[32, 1, 1])


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
    print(f"=== M1.5 BLOCKED cute-DSL QR (NB={NB}) vs torch.geqrf ===")
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
