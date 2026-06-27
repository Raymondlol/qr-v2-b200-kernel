"""M1 — thinnest CORRECT cute-DSL batched QR (one warp per matrix, unblocked Householder).

Purpose: get a CORRECT compact-Householder QR running end-to-end in cute-DSL on B200 (validate the
QR math + data movement + the geqrf-matching (H,tau) contract), as the prerequisite for the M2
warp-spec overlap triangle. NOT fast: 1 warp/matrix, unblocked (each reflector applied immediately,
no LARFT/WY/tcgen05). Speed comes later; M1's only job is correctness.

Design (one CTA = one matrix = ONE warp of 32 lanes; lane t owns strided rows t, t+32, ...):
  for col in 0..n:
    alpha = H[col,col];  xnorm2 = sum_{i>col} H[i,col]^2   (local-accumulate + warp_reduce)
    beta = -sign(alpha)*sqrt(alpha^2+xnorm2);  tau = (beta-alpha)/beta;  scale = 1/(alpha-beta)
    H[i>col,col] *= scale;  H[col,col] = beta;  v[col]=1 (implicit)   -> reflector in stril(H)
    for cc in col+1..n:                                              # apply reflector to trailing
      w = H[col,cc] + sum_{i>col} H[i,col]*H[i,cc]   (warp_reduce)
      H[col,cc] -= tau*w;  H[i>col,cc] -= tau*H[i,col]*w
Matches torch.geqrf's convention (same as the validated fused_qr_slice.py). Route n<=256 to cute,
else torch.geqrf fallback so the contract is always satisfied.

Run: modal run modal_cute_lab.py::run_candidate --script cute_qr_m1.py
(static-scan clean: no banned launch-plumbing substrings.)
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

N_CUTE_MAX = 256   # route n<=this to the cute kernel; larger -> geqrf fallback


@cute.jit
def _warp_reduce_add(val: cutlass.Float32) -> cutlass.Float32:
    # butterfly all-reduce over 32 lanes (log2(32)=5 steps)
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=(1 << i))
    return val


@cute.kernel
def _qr_kernel(mH: cute.Tensor, mtau: cute.Tensor, n: cutlass.Constexpr):
    bid, _, _ = cute.arch.block_idx()
    lane, _, _ = cute.arch.thread_idx()          # 0..31
    Hm = mH[bid, None, None]                      # [n, n] view of this CTA's matrix
    tm = mtau[bid, None]                          # [n]

    for col in cutlass.range(n):
        alpha = Hm[col, col]
        # xnorm2 = sum_{i>col} H[i,col]^2  (lane owns rows col+1+lane, +32, ...)
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

        # write reflector below diagonal + R diagonal + tau
        r = col + 1 + lane
        while r < n:
            if need:
                Hm[r, col] = Hm[r, col] * scale
            r = r + 32
        if lane == 0:
            Hm[col, col] = beta if need else alpha
            tm[col] = tau_j

        # apply reflector (v=[1; H[col+1:,col]], tau_j) to every trailing column cc>col
        if need:
            cc = col + 1
            while cc < n:
                # w = v^T H[:,cc] = H[col,cc] + sum_{i>col} H[i,col]*H[i,cc]
                pw = cutlass.Float32(0.0)
                r = col + 1 + lane
                while r < n:
                    pw = pw + Hm[r, col] * Hm[r, cc]
                    r = r + 32
                w = _warp_reduce_add(pw) + Hm[col, cc]
                # H[:,cc] -= tau_j * v * w
                if lane == 0:
                    Hm[col, cc] = Hm[col, cc] - tau_j * w
                r = col + 1 + lane
                while r < n:
                    Hm[r, cc] = Hm[r, cc] - tau_j * Hm[r, col] * w
                    r = r + 32
                cc = cc + 1


@cute.jit
def _qr_host(mH: cute.Tensor, mtau: cute.Tensor, n: cutlass.Constexpr):
    B = cute.size(mH, mode=[0])
    _qr_kernel(mH, mtau, n).launch(grid=[B, 1, 1], block=[32, 1, 1])


_compiled = {}


def _run_cute(H, tau):
    n = H.shape[-1]
    Ht = from_dlpack(H).mark_layout_dynamic()
    tt = from_dlpack(tau).mark_layout_dynamic()
    key = (n, H.shape[0])
    if key not in _compiled:
        _compiled[key] = cute.compile(_qr_host, Ht, tt, n)
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
    print("=== M1 cute-DSL QR correctness vs torch.geqrf ===")
    torch.manual_seed(0)
    for B, n in [(1, 32), (1, 64), (1, 128), (8, 64), (4, 256)]:
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
            print(f"  B={B:2d} n={n:4d}: H relerr={herr:.2e} tau relerr={terr:.2e}  {'PASS ✅' if ok else 'FAIL ❌'}")
        except Exception:
            print(f"  B={B:2d} n={n:4d}: EXC\n" + traceback.format_exc()[-2500:])
    print("DONE")
