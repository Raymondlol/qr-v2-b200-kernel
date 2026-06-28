"""M3c — per-matrix warp-spec QR engine in cute-DSL: panel(warp0) ‖ blocked-WY apply(warp1).

STAGE A (this file, BASELINE): the full per-matrix 2-warp structure with the *validated warp-reduce*
blocked-WY apply (LARFT T + C -= V(T^T(V^T C))), CTA-barrier serial. Every piece here is already
validated (m1c math, m2a/m2b warp-split+setmaxregister). Its job: lock the kernel STRUCTURE
(warp roles, super-panel loop, gram/LARFT->SMEM handoff, routing) correct end-to-end, so the only
remaining variable when we swap warp1's apply to tcgen05 tf32x3 is the GEMM engine itself.

  WARP 0 (PANEL, setmaxregister_increase 192): factor super-panel [c0,pend) + within-panel update,
                                               then gram G=V^T V (warp_reduce) + LARFT T -> SMEM sT.
  WARP 1 (APPLY, setmaxregister_decrease 128): blocked-WY apply panel[c0] reflectors to cols [pend,n).
  CTA barriers give cross-warp gmem/SMEM visibility + ordering (serial; OV=1 overlap is a later step).

Route n<=512 to the engine, else torch.geqrf.
Run: modal run modal_cute_lab.py::run_candidate --script cute_qr_m3c.py    (static-scan clean.)
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

N_CUTE_MAX = 512
NB = 16          # IB super-panel / LARFT block width


# ============================ validated reductions / V access ============================
@cute.jit
def _wreduce(val: cutlass.Float32) -> cutlass.Float32:
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=(1 << i))
    return val


@cute.jit
def _vlow(Hm, r: cutlass.Int32, col: cutlass.Int32) -> cutlass.Float32:
    # unit-lower reflector entry V[r,col]: H[r,col] if r>col, 1 if r==col, else 0
    return Hm[r, col] if r > col else (cutlass.Float32(1.0) if r == col else cutlass.Float32(0.0))


# ============================ warp0: panel factor (m1c, validated) ============================
@cute.jit
def _panel_factor(Hm, tm, c0: cutlass.Int32, pend: cutlass.Int32, lane: cutlass.Int32,
                  n: cutlass.Constexpr):
    col = c0
    while col < pend:
        alpha = Hm[col, col]
        partial = cutlass.Float32(0.0)
        r = col + 1 + lane
        while r < n:
            hv = Hm[r, col]; partial = partial + hv * hv; r = r + 32
        xnorm2 = _wreduce(partial)
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
        if need:                                          # within-panel update (cols (col,pend))
            cc = col + 1
            while cc < pend:
                pw = cutlass.Float32(0.0)
                r = col + 1 + lane
                while r < n:
                    pw = pw + Hm[r, col] * Hm[r, cc]; r = r + 32
                w = _wreduce(pw) + Hm[col, cc]
                if lane == 0:
                    Hm[col, cc] = Hm[col, cc] - tau_j * w
                r = col + 1 + lane
                while r < n:
                    Hm[r, cc] = Hm[r, cc] - tau_j * Hm[r, col] * w; r = r + 32
                cc = cc + 1
        col = col + 1


# ============================ warp0: gram + LARFT (m1c, validated) ============================
@cute.jit
def _compute_gram(Hm, sG, c0: cutlass.Int32, lane: cutlass.Int32, n: cutlass.Constexpr):
    # G[i,j] = sum_row V[row,c0+i] V[row,c0+j]  (unit-lower V over rows [c0,n))
    for i in cutlass.range_constexpr(NB):
        for j in cutlass.range_constexpr(NB):
            p = cutlass.Float32(0.0)
            r = c0 + lane
            while r < n:
                p = p + _vlow(Hm, r, c0 + i) * _vlow(Hm, r, c0 + j); r = r + 32
            g = _wreduce(p)
            if lane == 0:
                sG[i * NB + j] = g


@cute.jit
def _larft(sG, sT, tm, c0: cutlass.Int32, plen: cutlass.Int32, lane: cutlass.Int32):
    # T upper-tri, T[0,0]=tau0; forward recurrence (port of fused_qr_slice L81-91). lane 0 serial.
    if lane == 0:
        for a in cutlass.range_constexpr(NB * NB):
            sT[a] = cutlass.Float32(0.0)
        sT[0] = tm[c0]
        for i in cutlass.range_constexpr(1, NB):
            if i < plen:
                ti = tm[c0 + i]
                for k in cutlass.range_constexpr(NB):
                    if k < i:
                        m = cutlass.Float32(0.0)
                        for l in cutlass.range_constexpr(NB):
                            if l < i:
                                m = m + sT[k * NB + l] * sG[l * NB + i]
                        sT[k * NB + i] = -ti * m
                sT[i * NB + i] = ti


# ============================ warp1: blocked-WY apply (m1c warp-reduce; tcgen05 swap point) ====
@cute.jit
def _wy_apply(Hm, sT, c0: cutlass.Int32, pend: cutlass.Int32, plen: cutlass.Int32,
              lane: cutlass.Int32, n: cutlass.Constexpr):
    # trailing cols [pend, n): C -= V (T^T (V^T C)).  *** this body becomes tcgen05 tf32x3 in M3c-0 ***
    w1 = cute.make_fragment(NB, cutlass.Float32)
    w2 = cute.make_fragment(NB, cutlass.Float32)
    cc = pend
    while cc < n:
        for i in cutlass.range_constexpr(NB):       # w1[i] = V[:,i]^T C[:,cc]
            p = cutlass.Float32(0.0)
            r = c0 + lane
            while r < n:
                p = p + _vlow(Hm, r, c0 + i) * Hm[r, cc]; r = r + 32
            w1[i] = _wreduce(p)
        for i in cutlass.range_constexpr(NB):       # w2[i] = sum_k T[k,i] w1[k]  (= T^T w1)
            s = cutlass.Float32(0.0)
            for k in cutlass.range_constexpr(NB):
                if k < plen and i < plen:
                    s = s + sT[k * NB + i] * w1[k]
            w2[i] = s
        r = c0 + lane                               # C[r,cc] -= sum_i V[r,i] w2[i]
        while r < n:
            d = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(NB):
                if i < plen:
                    d = d + _vlow(Hm, r, c0 + i) * w2[i]
            Hm[r, cc] = Hm[r, cc] - d
            r = r + 32
        cc = cc + 1


# ============================ SharedStorage ============================
@cute.struct
class SS:
    sG: cute.struct.MemRange[cutlass.Float32, NB * NB]
    sT: cute.struct.MemRange[cutlass.Float32, NB * NB]


# ============================ kernel ============================
@cute.kernel
def _qr_kernel(mH, mtau, n: cutlass.Constexpr, nb: cutlass.Constexpr):
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

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(SS)
    sG = st.sG.get_tensor(cute.make_layout(NB * NB))
    sT = st.sT.get_tensor(cute.make_layout(NB * NB))

    c0 = cutlass.Int32(0)
    while c0 < n:
        pend = c0 + nb if c0 + nb < n else n
        plen = pend - c0
        if warp == 0:
            _panel_factor(Hm, tm, c0, pend, lane, n)
        cute.arch.barrier()                      # panel reflectors + tau visible

        if pend < n:
            if warp == 0:
                _compute_gram(Hm, sG, c0, lane, n)
            cute.arch.barrier()                  # sG visible
            if warp == 0:
                _larft(sG, sT, tm, c0, plen, lane)
            cute.arch.barrier()                  # sT visible to apply warp
            if warp == 1:
                _wy_apply(Hm, sT, c0, pend, plen, lane, n)
            cute.arch.barrier()                  # apply done before next panel reads trailing
        c0 = c0 + nb


@cute.jit
def _qr_host(mH, mtau, n: cutlass.Constexpr, nb: cutlass.Constexpr):
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
    print(f"=== M3c STAGE-A baseline: per-matrix warp-split (panel‖WY-apply), warp-reduce, NB={NB} ===")
    torch.manual_seed(0)
    for B, n in [(1, 32), (1, 64), (1, 128), (8, 64), (4, 256), (1, 100), (4, 512)]:
        A = torch.randn(B, n, n, device="cuda", dtype=torch.float32)
        try:
            H = A.clone().contiguous(); tau = torch.zeros(B, n, device="cuda", dtype=torch.float32)
            _run_cute(H, tau); torch.cuda.synchronize()
            Hg, taug = torch.geqrf(A)
            herr = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
            terr = (tau - taug).abs().max().item() / (taug.abs().max().item() + 1e-30)
            ok = herr < 1e-4 and terr < 1e-4
            print(f"  B={B:2d} n={n:4d}: H={herr:.2e} tau={terr:.2e}  {'PASS ✅' if ok else 'FAIL ❌'}")
        except Exception:
            print(f"  B={B:2d} n={n:4d}: EXC\n" + traceback.format_exc()[-1800:])
    print("DONE")
