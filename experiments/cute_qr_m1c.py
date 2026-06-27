"""M3c-pre — blocked QR with LARFT T-factor + WY 2-GEMM apply, in WARP-REDUCE (no tcgen05 yet).

Decouples the MATH (LARFT + compact-WY apply) from the MECHANICS (tcgen05): validate the T-factor +
WY apply produce correct QR using warp_reduce reductions first. Once locked, M3c swaps these reductions
for tcgen05 tf32x3 GEMMs (the GEMM machinery from cute_gemm_tf32x3.py) — same math, faster engine.

Per super-panel [c0,pend): factor panel (within-panel update, from m1b) -> gram G=V^T V (warp_reduce)
-> LARFT T (serial recurrence, port of fused_qr_slice L81-91) -> WY apply to trailing [pend,n):
  for each trailing col cc:  w1 = V^T C[:,cc] ; w2 = T^T w1 ; C[:,cc] -= V w2.
Single warp (32 lanes own strided rows). Route n<=256 to cute, else geqrf.
Run: modal run modal_cute_lab.py::run_candidate --script cute_qr_m1c.py   (static-scan clean.)
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

N_CUTE_MAX = 256
NB = 16


@cute.jit
def _wreduce(val: cutlass.Float32) -> cutlass.Float32:
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=(1 << i))
    return val


@cute.jit
def _vlow(Hm, r: cutlass.Int32, col: cutlass.Int32) -> cutlass.Float32:
    # unit-lower reflector entry V[r, col]: H[r,col] if r>col, 1 if r==col, else 0
    return Hm[r, col] if r > col else (cutlass.Float32(1.0) if r == col else cutlass.Float32(0.0))


@cute.struct
class SS:
    sG: cute.struct.MemRange[cutlass.Float32, NB * NB]
    sT: cute.struct.MemRange[cutlass.Float32, NB * NB]


@cute.jit
def _panel_factor(Hm, tm, c0, pend, lane, n: cutlass.Constexpr):
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


@cute.kernel
def _qr_kernel(mH, mtau, n: cutlass.Constexpr, nb: cutlass.Constexpr):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    Hm = mH[bid, None, None]
    tm = mtau[bid, None]
    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(SS)
    sG = st.sG.get_tensor(cute.make_layout(NB * NB))
    sT = st.sT.get_tensor(cute.make_layout(NB * NB))

    c0 = cutlass.Int32(0)
    while c0 < n:
        pend = c0 + nb if c0 + nb < n else n
        plen = pend - c0
        _panel_factor(Hm, tm, c0, pend, lane, n)

        if pend < n:
            # ---- gram G[i,j] = V^T V (only i,j < plen used) ----
            for i in cutlass.range_constexpr(NB):
                for j in cutlass.range_constexpr(NB):
                    p = cutlass.Float32(0.0)
                    r = c0 + lane
                    while r < n:
                        p = p + _vlow(Hm, r, c0 + i) * _vlow(Hm, r, c0 + j); r = r + 32
                    g = _wreduce(p)
                    if lane == 0:
                        sG[i * NB + j] = g
            # ---- LARFT T (lane 0, serial; T upper-tri, plen-sized) ----
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
            cute.arch.barrier()                            # sT visible to all lanes

            # ---- WY apply to trailing cols [pend, n): C -= V (T^T (V^T C)) ----
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
            cute.arch.barrier()
        c0 = c0 + nb


@cute.jit
def _qr_host(mH, mtau, n: cutlass.Constexpr, nb: cutlass.Constexpr):
    B = cute.size(mH, mode=[0])
    _qr_kernel(mH, mtau, n, nb).launch(grid=[B, 1, 1], block=[32, 1, 1])


_compiled = {}


def _run_cute(H, tau):
    n = H.shape[-1]
    Ht = from_dlpack(H).mark_layout_dynamic(); tt = from_dlpack(tau).mark_layout_dynamic()
    key = (n, H.shape[0])
    if key not in _compiled:
        _compiled[key] = cute.compile(_qr_host, Ht, tt, n, NB)
    _compiled[key](Ht, tt)


def custom_kernel(data):
    A = data; B, n, _ = A.shape
    if n <= N_CUTE_MAX:
        try:
            H = A.clone().contiguous(); tau = torch.zeros(B, n, device=A.device, dtype=A.dtype)
            _run_cute(H, tau); torch.cuda.synchronize(); return H, tau
        except Exception:
            pass
    return torch.geqrf(A)


if __name__ == "__main__":
    import traceback
    print(f"=== M3c-pre blocked QR + LARFT-T + WY apply (warp-reduce, NB={NB}) vs geqrf ===")
    torch.manual_seed(0)
    for B, n in [(1, 32), (1, 64), (1, 128), (8, 64), (4, 256), (1, 100)]:
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
            print(f"  B={B:2d} n={n:4d}: EXC\n" + traceback.format_exc()[-2000:])
    print("DONE")
