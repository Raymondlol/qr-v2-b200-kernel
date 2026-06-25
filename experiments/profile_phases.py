"""Engine attack, step 2: where does the n=512 / n=1024 wall-clock ACTUALLY go?
Before investing in a warp-specialized trailing-GEMM engine we must confirm the
trailing GEMM dominates -- the panel-warps win (5x on n=1024) came from the PANEL,
not the trailing, so the panel / T-solve may be a large share.

Replicates submission.py's two-level factorization EXACTLY (same kernels, same
_blocking, same precision routing) but accumulates per-phase CUDA-event time:
  panel   = the fused _panel_kernel launches
  narrow  = within-super-panel block-reflector applies (K=ib)
  fat     = the fat K=NB trailing applies
and inside every apply: gram(VtV) / Tsolve(triangular) / gemmW (Vt@C) / gemmU (V@Y).
No banned submission substrings. Microbench only.
"""
import torch, triton, triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

T = {}  # phase -> accumulated ms
def _ev():
    e = torch.cuda.Event(enable_timing=True); e.record(); return e


# ---- tf32x3 fused batched GEMM (identical to submission _bmm3) ----
@triton.autotune(configs=[
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
    triton.Config({'BM': 32, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
], key=['M', 'N', 'K'])
@triton.jit
def _bmm_x3_kernel(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0); pm = tl.program_id(1); pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = A + pb * sab + (rm[:, None] * sam + rk[None, :] * sak)
    bp = B + pb * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(bp, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32x3")
        ap += BK * sak; bp += BK * sbk
    cp = C + pb * scb + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(cp, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _bmm3(A, B):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = lambda m: (Bb, triton.cdiv(M, m['BM']), triton.cdiv(N, m['BN']))
    _bmm_x3_kernel[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride())
    return C


def _split_tf32(x):
    hi = (x.view(torch.int32) & -8192).view(torch.float32); return hi, x - hi
def _mm3(A, B):
    Ah, Al = _split_tf32(A); Bh, Bl = _split_tf32(B)
    out = torch.matmul(Ah, Bh)
    out = torch.baddbmm(out, Ah, Bl); out = torch.baddbmm(out, Al, Bh); return out

_BIG_X3 = False
def _mm(A, B):
    if _BIG_X3 and A.is_cuda: return _bmm3(A, B)
    return torch.matmul(A, B)


@triton.jit
def _panel_kernel(Hptr, tauptr, N, K, M, BWID, BN: tl.constexpr, BCOLS: tl.constexpr):
    bid = tl.program_id(0)
    Hb = Hptr + bid * N * N; taub = tauptr + bid * N
    rar = tl.arange(0, BN); car = tl.arange(0, BCOLS)
    rmask = rar < M; cmask = car < BWID
    ptr = Hb + (K + rar)[:, None] * N + (K + car)[None, :]
    tmask = rmask[:, None] & cmask[None, :]
    tile = tl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        colj = tl.sum(tl.where(car[None, :] == j, tile, 0.0), axis=1)
        alpha = tl.sum(tl.where(rar == j, colj, 0.0))
        xnorm2 = tl.sum(tl.where(rar > j, colj * colj, 0.0))
        normfull = tl.sqrt(alpha * alpha + xnorm2)
        sgn = tl.where(alpha >= 0.0, 1.0, -1.0); beta = -sgn * normfull
        need = xnorm2 > 0.0
        scale = tl.where(need, 1.0 / (alpha - beta), 0.0)
        tau_j = tl.where(need, (beta - alpha) / beta, 0.0)
        v = tl.where(rar > j, colj * scale, 0.0); v = tl.where(rar == j, 1.0, v)
        w = tl.sum(v[:, None] * tile, axis=0)
        upd = tile - tau_j * (v[:, None] * w[None, :])
        tile = tl.where(car[None, :] > j, upd, tile)
        diagval = tl.where(need, beta, alpha)
        newcol = tl.where(rar > j, colj * scale, colj); newcol = tl.where(rar == j, diagval, newcol)
        tile = tl.where(car[None, :] == j, newcol[:, None], tile)
        tl.store(taub + K + j, tau_j, mask=(j < BWID))
    tl.store(ptr, tile, mask=tmask)


def _apply_block(H, col, b, tau, c0, c1):
    if c1 <= c0: return
    P = H[:, col:, col:col + b]
    V = torch.tril(P[:, :, :b], diagonal=-1).clone()
    idx = torch.arange(b, device=H.device); V[:, idx, idx] = 1.0
    tau_blk = tau[:, col:col + b]
    e0 = _ev(); G = _mm3(V.transpose(1, 2), V); e1 = _ev()
    nz = tau_blk != 0
    inv_tau = torch.where(nz, 1.0 / torch.where(nz, tau_blk, torch.ones_like(tau_blk)), torch.full_like(tau_blk, 1e30))
    M = torch.triu(G, diagonal=1); M[:, idx, idx] = inv_tau
    C = H[:, col:, c0:c1]
    e2 = _ev(); W = _mm(V.transpose(1, 2), C); e3 = _ev()
    Y = torch.linalg.solve_triangular(M.transpose(1, 2), W, upper=False); e4 = _ev()
    C.sub_(_mm(V, Y)); e5 = _ev()
    torch.cuda.synchronize()
    T['gram'] = T.get('gram', 0) + e0.elapsed_time(e1)
    T['gemmW'] = T.get('gemmW', 0) + e2.elapsed_time(e3)
    T['Tsolve'] = T.get('Tsolve', 0) + e3.elapsed_time(e4)
    T['gemmU'] = T.get('gemmU', 0) + e4.elapsed_time(e5)


def _blocking(n):
    if n <= 128: return (16, 16)
    if n <= 256: return (64, 64)
    if n <= 512: return (128, 64)
    if n <= 1024: return (256, 32)
    if n <= 2048: return (256, 16)
    return (256, 8)


def factor_timed(A):
    B, n, _ = A.shape
    H = A.clone(); tau = torch.zeros(B, n, dtype=A.dtype, device=A.device)
    NB, ib = _blocking(n)
    k = 0
    while k < n:
        nb = min(NB, n - k); j = 0
        while j < nb:
            b = min(ib, nb - j); col = k + j
            m = n - col; BN = triton.next_power_of_2(m); BCOLS = triton.next_power_of_2(b)
            nw = 4 if BN <= 128 else (8 if BN <= 512 else (16 if BN <= 1024 else 32))
            e0 = _ev(); _panel_kernel[(B,)](H, tau, n, col, m, b, BN=BN, BCOLS=BCOLS, num_warps=nw); e1 = _ev()
            torch.cuda.synchronize(); T['panel'] = T.get('panel', 0) + e0.elapsed_time(e1)
            if j + b < nb:
                before = {kk: T.get(kk, 0) for kk in ('gram', 'gemmW', 'Tsolve', 'gemmU')}
                _apply_block(H, col, b, tau, col + b, k + nb)
                for kk in before: T['narrow'] = T.get('narrow', 0) + (T[kk] - before[kk])
            j += b
        if k + nb < n:
            before = {kk: T.get(kk, 0) for kk in ('gram', 'gemmW', 'Tsolve', 'gemmU')}
            _apply_block(H, k, nb, tau, k + nb, n)
            for kk in before: T['fat'] = T.get('fat', 0) + (T[kk] - before[kk])
        k += nb
    return H, tau


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def main():
    global _BIG_X3, T
    print("dev", torch.cuda.get_device_name(0))
    for (B, n, big) in [(640, 512, True), (60, 1024, False)]:
        _BIG_X3 = big
        A = torch.randn(B, n, n, device="cuda")
        for _ in range(3): factor_timed(A.clone()); T = {}  # warmup, reset
        reps = 5
        for _ in range(reps):
            clear_l2(); factor_timed(A.clone())
        tot = sum(T[k] for k in ('panel', 'narrow', 'fat'))
        print(f"\n=== n={n} b={B}  (_BIG_X3={big}, trailing={'tf32x3' if big else '1xTF32'})  "
              f"avg over {reps} reps ===")
        for k in ('panel', 'narrow', 'fat'):
            print(f"  {k:8s}: {T[k]/reps:8.2f} ms  ({100*T[k]/tot:4.0f}% of factor)")
        print(f"  -- sub-phase split (gram/gemmW/Tsolve/gemmU, summed over all applies): --")
        for k in ('gram', 'gemmW', 'Tsolve', 'gemmU'):
            print(f"  {k:8s}: {T.get(k,0)/reps:8.2f} ms")
        print(f"  TOTAL panel+narrow+fat: {tot/reps:8.2f} ms")
        T = {}


if __name__ == "__main__":
    main()
