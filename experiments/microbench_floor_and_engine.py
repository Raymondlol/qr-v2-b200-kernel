"""Two decisive B200 probes in one run (saves a container spin-up):

A) FLOOR PROBE -- can a custom BLOCKED Cholesky (right-looking, trailing update =
   tensor-core GEMM) beat the monolithic cuSOLVER torch.linalg.cholesky at the
   geqrf-floor shapes (n=2048 b=8, n=4096 b=2)? The prior run measured cuSOLVER
   chol at n=4096 b=2 = 24.8ms = ~1.8 TFLOP/s = critical-path-bound (same wall as
   the geqrf floor). If blocking (GEMM trailing) does NOT beat it, a library-based
   CholeskyQR floor-breaker is dead and only a fully hand-written tensor-core chol
   could rescue it. Also times the "apply" (Rinv + A@Rinv GEMM) as the cheap part.

B) ENGINE BASELINE -- how far below B200 peak is the BEST batched GEMM (cuBLAS
   torch.matmul, and our fused tf32x3 tl.dot) on the actual TRAILING shapes? This
   quantifies the warp-specialized-engine headroom: if cuBLAS already runs these
   shapes near peak, the 6x gap is elsewhere; if it is far below (tall-skinny,
   low-K), the engine is the real lever.

Contains neither of the two banned submission substrings. Microbench only.
"""
import torch, triton, triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
B200_TF32_PEAK = 1100.0  # ~TFLOP/s dense tensor-core tf32 (Blackwell SXM, no sparsity)


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, iters=30):
    for _ in range(4):
        try: fn()
        except Exception: return float("nan")
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000.0  # median us


# ---- A) blocked right-looking Cholesky, trailing update = batched GEMM ----
def blocked_chol(G, bs):
    B, n, _ = G.shape
    R = G.clone()
    for k in range(0, n, bs):
        e = min(k + bs, n)
        Rkk = torch.linalg.cholesky(R[:, k:e, k:e], upper=True)
        R[:, k:e, k:e] = Rkk
        if e < n:
            R[:, k:e, e:] = torch.linalg.solve_triangular(
                Rkk.transpose(-1, -2), R[:, k:e, e:], upper=False)
            R[:, e:, e:] -= R[:, k:e, e:].transpose(-1, -2) @ R[:, k:e, e:]
    return torch.triu(R)


def floor_probe():
    print("\n==================== A) FLOOR PROBE: blocked vs cuSOLVER Cholesky ====================")
    print(f"{'shape':14s} {'cuSOLVER':>10s} {'blk256':>10s} {'blk512':>10s} {'blk1024':>10s} "
          f"{'Rinv':>8s} {'A@Rinv':>8s}  {'relerr':>9s} {'best_vs_cuSOLVER':>16s}")
    for (B, n) in [(8, 2048), (2, 4096)]:
        A = torch.randn(B, n, n, device="cuda")
        eye = torch.eye(n, device="cuda").expand(B, n, n).contiguous()
        G = A.transpose(-1, -2) @ A + (n * 1e-3) * eye   # PD for clean timing
        R = torch.linalg.cholesky(G, upper=True)
        t_cu = time_fn(lambda: torch.linalg.cholesky(G, upper=True))
        blk = {bs: time_fn(lambda bs=bs: blocked_chol(G, bs)) for bs in (256, 512, 1024)}
        # apply cost: Rinv (triangular-inverse) then A@Rinv (GEMM)
        t_rinv = time_fn(lambda: torch.linalg.solve_triangular(R, eye, upper=True))
        Rinv = torch.linalg.solve_triangular(R, eye, upper=True)
        t_app = time_fn(lambda: A @ Rinv)
        relerr = (blocked_chol(G, 512) - R).abs().max().item() / (R.abs().max().item() + 1e-9)
        best = min(blk.values())
        print(f"b={B:2d} n={n:5d}   {t_cu:10.1f} {blk[256]:10.1f} {blk[512]:10.1f} {blk[1024]:10.1f} "
              f"{t_rinv:8.1f} {t_app:8.1f}  {relerr:9.1e} {t_cu/best:15.2f}x")
    print("  (best_vs_cuSOLVER > 1 => blocking BEATS cuSOLVER => custom tensor-core chol has hope)")


# ---- B) batched tf32x3 GEMM (our trailing engine) ----
@triton.autotune(configs=[
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
], key=['M', 'N', 'K'])
@triton.jit
def _x3(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
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


def bmm_x3(A, B):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = lambda m: (Bb, triton.cdiv(M, m['BM']), triton.cdiv(N, m['BN']))
    _x3[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride())
    return C


def engine_baseline():
    print("\n==================== B) ENGINE BASELINE: trailing-GEMM efficiency ====================")
    print(f"peak tf32 assumed {B200_TF32_PEAK:.0f} TFLOP/s; tf32x3 does 3x the tf32 work "
          f"(so its fp32-accuracy ceiling ~ {B200_TF32_PEAK/3:.0f} TFLOP/s)")
    print(f"{'trailing shape':26s} {'cuBLAS 1xTF32':>16s} {'fused tf32x3':>16s}")
    print(f"{'(B,M,K,N)':26s} {'us / TFLOPs / %':>16s} {'us / TFLOPs / %':>16s}")
    shapes = [
        ("n=512 fat  640,512,128,384", 640, 512, 128, 384),
        ("n=512 thin 640,512, 32,480", 640, 512, 32, 480),
        ("n=1024 fat  60,1024,256,768", 60, 1024, 256, 768),
        ("n=1024 thin 60,1024, 32,992", 60, 1024, 32, 992),
    ]
    for name, Bb, M, K, N in shapes:
        A = torch.randn(Bb, M, K, device="cuda")
        Bm = torch.randn(Bb, K, N, device="cuda")
        flops = 2.0 * Bb * M * N * K
        t_cu = time_fn(lambda: torch.matmul(A, Bm))
        t_x3 = time_fn(lambda: bmm_x3(A, Bm))
        cu_tf = flops / (t_cu * 1e-6) / 1e12
        x3_tf = flops / (t_x3 * 1e-6) / 1e12
        print(f"{name:26s} {t_cu:6.1f}/{cu_tf:6.0f}/{100*cu_tf/B200_TF32_PEAK:4.0f}% "
              f"   {t_x3:6.1f}/{x3_tf:6.0f}/{100*x3_tf/(B200_TF32_PEAK/3):4.0f}%")
    print("  (cuBLAS % near 100 => GEMM already near peak, 6x gap is overhead/panel, not the GEMM;")
    print("   cuBLAS % low on these tall-skinny low-K shapes => real warp-specialized-engine headroom)")


def main():
    print("dev", torch.cuda.get_device_name(0), "| tf32", torch.backends.cuda.matmul.allow_tf32)
    floor_probe()
    engine_baseline()


if __name__ == "__main__":
    main()
