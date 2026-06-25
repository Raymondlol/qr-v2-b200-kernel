"""Engine attack, step 1: find the BEST achievable tf32x3 trailing GEMM with
available tools (before any raw-PTX/TLX work), and quantify the recursive-blocking
(bigger-K) win. Baseline run showed: cuBLAS 1xTF32 = 21-34% peak on fat trailing,
6-7% on thin K=32; our fused tf32x3 Triton kernel ~6x slower than cuBLAS 1xTF32.

Section 1 (IMPL bake-off, tf32x3-accurate, fat shapes): fused-Triton-v1 (current
configs) vs fused-Triton-v2 (bigger tiles/stages) vs 3xcuBLAS hi/lo split vs
cuBLAS-1xTF32 (precision-loose ceiling). Which is fastest? Is fused even optimal?
Section 2 (K-SWEEP): same total work as one fat K=480 GEMM vs 15x K=32 GEMMs vs
cuBLAS, to quantify how much merging thin updates into fat-K (recursive blocking)
buys. All accuracy-checked vs fp32. No banned submission substrings.
"""
import torch, triton, triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
PEAK = 1100.0  # tf32 dense tensor-core TFLOP/s (Blackwell)


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, iters=40):
    for _ in range(5):
        try: fn()
        except Exception: return float("nan")
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000.0


def _kernel_factory(configs):
    @triton.autotune(configs=configs, key=['M', 'N', 'K'])
    @triton.jit
    def k(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
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
    return k


V1 = _kernel_factory([
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
])
V2 = _kernel_factory([
    triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=4),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_warps=8, num_stages=3),
    triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=4),
    triton.Config({'BM': 256, 'BN': 64, 'BK': 64}, num_warps=8, num_stages=3),
    triton.Config({'BM': 64, 'BN': 256, 'BK': 64}, num_warps=8, num_stages=4),
])


def make_runner(kern):
    def run(A, B):
        Bb, M, K = A.shape; N = B.shape[2]
        C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
        grid = lambda m: (Bb, triton.cdiv(M, m['BM']), triton.cdiv(N, m['BN']))
        kern[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride())
        return C
    return run


run_v1, run_v2 = make_runner(V1), make_runner(V2)


def split_tf32(x):
    hi = (x.view(torch.int32) & -8192).view(torch.float32)
    return hi, x - hi


def mm3_split(A, B):  # 3xcuBLAS hi/lo tf32x3
    Ah, Al = split_tf32(A); Bh, Bl = split_tf32(B)
    out = torch.matmul(Ah, Bh)
    out = torch.baddbmm(out, Ah, Bl); out = torch.baddbmm(out, Al, Bh)
    return out


def relerr(C, ref):
    return (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)


def bakeoff():
    print("\n========== 1) IMPL BAKE-OFF (tf32x3-accurate trailing GEMMs) ==========")
    print(f"{'shape (B,M,K,N)':24s} {'method':16s} {'us':>9s} {'TFLOPs':>8s} {'%pk*':>6s} {'relerr':>9s}")
    for name, Bb, M, K, N in [("n512 fat 640,512,128,384", 640, 512, 128, 384),
                              ("n1024 fat 60,1024,256,768", 60, 1024, 256, 768)]:
        A = torch.randn(Bb, M, K, device="cuda"); B = torch.randn(Bb, K, N, device="cuda")
        flops = 2.0 * Bb * M * N * K
        ref = torch.matmul(A.double(), B.double()).float()
        cands = [("fused-Triton-v1", run_v1), ("fused-Triton-v2", run_v2),
                 ("3xcuBLAS-split", mm3_split)]
        ceil = PEAK / 3.0
        print(f"{name}")
        for mlabel, fn in cands:
            t = time_fn(lambda fn=fn: fn(A, B)); tf = flops / (t * 1e-6) / 1e12
            print(f"{'':24s} {mlabel:16s} {t:9.1f} {tf:8.0f} {100*tf/ceil:5.0f}% {relerr(fn(A,B),ref):9.1e}")
        # cuBLAS 1xTF32 ceiling (precision-loose; only valid for n>=1024 in the real submission)
        t = time_fn(lambda: torch.matmul(A, B)); tf = flops / (t * 1e-6) / 1e12
        print(f"{'':24s} {'cuBLAS-1xTF32':16s} {t:9.1f} {tf:8.0f} {100*tf/PEAK:5.0f}% {relerr(torch.matmul(A,B),ref):9.1e}  (loose)")


def ksweep():
    print("\n========== 2) K-SWEEP: fat-K vs many thin-K (recursive-blocking motivation) ==========")
    Bb, M, N, Kt = 640, 512, 480, 480
    A = torch.randn(Bb, M, Kt, device="cuda"); B = torch.randn(Bb, Kt, N, device="cuda")
    flops = 2.0 * Bb * M * N * Kt
    ceil = PEAK / 3.0
    # one fat K=480 tf32x3 GEMM
    t_fat = time_fn(lambda: run_v2(A, B))
    # same work as 15 sequential K=32 slices accumulated (mimics thin within-panel updates)
    def thin():
        acc = run_v2(A[:, :, 0:32].contiguous(), B[:, 0:32, :].contiguous())
        for k in range(32, Kt, 32):
            acc = acc + run_v2(A[:, :, k:k+32].contiguous(), B[:, k:k+32, :].contiguous())
        return acc
    t_thin = time_fn(thin, iters=15)
    t_cu = time_fn(lambda: mm3_split(A, B))
    for lbl, t in [("fat K=480 (1 GEMM)", t_fat), ("15x thin K=32", t_thin), ("3xcuBLAS-split fat", t_cu)]:
        tf = flops / (t * 1e-6) / 1e12
        print(f"  {lbl:22s} {t:9.1f} us  {tf:6.0f} TFLOPs  {100*tf/ceil:4.0f}% of tf32x3 ceiling")
    print(f"  => fat/thin speedup = {t_thin/t_fat:.2f}x  (how much merging thin updates into one fat-K GEMM buys)")


def main():
    print("dev", torch.cuda.get_device_name(0), "| *%pk vs tf32x3 ceiling (peak/3); cuBLAS-1xTF32 vs full peak")
    bakeoff()
    ksweep()


if __name__ == "__main__":
    main()
