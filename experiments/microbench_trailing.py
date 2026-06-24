"""Phase-2 gate microbench: single-CTA-per-matrix trailing GEMM vs batched tf32x3.

The full mega-kernel does each matrix's trailing update with ONE CTA. The current
path does it as a BATCHED GEMM tiled across many CTAs. This isolates the question:
does single-CTA-per-matrix trailing lose to the batched GEMM? (GO for full mega if
single-CTA >= ~0.9x batched.) Both are tf32x3 so the Modal-vs-official ratio bias
applies equally -> the ratio is trustworthy.
"""
import torch, triton, triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True


# ---------- batched tf32x3 GEMM (the current _bmm3 from submission.py) ----------
@triton.autotune(configs=[
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
], key=['M', 'N', 'K'])
@triton.jit
def _batched_kernel(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_b = tl.program_id(0); pid_m = tl.program_id(1); pid_n = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = A + pid_b * sab + (rm[:, None] * sam + rk[None, :] * sak)
    bp = B + pid_b * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(bp, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32x3")
        ap += BK * sak; bp += BK * sbk
    cp = C + pid_b * scb + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(cp, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def batched_gemm(A, B):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = lambda m: (Bb, triton.cdiv(M, m['BM']), triton.cdiv(N, m['BN']))
    _batched_kernel[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride())
    return C


# ---------- single-CTA-per-matrix tf32x3 GEMM (the full-mega trailing) ----------
@triton.autotune(configs=[
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=2),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=2),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=8, num_stages=2),
    triton.Config({'BM': 64, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=2),
], key=['M', 'N', 'K'])
@triton.jit
def _single_kernel(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_b = tl.program_id(0)                      # ONE CTA owns matrix pid_b
    rk = tl.arange(0, BK)
    for m0 in range(0, M, BM):
        rm = m0 + tl.arange(0, BM)
        for n0 in range(0, N, BN):
            rn = n0 + tl.arange(0, BN)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            ap = A + pid_b * sab + (rm[:, None] * sam + rk[None, :] * sak)
            bp = B + pid_b * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
            for k0 in range(0, K, BK):
                a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
                b = tl.load(bp, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
                acc += tl.dot(a, b, input_precision="tf32x3")
                ap += BK * sak; bp += BK * sbk
            cp = C + pid_b * scb + (rm[:, None] * scm + rn[None, :] * scn)
            tl.store(cp, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def single_cta_gemm(A, B):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    _single_kernel[(Bb,)](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride())
    return C


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, A, B, iters=100):
    for _ in range(5):
        fn(A, B)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(A, B); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1000  # median us


def main():
    print("dev", torch.cuda.get_device_name(0))
    # (name, B, M, K, N): representative trailing-GEMM shapes
    shapes = [
        ("n=512 fat   (NB=128)", 640, 512, 128, 384),
        ("n=512 mega  (NB=32) ", 640, 512, 32, 480),
        ("n=1024 fat  (NB=256)", 60, 1024, 256, 768),
        ("n=1024 mega (NB=32) ", 60, 1024, 32, 992),
    ]
    print(f"\n{'shape':24s} {'batched µs':>12s} {'1-CTA µs':>12s} {'1CTA/batched':>14s}")
    for name, Bb, M, K, N in shapes:
        A = torch.randn(Bb, M, K, device="cuda")
        B = torch.randn(Bb, K, N, device="cuda")
        # correctness sanity (tf32x3 ~ fp32)
        cb = batched_gemm(A, B); cs = single_cta_gemm(A, B)
        err = (cb - cs).abs().max().item() / (cb.abs().max().item() + 1e-9)
        tb = time_fn(batched_gemm, A, B)
        ts = time_fn(single_cta_gemm, A, B)
        flag = "GO" if ts <= 1.11 * tb else "no-go"
        print(f"{name:24s} {tb:12.1f} {ts:12.1f} {ts/tb:13.2f}x  {flag}  relerr={err:.1e}")


if __name__ == "__main__":
    main()
