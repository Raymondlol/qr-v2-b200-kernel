"""DEEP-DIVE 1 — roofline-by-shapes (FREE, pure arithmetic, no GPU).

Replicates submission.py::_factor_custom's loop to enumerate EVERY GEMM (M,N,K) in the
n<=512 / n=1024 factorization, computes arithmetic intensity (AI = useful FLOPs / HBM bytes)
per kernel, and classifies memory- vs compute-bound against the B200 tf32x3 ridge.

Goal: turn the profiler PREDICTIVE. (a) Which kernels are memory-bound (low AI) -> a traffic
cut like implicit-V CAN help there. (b) Does small-n (176/352) differ from n=512? (c) How many
GEMM launches per n -> the launch-bound story for small-n.

Caveat: these batched shapes are K-POOR; docs say tf32x3 achieves only ~40 TF/s (~24% of the
~165 TF/s effective peak), i.e. they're partly OCCUPANCY/latency-bound, not cleanly on the
roofline. So AI<ridge = clearly memory-side; AI>>ridge = compute-side; AI~ridge or low achieved
efficiency = latency/occupancy-bound. Still, the AI RANKING + launch counts are exact.
"""

# ---- B200 roofline constants (tf32x3 useful-flops basis) ----
TF32_DENSE_TFLOPS = 495.0          # B200 tf32 dense tensor-core (sparse 2x = ~990)
TF32X3_EFF_TFLOPS = TF32_DENSE_TFLOPS / 3.0   # tf32x3 = 3 tensor passes for the useful GEMM
HBM_GBPS = 8000.0                  # B200 HBM3e ~8 TB/s
RIDGE = (TF32X3_EFF_TFLOPS * 1e12) / (HBM_GBPS * 1e9)   # FLOP/byte at which compute==memory


def next_pow2(m):
    return 1 << (m - 1).bit_length()


def _blocking(n):
    if n <= 128: return (16, 16)
    if n <= 256: return (64, 64)
    if n <= 512: return (128, 64)
    if n <= 1024: return (256, 128)
    if n <= 2048: return (256, 128)
    return (256, 8)


def _max_ib(m, ib_max):
    BN = next_pow2(m)
    ib = ib_max
    while ib > 1 and BN * next_pow2(ib) > 48 * 1024:
        ib //= 2
    return ib


def gemms_of_apply(col, b, c0, c1, n):
    """The 3 GEMMs of _apply_block(col,b,[c0:c1]): gram V^TV, W=V^TC, update C-=VY."""
    m = n - col
    Wd = c1 - c0
    if Wd <= 0:
        return []
    return [
        ("gram   V^TV ", b, b, m, False),   # M,N,K, is_sub
        ("W      V^TC ", b, Wd, m, False),
        ("update C-=VY", m, Wd, b, True),
    ]


def factor_gemms(n, big_x3=True):
    """Replicate _factor_custom; emit every GEMM + count panel/apply launches."""
    NB, ib_max = _blocking(n)
    gemms = []
    n_panel = 0
    n_apply = 0
    k = 0
    while k < n:
        nb = min(NB, n - k)
        j = 0
        while j < nb:
            col = k + j
            m = n - col
            b = min(_max_ib(m, ib_max), nb - j)
            n_panel += 1                              # _panel_kernel (latency-bound, not a GEMM)
            if j + b < nb:                            # within-super-panel apply
                gemms += gemms_of_apply(col, b, col + b, k + nb, n)
                n_apply += 1
            j += b
        if k + nb < n:                                # fat trailing apply
            gemms += gemms_of_apply(k, nb, k + nb, n, n)
            n_apply += 1
        k += nb
    return gemms, n_panel, n_apply


def ai_of(M, N, K, is_sub):
    flops = 2.0 * M * N * K                            # useful GEMM flops
    rd = 4.0 * (M * K + K * N)                         # read A,B (fp32)
    wr = 4.0 * M * N * (2 if is_sub else 1)            # write C (sub also reads C)
    return flops, rd + wr, flops / (rd + wr)


def main():
    print(f"B200 tf32x3 ridge ~= {RIDGE:.1f} FLOP/byte "
          f"(eff peak {TF32X3_EFF_TFLOPS:.0f} TF/s / HBM {HBM_GBPS/1000:.0f} TB/s). "
          f"AI<ridge => memory-side; AI>>ridge => compute-side.\n")
    for n in (176, 352, 512, 1024):
        gemms, n_panel, n_apply = factor_gemms(n)
        print(f"================ n={n} ================")
        print(f"  panel-kernel launches: {n_panel:2d}   apply_block calls: {n_apply:2d}   "
              f"GEMM launches: {len(gemms):2d}  (= {n_apply}x3)")
        tot_f = tot_b = 0.0
        memside = compside = 0
        # aggregate by unique shape
        agg = {}
        for kind, M, N, K, sub in gemms:
            f, b, ai = ai_of(M, N, K, sub)
            tot_f += f; tot_b += b
            agg.setdefault((kind, M, N, K, sub), [0, ai])
            agg[(kind, M, N, K, sub)][0] += 1
            if ai < RIDGE: memside += 1
            else: compside += 1
        print(f"  {'kernel':12s} {'M':>5}{'N':>5}{'K':>5} {'cnt':>4} {'AI':>7}  bound")
        for (kind, M, N, K, sub), (cnt, ai) in sorted(agg.items(), key=lambda x: -x[1][1]):
            tag = "MEM " if ai < RIDGE else "comp"
            print(f"  {kind:12s} {M:>5}{N:>5}{K:>5} {cnt:>4} {ai:>7.1f}  {tag}")
        eff_ai = tot_f / tot_b
        print(f"  ---- GEMM totals: useful {tot_f/1e9:.2f} GFLOP, HBM {tot_b/1e6:.1f} MB/matrix, "
              f"flop-weighted AI = {eff_ai:.1f} ({'MEM-side' if eff_ai<RIDGE else 'compute-side'})")
        print(f"       GEMM-shape instances memory-bound: {memside}/{memside+compside}\n")


if __name__ == "__main__":
    main()
