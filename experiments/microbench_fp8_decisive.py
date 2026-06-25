"""DECISIVE fp8/fp4 feasibility microbench for qr_v2 (B200).

Settles the question the prior "fp8 DEAD" verdict left open: with the REAL
fp32-panel pipeline (not the fp64-panel scratch sim), how many fp8(e4m3) Ozaki
dots does the trailing/gram need to hold the mixed@640 factor-residual gate at a
SAFE >=2x margin -- and is that dot-count, at realized B200 fp8 throughput,
faster than tf32x3 (3 tf32 dots)?

PART A (accuracy, decisive): blocked compact-WY Householder QR with the PANEL in
  FP32 (matching submission.py) and the trailing/gram/W GEMMs routed through a
  simulated low-precision Ozaki product. Runs the REAL competition checker
  (reference.check_implementation) on the real mixed n=512 batch=640 input.
  Calibration anchors that MUST reproduce or the harness is untrustworthy:
    fp32 trailing      -> sfr ~ 0.016
    tf32x3 trailing    -> sfr ~ 10   (2.0x margin = the documented safe floor)
    1xTF32 trailing    -> sfr ~ 18-21 (marginal/fail)
  Then sweeps fp8 e4m3 at 6/10/13/15 dots -> min dots for >=2x margin.

PART B (throughput): realized TFLOP/s on the REAL trailing + gram shapes for the
  tf32x3 fused kernel vs a batched mxfp8 (tl.dot_scaled) dot vs 1xTF32 cuBLAS, so
  the crossover (min_dots x cost-per-dot vs tf32x3) is grounded in measurement,
  not the 4x-peak assumption. Guarded so PART A always reports even if B fails.

Run: modal run modal_microbench.py --script microbench_fp8_decisive.py
"""
import sys, os, time, re
import torch

sys.path.insert(0, "/work")
sys.path.insert(0, "/work/harness")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
import reference

torch.set_grad_enabled(False)
DEV = "cuda"
F64 = torch.float64
GATE = 20.0


# ============================================================================
# Low-precision element emulation (round-to-nearest), GPU-safe (no CPU tensors)
# ============================================================================
def _round_mantissa(x, mbits, emin, max_normal):
    x = x.to(F64)
    sign = torch.sign(x)
    ax = x.abs()
    nz = ax > 0
    e = torch.floor(torch.log2(ax.clamp_min(1e-300)))
    e_eff = torch.clamp(e, min=emin)
    scale = torch.exp2(mbits - e_eff)              # 2^(mbits - e_eff), on-device
    q = torch.round(ax * scale) / scale
    q = torch.clamp(q, max=max_normal)
    out = torch.where(nz, q, torch.zeros_like(ax))
    return (sign * out).to(torch.float32)


FORMATS = {
    "e4m3": dict(mbits=3, emin=-6, max_normal=448.0),
    "e5m2": dict(mbits=2, emin=-14, max_normal=57344.0),
    "e2m1": dict(mbits=1, emin=0, max_normal=6.0),
}


def to_lowp(x, fmt):
    p = FORMATS[fmt]
    return _round_mantissa(x, p["mbits"], p["emin"], p["max_normal"])


def ozaki_split(A, fmt, k, role, scale_mode="per_tensor"):
    p = FORMATS[fmt]
    max_normal = p["max_normal"]
    A = A.to(F64)
    slices = []
    R = A.clone()
    for _ in range(k):
        amax = R.abs()
        if scale_mode == "per_tensor":
            m = amax.amax(dim=(-2, -1), keepdim=True)
        else:  # per_vec: per-row for left (max over K=-1), per-col for right (max over K=-2)
            cdim = -1 if role == "left" else -2
            m = amax.amax(dim=cdim, keepdim=True)
        m = m.clamp_min(1e-300)
        scale = m / max_normal
        q = to_lowp((R / scale).to(torch.float32), fmt).to(F64)
        slices.append((q, scale))
        R = R - q * scale
    return slices


def _to_tf32(x):
    xi = x.to(torch.float32).view(torch.int32)
    xi = (xi + (1 << 12)) & (~((1 << 13) - 1))     # round-to-nearest, keep 10 mant bits
    return xi.view(torch.float32)


def _mm32(A, B):
    # TRUE fp32 accumulate (tf32 disabled globally in part A) of fp32 inputs.
    return (A.to(torch.float32) @ B.to(torch.float32)).to(F64)


def tf32_dot(A, B):
    return _mm32(_to_tf32(A.to(torch.float32)), _to_tf32(B.to(torch.float32)))


def tf32x3_dot(A, B):
    Ah = _to_tf32(A.to(torch.float32)); Al = _to_tf32(A.to(torch.float32) - Ah)
    Bh = _to_tf32(B.to(torch.float32)); Bl = _to_tf32(B.to(torch.float32) - Bh)
    return _mm32(Ah, Bh) + _mm32(Ah, Bl) + _mm32(Al, Bh)


def ozaki_dot(A, B, fmt, kA, kB, T, scaleA, scaleB):
    sa = ozaki_split(A, fmt, kA, "left", scaleA)
    sb = ozaki_split(B, fmt, kB, "right", scaleB)
    out = None
    ndots = 0
    for i, (Ai, scA) in enumerate(sa):
        for j, (Bj, scB) in enumerate(sb):
            if i + j > T:
                continue
            prod = _mm32(Ai, Bj) * (scA * scB)
            out = prod if out is None else out + prod
            ndots += 1
    return out, ndots


# ============================================================================
# Blocked compact-WY Householder QR -- PANEL IN FP32 (the key fix vs scratch sim)
# ============================================================================
def panel_factor(P, tau_out):
    B, m, b = P.shape
    for j in range(b):
        alpha = P[:, j, j]
        x = P[:, j + 1:, j]
        xnorm = torch.linalg.vector_norm(x, dim=1)
        normfull = torch.sqrt(alpha * alpha + xnorm * xnorm)
        sign = torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        beta = -sign * normfull
        mask = xnorm > 0
        safe_denom = torch.where(mask, alpha - beta, torch.ones_like(alpha))
        safe_beta = torch.where(mask, beta, torch.ones_like(beta))
        tau_j = torch.where(mask, (beta - alpha) / safe_beta, torch.zeros_like(alpha))
        vtail = torch.where(mask.unsqueeze(1), x / safe_denom.unsqueeze(1), torch.zeros_like(x))
        P[:, j + 1:, j] = vtail
        P[:, j, j] = torch.where(mask, beta, alpha)
        tau_out[:, j] = tau_j
        if j < b - 1:
            ones = torch.ones(B, 1, dtype=P.dtype, device=P.device)
            V = torch.cat([ones, vtail], dim=1)
            sub = P[:, j:, j + 1:]
            w = torch.einsum('bm,bmt->bt', V, sub)
            sub.sub_(tau_j.view(B, 1, 1) * torch.einsum('bm,bt->bmt', V, w))


def apply_block(H, col, b, tau, c0, c1, cfg):
    if c1 <= c0:
        return
    P = H[:, col:, col:col + b]
    V = torch.tril(P[:, :, :b], diagonal=-1).clone()
    V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    tau_blk = tau[:, col:col + b]
    Vt = V.transpose(1, 2)
    prec = cfg.get("prec")

    def gemm(X, Y, lowp_flag):
        if prec == "tf32":
            return tf32_dot(X, Y)
        if prec == "tf32x3":
            return tf32x3_dot(X, Y)
        if prec == "fp32":
            return _mm32(X, Y)
        # ozaki path, only where lowp_flag set; else tf32x3 (safe) for the rest
        if lowp_flag:
            r, _ = ozaki_dot(X, Y, cfg["fmt"], cfg["kV"], cfg["kCY"], cfg["T"],
                             cfg["scaleV"], cfg["scaleCY"])
            return r
        return tf32x3_dot(X, Y)

    G = gemm(Vt, V, cfg.get("lowp_gram", False))
    nz = tau_blk != 0
    inv_tau = torch.where(nz, 1.0 / torch.where(nz, tau_blk, torch.ones_like(tau_blk)),
                          torch.full_like(tau_blk, 1e30))
    M = torch.triu(G, diagonal=1)
    M.diagonal(dim1=-2, dim2=-1).copy_(inv_tau)
    C = H[:, col:, c0:c1]
    W = gemm(Vt, C, cfg.get("lowp_W", False))
    Y = torch.linalg.solve_triangular(M.transpose(1, 2), W.to(torch.float32), upper=False).to(F64)
    VY = gemm(V, Y, cfg.get("lowp_trail", False))
    C.sub_(VY)


def blocking(n):
    if n <= 128: return (16, 16)
    if n <= 256: return (64, 64)
    if n <= 512: return (128, 64)
    return (256, 128)


def factor_custom(A, cfg):
    B, n, _ = A.shape
    # PANEL/working precision = FP32 (matches submission.py; this is the whole point)
    H = A.to(torch.float32).to(F64) if cfg.get("panel64") else A.to(torch.float32)
    H = H.clone()
    tau = torch.zeros(B, n, dtype=H.dtype, device=A.device)
    NB, ib = blocking(n)
    k = 0
    while k < n:
        nb = min(NB, n - k)
        j = 0
        while j < nb:
            col = k + j
            b = min(ib, nb - j)
            panel_factor(H[:, col:, col:col + b], tau[:, col:col + b])
            if j + b < nb:
                apply_block(H, col, b, tau, col + b, k + nb, cfg)
            j += b
        if k + nb < n:
            apply_block(H, k, nb, tau, k + nb, n, cfg)
        k += nb
    return H.to(torch.float32), tau.to(torch.float32)


def run_cfg(data, cfg):
    torch.cuda.synchronize(); t0 = time.time()
    H, tau = factor_custom(data, cfg)
    torch.cuda.synchronize(); dt = time.time() - t0
    good, msg = reference.check_implementation(data, (H, tau))
    m = re.search(r"scaled_factor_residual=([0-9.eE+-]+)", msg)
    sfr = float(m.group(1)) if m else float("nan")
    return good, sfr, dt, msg


def dots_for(kV, kCY, T):
    return sum(1 for i in range(kV) for j in range(kCY) if i + j <= T)


# ============================================================================
# PART A
# ============================================================================
def part_a():
    print("=" * 78)
    print("PART A -- accuracy with REAL FP32 PANEL (gate sfr<20; safe margin>=2x)")
    print("=" * 78)
    torch.backends.cuda.matmul.allow_tf32 = False   # TRUE fp32 accumulate everywhere
    torch.backends.cudnn.allow_tf32 = False
    data = reference.generate_input(batch=640, n=512, cond=2, seed=32530, case="mixed")
    data = data.to(DEV)

    configs = [
        ("fp32  (calib ~0.016)",            dict(prec="fp32")),
        ("tf32x3 ALL (calib ~10, safe2x)",  dict(prec="tf32x3")),
        ("1xTF32 ALL (calib ~18-21)",       dict(prec="tf32")),
    ]
    # fp8 e4m3 on trail+W+gram (deploy-realistic) at increasing dots
    for (kV, kCY, T) in [(3, 3, 2), (4, 4, 3), (4, 4, 4), (5, 5, 4)]:
        nd = dots_for(kV, kCY, T)
        configs.append((f"fp8 e4m3 trail+W+gram k{kV}/{kCY}/T{T} ={nd}dot",
                        dict(prec=None, fmt="e4m3", kV=kV, kCY=kCY, T=T,
                             scaleV="per_tensor", scaleCY="per_tensor",
                             lowp_trail=True, lowp_W=True, lowp_gram=True)))
    # fp8 e4m3 on TRAIL ONLY (gram+W stay tf32x3) -- safer, isolates the big GEMM
    for (kV, kCY, T) in [(3, 3, 2), (4, 4, 3)]:
        nd = dots_for(kV, kCY, T)
        configs.append((f"fp8 e4m3 TRAIL-ONLY k{kV}/{kCY}/T{T} ={nd}dot",
                        dict(prec=None, fmt="e4m3", kV=kV, kCY=kCY, T=T,
                             scaleV="per_tensor", scaleCY="per_tensor",
                             lowp_trail=True, lowp_W=False, lowp_gram=False)))
    # per-vec (per-row/col) scaling on the 10-dot config
    configs.append(("fp8 e4m3 trail+W+gram k4/4/T3 =10dot PER-VEC",
                    dict(prec=None, fmt="e4m3", kV=4, kCY=4, T=3,
                         scaleV="per_vec", scaleCY="per_vec",
                         lowp_trail=True, lowp_W=True, lowp_gram=True)))

    print(f"{'config':52s} {'sfr':>10s} {'margin':>8s} {'pass':>5s} {'safe':>5s} {'s':>5s}")
    for name, cfg in configs:
        try:
            good, sfr, dt, msg = run_cfg(data, cfg)
            margin = GATE / sfr if sfr > 0 else float("inf")
            pas = "Y" if (good and sfr < GATE) else "."
            safe = "Y" if (good and margin >= 2.0) else "."
            print(f"{name:52s} {sfr:10.4g} {margin:8.2f} {pas:>5s} {safe:>5s} {dt:5.0f}",
                  flush=True)
        except Exception as e:
            print(f"{name:52s}  ERROR {type(e).__name__}: {str(e)[:50]}", flush=True)


# ============================================================================
# PART B -- realized throughput on the real trailing/gram shapes
# ============================================================================
def _time(fn, it=30):
    for _ in range(5):
        try: fn()
        except Exception as e: return "ERR:" + type(e).__name__
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.empty((32, 1024, 1024), dtype=torch.int64, device=DEV).fill_(0)  # clear L2
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000.0  # us


def part_b():
    print("\n" + "=" * 78)
    print("PART B -- realized throughput on real shapes (one dot's cost)")
    print("=" * 78)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        import triton, triton.language as tl
    except Exception as e:
        print("no triton:", e); return

    # real trailing shape (fat update) and gram shape, [B, M, K, N]
    shapes = [("trail n512 fat", 640, 512, 128, 384),
              ("trail n512 thinK", 640, 480, 32, 480),
              ("gram   n512",      640, 64, 512, 64)]

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

    def x3(A, B):
        Bb, M, K = A.shape; N = B.shape[2]
        C = torch.empty((Bb, M, N), device=DEV, dtype=torch.float32)
        grid = (Bb, triton.cdiv(M, 64), triton.cdiv(N, 64))
        _x3[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(),
                  BM=64, BN=64, BK=32, num_warps=4, num_stages=3)
        return C

    has_ds = hasattr(tl, "dot_scaled")

    @triton.jit
    def _mxfp8(A, B, SA, SB, C, M, N, K, sab, sam, sak, sbb, sbk, sbn,
               ssab, ssam, ssak, ssbb, ssbk, ssbn, scb, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pb = tl.program_id(0); pm = tl.program_id(1); pn = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
        rsk = tl.arange(0, BK // 32)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ap = A + pb * sab + (rm[:, None] * sam + (rk[None, :] + k0) * sak)
            bp = B + pb * sbb + ((rk[:, None] + k0) * sbk + rn[None, :] * sbn)
            a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
            b = tl.load(bp, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
            sap = SA + pb * ssab + (rm[:, None] * ssam + (rsk[None, :] + k0 // 32) * ssak)
            sbp = SB + pb * ssbb + ((rsk[:, None] + k0 // 32) * ssbk + rn[None, :] * ssbn)
            sa = tl.load(sap, mask=rm[:, None] < M, other=127)
            sb = tl.load(sbp, mask=rn[None, :] < N, other=127)
            acc += tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3")
            # (single dot per k-block; one Ozaki slice-pair = this)
        cp = C + pb * scb + (rm[:, None] * scm + rn[None, :] * scn)
        tl.store(cp, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    def mxfp8(A8, B8, SA, SB, M, N, K):
        Bb = A8.shape[0]
        C = torch.empty((Bb, M, N), device=DEV, dtype=torch.float32)
        grid = (Bb, triton.cdiv(M, 64), triton.cdiv(N, 64))
        _mxfp8[grid](A8, B8, SA, SB, C, M, N, K, *A8.stride(), *B8.stride(),
                     *SA.stride(), *SB.stride(), *C.stride(),
                     BM=64, BN=64, BK=64, num_warps=4, num_stages=3)
        return C

    print(f"{'shape':18s} {'B,M,K,N':>20s} {'tf32x3_us':>10s} {'1xTF32_us':>10s} "
          f"{'mxfp8_1dot':>11s} {'fp8/tf32x3':>11s}")
    for name, Bb, M, K, N in shapes:
        A = torch.randn(Bb, M, K, device=DEV); B = torch.randn(Bb, K, N, device=DEV)
        t_x3 = _time(lambda: x3(A, B))
        t_1x = _time(lambda: torch.matmul(A, B))
        t_mx = "n/a"; ratio = "n/a"
        if has_ds and K % 32 == 0:
            try:
                A8 = (A.clamp(-448, 448)).to(torch.float8_e4m3fn)
                B8 = (B.clamp(-448, 448)).to(torch.float8_e4m3fn)
                SA = torch.full((Bb, M, K // 32), 127, dtype=torch.uint8, device=DEV)
                SB = torch.full((Bb, K // 32, N), 127, dtype=torch.uint8, device=DEV)
                t_mx = _time(lambda: mxfp8(A8, B8, SA, SB, M, N, K))
                if isinstance(t_mx, float) and isinstance(t_x3, float):
                    # tf32x3 = 3 tf32 dots; mxfp8 here = 1 fp8 dot. ratio = 1fp8 / (tf32x3/3)
                    ratio = f"{t_mx / (t_x3 / 3.0):.2f}"
            except Exception as e:
                t_mx = "ERR:" + type(e).__name__
        sh = f"{Bb},{M},{K},{N}"
        tx = f"{t_x3:.1f}" if isinstance(t_x3, float) else str(t_x3)
        t1 = f"{t_1x:.1f}" if isinstance(t_1x, float) else str(t_1x)
        tm = f"{t_mx:.1f}" if isinstance(t_mx, float) else str(t_mx)
        print(f"{name:18s} {sh:>20s} {tx:>10s} {t1:>10s} {tm:>11s} {str(ratio):>11s}",
              flush=True)
    print("\nINTERPRET: per-fp8-dot cost ~= mxfp8_1dot. fp8 wins iff "
          "min_safe_dots * mxfp8_1dot < tf32x3_us. (tf32x3 ~= 3 tf32 dots.)")


def main():
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
    part_a()
    try:
        part_b()
    except Exception as e:
        print("PART B failed:", type(e).__name__, str(e)[:200])


if __name__ == "__main__":
    main()
