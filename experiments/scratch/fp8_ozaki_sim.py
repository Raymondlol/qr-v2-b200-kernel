"""DECISIVE hardware-free fp8/fp4 Ozaki-slice simulation for the mixed@640 gate.

Implements the SAME blocked compact-WY Householder QR as submission.py's
_factor_custom, but routes the trailing GEMM (C -= V@Y) and optionally the
Gram (V^T V) and W (=V^T C) through a SIMULATED low-precision Ozaki product:
each fp32 operand is split into k fp8(e4m3)/fp4(e2m1) slices, cross-products
with i+j<=T are accumulated in fp32; everything else stays fp32.

Then runs the EXACT competition factor-residual metric (reference.check_implementation)
on the real mixed n=512 batch=640 input (worst-of-640).

Run: python experiments/scratch/fp8_ozaki_sim.py
"""
import sys, os, time, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "harness"))
import reference

torch.set_grad_enabled(False)
DEV = "cpu"
F64 = torch.float64


# ============================================================================
# Low-precision element emulation (exact IEEE-style round-to-nearest-even)
# ============================================================================
def _round_mantissa(x, mbits, emin, emax, max_normal):
    """Round fp32 tensor x to a float format with `mbits` mantissa bits and
    exponent range [emin, emax] (subnormals supported down to emin). Returns
    fp32 tensor holding the rounded values. Round-to-nearest-even."""
    x = x.to(torch.float64)
    sign = torch.sign(x)
    ax = x.abs()
    out = torch.zeros_like(ax)
    nz = ax > 0
    # exponent of each element
    e = torch.floor(torch.log2(ax.clamp_min(1e-300)))
    # clamp exponent to normal range; subnormals use emin
    e_eff = torch.clamp(e, min=emin)
    # scale so mantissa is integer with mbits fractional bits
    scale = torch.pow(torch.tensor(2.0, dtype=F64), (mbits - e_eff))
    q = torch.round(ax * scale) / scale
    # overflow clamp to max representable
    q = torch.clamp(q, max=max_normal)
    out = torch.where(nz, q, out)
    return (sign * out).to(torch.float32)


# Format params: e4m3 (fp8): 3 mantissa bits, exp bias 7 -> normal exp [-6,8],
#   max normal 448. e2m1 (fp4): 1 mantissa bit, exp range [-1? ] -> standard
#   nvfp4 element e2m1: max 6.0, min normal 1.0, subnormal 0.5.
# e5m2 (fp8 alt): 2 mantissa bits, exp [-14,15], max 57344.
FORMATS = {
    "e4m3": dict(mbits=3, emin=-6, emax=8, max_normal=448.0),
    "e5m2": dict(mbits=2, emin=-14, emax=15, max_normal=57344.0),
    "e2m1": dict(mbits=1, emin=0, emax=2, max_normal=6.0),  # fp4, normals 1..6, subnormal 0.5
    "e3m2": dict(mbits=2, emin=-2, emax=4, max_normal=28.0),  # fp6 e3m2 (for reference)
}


def to_lowp(x, fmt):
    p = FORMATS[fmt]
    return _round_mantissa(x, p["mbits"], p["emin"], p["emax"], p["max_normal"])


# ============================================================================
# Ozaki splitting: split fp32 A into k low-precision slices.
# A ~= sum_i A_i * 2^{s_i}, each A_i a lowp-rounded residual.
# We do per-tile (per-tensor) or per-row scaling so the leading slice uses the
# full dynamic range of the format. Slices: residual = A - reconstructed.
# ============================================================================
def ozaki_split(A, fmt, k, role, scale_mode="per_tensor"):
    """Return list of (slice_lowp_fp32, scale) so A ~= sum slice_i * scale_i.
    `role`: 'left' (A is M×K, contraction dim = -1) or 'right' (A is K×N,
    contraction dim = -2). scale_mode:
      'per_tensor' -> one scalar scale for the whole matrix.
      'per_vec'    -> scale per output-broadcastable vector: for 'left',
                      per-row (max over K, shape M×1); for 'right', per-col
                      (max over K, shape 1×N). This is the per-row/per-col
                      mxfp-style block scale that survives the K-reduction
                      cleanly (scale_left(M×1) * scale_right(1×N) = output M×N).
    """
    p = FORMATS[fmt]
    max_normal = p["max_normal"]
    A = A.to(torch.float64)
    slices = []
    R = A.clone()
    for i in range(k):
        amax = R.abs()
        if scale_mode == "per_tensor":
            m = amax.amax(dim=(-2, -1), keepdim=True)
        elif scale_mode == "per_vec":
            cdim = -1 if role == "left" else -2
            m = amax.amax(dim=cdim, keepdim=True)
        else:
            raise ValueError(scale_mode)
        m = m.clamp_min(1e-300)
        scale = m / max_normal
        Rs = R / scale
        q = to_lowp(Rs.to(torch.float32), fmt).to(torch.float64)
        slices.append((q, scale))
        R = R - q * scale
    return slices


def _to_tf32(x):
    """Round fp32 tensor to TF32: keep 10 explicit mantissa bits, full fp32
    exponent range. Round-to-nearest via bit truncation on the float32 mantissa
    (TF32 truncates; we add the round bit for nearest)."""
    xi = x.to(torch.float32).view(torch.int32)
    # fp32 mantissa = 23 bits; TF32 keeps top 10 -> drop low 13 bits.
    # round-to-nearest: add half-ulp then mask.
    round_bit = (1 << 12)
    mask = ~((1 << 13) - 1)
    xi = (xi + round_bit) & mask
    return xi.view(torch.float32)


def tf32_dot(A, B):
    """1xTF32: round both operands to TF32, single fp32-accumulated dot."""
    return (_to_tf32(A.to(torch.float32)) @ _to_tf32(B.to(torch.float32))).to(torch.float64)


def _split_tf32_hilo(x):
    hi = _to_tf32(x.to(torch.float32))
    lo = _to_tf32((x.to(torch.float32) - hi))
    return hi, lo


def tf32x3_dot(A, B):
    """tf32x3: 3-term hi/lo split (Ah*Bh + Ah*Bl + Al*Bh), ~22 effective
    mantissa bits. Matches submission _mm3 / _bmm_x3 input_precision=tf32x3."""
    Ah, Al = _split_tf32_hilo(A.to(torch.float32))
    Bh, Bl = _split_tf32_hilo(B.to(torch.float32))
    out = (Ah @ Bh).to(torch.float64)
    out = out + (Ah @ Bl).to(torch.float64)
    out = out + (Al @ Bh).to(torch.float64)
    return out


def ozaki_dot(A, B, fmt, kA, kB, T, scaleA="per_tensor", scaleB="per_tensor",
              count=None):
    """Compute A @ B via Ozaki low-precision slices.
    A: (...,M,K), B: (...,K,N). Split A into kA slices, B into kB slices.
    Accumulate cross products i (from A) + j (from B) with i+j <= T, in fp32.
    Returns fp64 result (the accumulation itself simulates fp32 tensor-core
    accumulate; we use fp64 to avoid double-rounding the *accumulator*, which
    on real hardware is fp32 -- but for a 512-K dot fp32 accum error is tiny
    vs the mantissa-truncation we are studying; verified separately).
    `count` (a mutable list [n]) tallies the number of low-precision dots used.
    """
    sa = ozaki_split(A, fmt, kA, "left", scaleA)
    sb = ozaki_split(B, fmt, kB, "right", scaleB)
    out = None
    ndots = 0
    for i, (Ai, scA) in enumerate(sa):
        for j, (Bj, scB) in enumerate(sb):
            if i + j > T:
                continue
            # one low-precision matmul (tensor-core dot), fp32-accumulated
            prod = (Ai.to(torch.float32) @ Bj.to(torch.float32)).to(torch.float64)
            prod = prod * (scA * scB)
            out = prod if out is None else out + prod
            ndots += 1
    if count is not None:
        count[0] += ndots
    return out, ndots


# ============================================================================
# Blocked compact-WY Householder QR (mirrors submission _factor_custom).
# trailing GEMM / Gram / W routed through ozaki_dot when cfg requests it.
# Panel factorization stays FP32 always.
# ============================================================================
def panel_factor(P, tau_out):
    # P: (B,m,b) fp64 working copy; same math as submission _panel_factor.
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


def apply_block(H, col, b, tau, c0, c1, cfg, count):
    if c1 <= c0:
        return
    P = H[:, col:, col:col + b]
    V = torch.tril(P[:, :, :b], diagonal=-1).clone()
    V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    tau_blk = tau[:, col:col + b]
    Vt = V.transpose(1, 2)
    prec = cfg.get("prec")  # 'tf32', 'tf32x3', or None (=fp32 / ozaki)

    def gemm(X, Y, kX, kY, sX, sY, lowp_flag):
        if prec == "tf32":
            return tf32_dot(X, Y)
        if prec == "tf32x3":
            return tf32x3_dot(X, Y)
        if lowp_flag:
            r, _ = ozaki_dot(X, Y, cfg["fmt"], kX, kY, cfg["T"], sX, sY, count)
            return r
        return (X @ Y).to(torch.float64)

    # Gram G = V^T V
    G = gemm(Vt, V, cfg.get("kV"), cfg.get("kV"), cfg.get("scaleV"),
             cfg.get("scaleV"), cfg.get("lowp_gram"))
    nz = tau_blk != 0
    inv_tau = torch.where(nz, 1.0 / torch.where(nz, tau_blk, torch.ones_like(tau_blk)),
                          torch.full_like(tau_blk, 1e30))
    M = torch.triu(G, diagonal=1)
    M.diagonal(dim1=-2, dim2=-1).copy_(inv_tau)
    C = H[:, col:, c0:c1]
    # W = V^T C
    W = gemm(Vt, C, cfg.get("kV"), cfg.get("kC"), cfg.get("scaleV"),
             cfg.get("scaleC"), cfg.get("lowp_W"))
    # Y = solve_triangular(M^T, W)  -- always fp32-class
    Y = torch.linalg.solve_triangular(M.transpose(1, 2), W, upper=False)
    # C -= V @ Y  (THE trailing update)
    VY = gemm(V, Y, cfg.get("kV"), cfg.get("kY"), cfg.get("scaleV"),
              cfg.get("scaleY"), cfg.get("lowp_trail"))
    C.sub_(VY)


def blocking(n):
    if n <= 128: return (16, 16)
    if n <= 256: return (64, 64)
    if n <= 512: return (128, 64)
    if n <= 1024: return (256, 128)
    if n <= 2048: return (256, 128)
    return (256, 8)


def factor_custom(A, cfg, count):
    B, n, _ = A.shape
    H = A.to(F64).clone()
    tau = torch.zeros(B, n, dtype=F64, device=A.device)
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
                apply_block(H, col, b, tau, col + b, k + nb, cfg, count)
            j += b
        if k + nb < n:
            apply_block(H, k, nb, tau, k + nb, n, cfg, count)
        k += nb
    return H.to(torch.float32), tau.to(torch.float32)


def run(data, cfg):
    count = [0]
    t0 = time.time()
    H, tau = factor_custom(data, cfg, count)
    dt = time.time() - t0
    good, msg = reference.check_implementation(data, (H, tau))
    # extract scaled_factor_residual
    import re
    m = re.search(r"scaled_factor_residual=([0-9.eE+-]+)", msg)
    sfr = float(m.group(1)) if m else float("nan")
    return good, sfr, count[0], dt, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=640)
    ap.add_argument("--seed", type=int, default=32530)
    # single-config mode (for parallel sweeping)
    ap.add_argument("--mode", default="calib",
                    help="calib | tf32x3 | tf32 | ozaki")
    ap.add_argument("--fmt", default="e4m3")
    ap.add_argument("--kV", type=int, default=3)
    ap.add_argument("--kCY", type=int, default=3)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--scaleV", default="per_tensor")
    ap.add_argument("--scaleCY", default="per_tensor")
    # which GEMMs get low precision (default all three, matching submission n<=512)
    ap.add_argument("--trail", type=int, default=1)
    ap.add_argument("--W", type=int, default=1)
    ap.add_argument("--gram", type=int, default=1)
    args = ap.parse_args()

    data = reference.generate_input(batch=args.batch, n=args.n, cond=2,
                                    seed=args.seed, case="mixed")
    GATE = 20.0

    if args.mode == "calib":
        cfg = dict(prec=None)
        tag = "fp32"
    elif args.mode in ("tf32x3", "tf32"):
        cfg = dict(prec=args.mode)
        tag = args.mode
    else:  # ozaki
        cfg = dict(prec=None, fmt=args.fmt, kV=args.kV, kC=args.kCY, kY=args.kCY,
                   scaleV=args.scaleV, scaleC=args.scaleCY, scaleY=args.scaleCY,
                   lowp_trail=bool(args.trail), lowp_W=bool(args.W),
                   lowp_gram=bool(args.gram), T=args.T)
        tag = f"{args.fmt} kV={args.kV} kCY={args.kCY} T={args.T} sV={args.scaleV} sCY={args.scaleCY}"

    good, sfr, nd, dt, msg = run(data, cfg)
    ntrail = (sum(1 for i in range(args.kV) for j in range(args.kCY) if i + j <= args.T)
              if args.mode == "ozaki" else 0)
    status = "PASS" if (good and sfr < GATE) else "FAIL"
    print(f"RESULT b={args.batch} | {tag} | {status} | sfr={sfr:.4g} | "
          f"trail_dots={ntrail} | {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
