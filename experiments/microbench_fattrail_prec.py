"""DECISIVE (real pipeline): can the FAT trailing update (C -= V@Y, the biggest
GEMM) drop BELOW tf32x3 without breaking the mixed@640 gate? The reconciliation
proved the ~1.9x margin is PANEL-dominated, not fat-trailing-dominated -- so the
fat trailing should tolerate low precision. Test it in the REAL submission
pipeline (real Triton panel + real tf32x3 narrow/gram), swapping ONLY the fat
trailing precision. The narrow within-panel updates STAY tf32x3 (they feed the
reflectors). Accuracy via emulation (fp32-accumulated); measures real worst-of-640
margin. If 1xTF32 or fp8-Ozaki holds >=2x here, there's a real sub-tf32 win.

Run: modal run modal_microbench.py --script microbench_fattrail_prec.py
"""
import sys, re
import torch
sys.path.insert(0, "/work"); sys.path.insert(0, "/work/harness")
import reference
import _real_submission as sub

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False   # emulated dots accumulate in TRUE fp32
F64 = torch.float64

# ----- low-precision element emulation (round-to-nearest), batched, GPU-safe -----
FMT = {"e4m3": (3, -6, 448.0), "e5m2": (2, -14, 57344.0), "e2m1": (1, 0, 6.0)}
def _round_mantissa(x, mbits, emin, max_normal):
    x = x.to(F64); s = torch.sign(x); ax = x.abs(); nz = ax > 0
    e = torch.floor(torch.log2(ax.clamp_min(1e-300)))
    e_eff = torch.clamp(e, min=emin)
    sc = torch.exp2(mbits - e_eff)
    q = torch.clamp(torch.round(ax * sc) / sc, max=max_normal)
    return (s * torch.where(nz, q, torch.zeros_like(ax))).to(torch.float32)

def _ozaki_split(A, fmt, k, role):
    mbits, emin, mx = FMT[fmt]; A = A.to(F64); R = A.clone(); sl = []
    for _ in range(k):
        m = R.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-300)
        scale = m / mx
        q = _round_mantissa((R / scale).to(torch.float32), mbits, emin, mx).to(F64)
        sl.append((q, scale)); R = R - q * scale
    return sl

def _mm32(A, B):
    return (A.to(torch.float32) @ B.to(torch.float32)).to(F64)

def _to_tf32(x):
    xi = x.to(torch.float32).view(torch.int32)
    xi = (xi + (1 << 12)) & (~((1 << 13) - 1))
    return xi.view(torch.float32)

def emulate(A, B, mode):
    """A@B at the requested precision, returned fp32."""
    if mode == "1xtf32":
        return _mm32(_to_tf32(A), _to_tf32(B)).to(torch.float32)
    if mode == "tf32x3":
        Ah = _to_tf32(A); Al = _to_tf32(A.to(torch.float32) - Ah)
        Bh = _to_tf32(B); Bl = _to_tf32(B.to(torch.float32) - Bh)
        return (_mm32(Ah, Bh) + _mm32(Ah, Bl) + _mm32(Al, Bh)).to(torch.float32)
    # fp8 ozaki: mode like "fp8_e4m3_k3_T2"
    _, fmt, kk, TT = mode.split("_")
    k = int(kk[1:]); T = int(TT[1:])
    sa = _ozaki_split(A, fmt, k, "left"); sb = _ozaki_split(B, fmt, k, "right")
    out = None
    for i, (Ai, sca) in enumerate(sa):
        for j, (Bj, scb) in enumerate(sb):
            if i + j > T: continue
            p = _mm32(Ai, Bj) * (sca * scb)
            out = p if out is None else out + p
    return out.to(torch.float32)

def dots(k, T):
    return sum(1 for i in range(k) for j in range(k) if i + j <= T)

# ----- patch: route ONLY the fat trailing C -= V@Y through `emulate` -----
sub._IN_FAT = False
sub._FAT_MODE = None
_orig_mm_sub = sub._mm_sub
def _patched_mm_sub(A, B, C):
    if sub._IN_FAT and sub._FAT_MODE is not None:
        C.sub_(emulate(A, B, sub._FAT_MODE).to(C.dtype))
    else:
        _orig_mm_sub(A, B, C)
sub._mm_sub = _patched_mm_sub

# replace _factor_custom with a copy that flags the FAT trailing call
def _factor_custom(A):
    B, n, _ = A.shape
    H = A.clone()
    tau = torch.zeros(B, n, dtype=A.dtype, device=A.device)
    NB, ib_max = sub._blocking(n)
    k = 0
    while k < n:
        nb = min(NB, n - k)
        j = 0
        while j < nb:
            col = k + j
            m = n - col
            b = min(sub._max_ib(m, ib_max), nb - j)
            if A.is_cuda and sub._HAS_TRITON:
                import triton
                BN = triton.next_power_of_2(m); BCOLS = triton.next_power_of_2(b)
                nw = 4 if BN <= 128 else 8
                sub._panel_kernel[(B,)](H, tau, n, col, m, b, BN=BN, BCOLS=BCOLS, num_warps=nw)
            else:
                sub._panel_factor(H[:, col:, col:col + b], tau[:, col:col + b])
            if j + b < nb:
                sub._apply_block(H, col, b, tau, col + b, k + nb)   # narrow: stays tf32x3
            j += b
        if k + nb < n:
            sub._IN_FAT = True
            sub._apply_block(H, k, nb, tau, k + nb, n)              # FAT: routed by _FAT_MODE
            sub._IN_FAT = False
        k += nb
    return H, tau
sub._factor_custom = _factor_custom

def sfr_of(data, out):
    good, msg = reference.check_implementation(data, out)
    m = re.search(r"scaled_factor_residual=([0-9.eE+-]+)", msg)
    return good, (float(m.group(1)) if m else float("nan"))

def run_real(b, n, seed, fat_mode):
    sub._FAT_MODE = fat_mode
    data = reference.generate_input(batch=b, n=n, cond=2, seed=seed, case="mixed").cuda()
    sub._BIG_X3 = (n <= 512)
    H, tau = sub._factor_custom(data.clone())
    return sfr_of(data, (H, tau))

def main():
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
    print("=" * 92)
    print("REAL pipeline, FAT trailing precision swept (narrow updates + panel UNCHANGED).")
    print("margin = 20/sfr (worst-of-batch). SAFE = margin>=2.0. baseline=tf32x3 fused (current).")
    print("=" * 92)
    modes = [
        ("baseline tf32x3 (current)", None),
        ("fat=1xTF32",                "1xtf32"),
        ("fat=tf32x3 (emul check)",   "tf32x3"),
        (f"fat=fp8 e4m3 k3 T2 ={dots(3,2)}dot", "fp8_e4m3_k3_T2"),
        (f"fat=fp8 e4m3 k4 T3 ={dots(4,3)}dot", "fp8_e4m3_k4_T3"),
        (f"fat=fp8 e4m3 k2 T1 ={dots(2,1)}dot", "fp8_e4m3_k2_T1"),
    ]
    seeds = [32530, 1, 2024]
    for n, b, label in [(512, 640, "n512 mixed b640"), (1024, 60, "n1024 mixed b60")]:
        print(f"\n--- {label} ---")
        print(f"{'fat-trailing mode':34s} " + " ".join(f"sd{s:>6d}" for s in seeds) + "   worstmargin SAFE")
        for name, mode in modes:
            sfrs = []
            for s in seeds:
                try:
                    good, sfr = run_real(b, n, s, mode)
                    sfrs.append(sfr if good else float("nan"))
                except Exception as e:
                    sfrs.append(float("nan"))
                    name2 = name + f" ERR:{type(e).__name__}"
            worst = max([x for x in sfrs if x == x], default=float("nan"))
            mar = 20.0 / worst if worst == worst and worst > 0 else float("nan")
            safe = "Y" if (mar == mar and mar >= 2.0) else "."
            cells = " ".join(f"{x:8.3g}" for x in sfrs)
            print(f"{name:34s} {cells}   {mar:8.2f}  {safe}", flush=True)

if __name__ == "__main__":
    main()
