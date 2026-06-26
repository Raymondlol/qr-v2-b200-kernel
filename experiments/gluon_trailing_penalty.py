"""PHASE 1.1 — pin the design-A trailing penalty with the REAL primitive (tcgen05, not tl.dot).
At the real n=512 trailing shape (b=640, M=512, K=128(=NB), N=384), compare:
  (1) batched tl.dot tf32x3   = the CURRENT submission trailing path (baseline)
  (2) batched tcgen05         = multi-CTA tcgen05 (grid tiles all matrices x output tiles)
  (3) single-CTA/matrix tcgen05 = design-A trailing (one CTA loops its matrix's output tiles)
Ratio (3)/(1) = the real design-A trailing penalty that the panel-overlap must beat.
Also n=1024 for reference. No banned substrings.
"""
import torch, triton, triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import tl_dot, default_blocked_layout

torch.backends.cuda.matmul.allow_tf32 = True


# ---------- (1) batched tl.dot tf32x3 (current path) ----------
@triton.autotune(configs=[
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
], key=['M', 'N', 'K'])
@triton.jit
def _bt(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
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


def batched_tldot(A, B):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = lambda m: (Bb, triton.cdiv(M, m['BM']), triton.cdiv(N, m['BN']))
    _bt[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride())
    return C


# ---------- (2) batched tcgen05 (multi-CTA, grid tiles) ----------
@gluon.jit
def _bg(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
        BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, NPASS: gl.constexpr):
    pid_b = gl.program_id(0); pid_m = gl.program_id(1); pid_n = gl.program_id(2)
    a_blk: gl.constexpr = default_blocked_layout([BM, BK], gl.num_warps())
    b_blk: gl.constexpr = default_blocked_layout([BK, BN], gl.num_warps())
    c_blk: gl.constexpr = default_blocked_layout([BM, BN], gl.num_warps())
    acc = gl.zeros([BM, BN], gl.float32, layout=c_blk)
    for k0 in range(0, K, BK):
        rm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
        rka = gl.arange(0, BK, layout=gl.SliceLayout(0, a_blk))[None, :]
        rkb = gl.arange(0, BK, layout=gl.SliceLayout(1, b_blk))[:, None]
        rn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_blk))[None, :]
        a = gl.load(A + pid_b * sab + rm * sam + (k0 + rka) * sak)
        b = gl.load(B + pid_b * sbb + (k0 + rkb) * sbk + rn * sbn)
        for _p in range(NPASS):  # NPASS=3 => tf32x3-equivalent TIMING (3 tcgen05 passes)
            acc = tl_dot(a, b, acc=acc, input_precision="tf32")
    cm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, c_blk))[:, None]
    cn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, c_blk))[None, :]
    gl.store(C + pid_b * scb + cm * scm + cn * scn, acc)


def batched_tcgen05(A, B, BM=128, BN=128, BK=128, nw=8, npass=1):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = (Bb, triton.cdiv(M, BM), triton.cdiv(N, BN))
    _bg[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(),
              BM=BM, BN=BN, BK=BK, NPASS=npass, num_warps=nw)
    return C


# ---------- (3) single-CTA/matrix tcgen05 (design-A trailing) ----------
@gluon.jit
def _sg(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
        BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
        MT: gl.constexpr, NT: gl.constexpr, NPASS: gl.constexpr):
    pid_b = gl.program_id(0)
    a_blk: gl.constexpr = default_blocked_layout([BM, BK], gl.num_warps())
    b_blk: gl.constexpr = default_blocked_layout([BK, BN], gl.num_warps())
    c_blk: gl.constexpr = default_blocked_layout([BM, BN], gl.num_warps())
    for mt in range(MT):
        for nt in range(NT):
            acc = gl.zeros([BM, BN], gl.float32, layout=c_blk)
            for k0 in range(0, K, BK):
                rm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
                rka = gl.arange(0, BK, layout=gl.SliceLayout(0, a_blk))[None, :]
                rkb = gl.arange(0, BK, layout=gl.SliceLayout(1, b_blk))[:, None]
                rn = nt * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_blk))[None, :]
                a = gl.load(A + pid_b * sab + rm * sam + (k0 + rka) * sak)
                b = gl.load(B + pid_b * sbb + (k0 + rkb) * sbk + rn * sbn)
                for _p in range(NPASS):
                    acc = tl_dot(a, b, acc=acc, input_precision="tf32")
            cm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, c_blk))[:, None]
            cn = nt * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, c_blk))[None, :]
            gl.store(C + pid_b * scb + cm * scm + cn * scn, acc)


def single_tcgen05(A, B, BM=128, BN=128, BK=128, nw=8, npass=1):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    _sg[(Bb,)](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(),
               BM=BM, BN=BN, BK=BK, MT=triton.cdiv(M, BM), NT=triton.cdiv(N, BN),
               NPASS=npass, num_warps=nw)
    return C


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, A, B, it=60):
    for _ in range(6):
        try: fn(A, B)
        except Exception as e: return "ERR:" + repr(e)[:160]
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(A, B); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1000


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    shapes = [
        ("n=512 trailing  (b640,M512,K128,N384)", 640, 512, 128, 384),
        ("n=1024 trailing (b60,M1024,K256,N768)", 60, 1024, 256, 768),
    ]
    # tf32x3-equivalent timing: tcgen05 variants run 3 passes (3 tcgen05 mmas accumulated).
    print(f"\n{'shape':40s} {'(1)tldotX3':>11}{'(2)batTC3':>10}{'(3)1ctaTC3':>11} | {'3/1':>6}{'3/2':>6}")
    print("  (1)=batched tl.dot tf32x3 = CURRENT path;  (2),(3)=tcgen05 x3 passes (tf32x3-equiv timing)")
    for name, Bb, M, K, N in shapes:
        A = torch.randn(Bb, M, K, device="cuda"); B = torch.randn(Bb, K, N, device="cuda")
        t1 = time_fn(batched_tldot, A, B)
        t2 = time_fn(lambda a, b: batched_tcgen05(a, b, npass=3), A, B)
        t3 = time_fn(lambda a, b: single_tcgen05(a, b, npass=3), A, B)
        # also 1-pass single for reference (what design-A would cost if 1xTF32 trailing were legal)
        t3_1 = time_fn(lambda a, b: single_tcgen05(a, b, npass=1), A, B)
        if any(isinstance(t, str) for t in (t1, t2, t3)):
            print(f"{name:40s} t1={t1} t2={t2} t3={t3}"); continue
        print(f"{name:40s} {t1:11.1f}{t2:10.1f}{t3:11.1f} | {t3/t1:6.2f}{t3/t2:6.2f}   (1pass 1cta={t3_1:.0f})")
    print("\nPHASE1.1 DONE (3/1 = REAL design-A trailing penalty at tf32x3; A viable if panel(41%) hidden")
    print("  behind trailing(26%*penalty) nets >1.0: e.g. penalty<=1.58 => max(41,26*pen)<=67 always wins)")


if __name__ == "__main__":
    main()
