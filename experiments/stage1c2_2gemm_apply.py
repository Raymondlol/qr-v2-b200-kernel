"""STAGE 1C (corrected) — the trailing apply is 2 GEMMs, NOT 3.

1B's ~1.4x was the explicit T-apply (3rd GEMM). Folding T into V (VT = V@T^T, precomputed once)
makes the compact-WY apply just 2 GEMMs:  W1 = V^T@C ; C -= VT@W1. The W1 global round-trip is
CHEAP (~3% of apply). So the async (overlap-able) apply should be ~1.1x the current tl.dot 2-GEMM
apply -- which flips design-A economics from "marginal" to a clear ~1.2x on n=512. Confirm here.
No banned substrings.
"""
import torch, triton
import triton.language as tl
from stage1b_compactwy import gemm as async_gemm   # K-looped async tcgen05 tf32x3 GEMM (NTERM=3)


@triton.jit
def _bmm3_k(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, SUB: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0); pm = tl.program_id(1); pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = A + pb * sab + (rm[:, None] * sam + rk[None, :] * sak)
    bp = B + pb * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(bp, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32x3"); ap += BK * sak; bp += BK * sbk
    cp = C + pb * scb + (rm[:, None] * scm + rn[None, :] * scn); cmask = (rm[:, None] < M) & (rn[None, :] < N)
    if SUB:
        acc = tl.load(cp, mask=cmask, other=0.0) - acc
    tl.store(cp, acc, mask=cmask)


def bmm3(A, B, C=None, sub=False, BM=64, BN=64, BK=32, nw=4):
    Bb, M, K = A.shape; N = B.shape[2]
    if C is None:
        C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = (Bb, triton.cdiv(M, BM), triton.cdiv(N, BN))
    _bmm3_k[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(), sub, BM=BM, BN=BN, BK=BK, num_warps=nw)
    return C


def apply_current(V, VT, C):                       # 2x tl.dot tf32x3 (current path style)
    W1 = bmm3(V.transpose(1, 2).contiguous(), C)
    Cw = C.clone(); bmm3(VT, W1, C=Cw, sub=True)
    return Cw


def apply_async(V, VT, C, nterm=3):                # 2x async tcgen05 tf32x3 (overlap-able)
    W1, _ = async_gemm(V.transpose(1, 2).contiguous(), C, nterm=nterm)
    Cw = C.clone(); async_gemm(VT, W1, C=Cw, sub=True, nterm=nterm)
    return Cw


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=40):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:" + repr(e)[:160]
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    torch.manual_seed(0)
    print("\n=== correctness: 2-GEMM apply (async vs torch) ===")
    for Bb, m, b, Wd in [(8, 512, 128, 256), (8, 512, 128, 384)]:
        A = torch.randn(Bb, m, b, device="cuda")
        V = torch.tril(A, -1); V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        G = V.transpose(1, 2) @ V; T = torch.linalg.inv(torch.triu(G))
        VT = (V @ T.transpose(1, 2)).contiguous()
        C = torch.randn(Bb, m, Wd, device="cuda")
        ref = C - VT @ (V.transpose(1, 2) @ C)
        out = apply_async(V, VT, C.clone()); torch.cuda.synchronize()
        err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        print(f"  m{m} b{b} Wd{Wd}: async 2-GEMM relerr={err:.2e}  {'OK' if err<1e-4 else 'CHECK'}")

    print("\n=== speed: 2-GEMM apply  async vs current (B=640) ===")
    for Wd in [256, 384]:
        Bb, m, b = 640, 512, 128
        A = torch.randn(Bb, m, b, device="cuda")
        V = torch.tril(A, -1); V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        G = V.transpose(1, 2) @ V; T = torch.linalg.inv(torch.triu(G))
        VT = (V @ T.transpose(1, 2)).contiguous()
        C = torch.randn(Bb, m, Wd, device="cuda")
        t_cur = time_fn(lambda: apply_current(V, VT, C))
        t_asy = time_fn(lambda: apply_async(V, VT, C))
        if isinstance(t_cur, str) or isinstance(t_asy, str):
            print(f"  Wd{Wd}: cur={t_cur} asy={t_asy}"); continue
        print(f"  Wd{Wd}: current 2-GEMM {t_cur:7.1f} us | async 2-GEMM {t_asy:7.1f} us | {t_asy/t_cur:.2f}x")
    print("\nSTAGE 1C2 DONE (need: async 2-GEMM ~<=1.15x current -> trailing worker efficient, economics OK)")


if __name__ == "__main__":
    main()
