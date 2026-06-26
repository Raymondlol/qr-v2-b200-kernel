"""Gluon G2 (decisive speed): batched tiled tcgen05 GEMM (1-pass tf32) on the real
trailing shapes, vs our tl.dot tf32x3 (38/41 TF/s) and cuBLAS-1xTF32 (235/337). Does
the deployable Gluon tcgen05 path beat our plain-Triton 10% ceiling? (3-pass tf32x3
would be ~1/3 the tf32 throughput.) No banned substrings.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import tl_dot, default_blocked_layout

torch.backends.cuda.matmul.allow_tf32 = True


@gluon.jit
def gemm(A, B, C, Mtiles, Ntiles, M, N, K,
         sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
         BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    pid_b = gl.program_id(0)
    pid_m = gl.program_id(1)
    pid_n = gl.program_id(2)
    a_blk: gl.constexpr = default_blocked_layout([BM, BK], gl.num_warps())
    b_blk: gl.constexpr = default_blocked_layout([BK, BN], gl.num_warps())
    c_blk: gl.constexpr = default_blocked_layout([BM, BN], gl.num_warps())
    rm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
    rk_a = gl.arange(0, BK, layout=gl.SliceLayout(0, a_blk))[None, :]
    rk_b = gl.arange(0, BK, layout=gl.SliceLayout(1, b_blk))[:, None]
    rn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_blk))[None, :]
    acc = gl.zeros([BM, BN], gl.float32, layout=c_blk)
    for k0 in range(0, K, BK):
        a = gl.load(A + pid_b * sab + rm * sam + (k0 + rk_a) * sak)
        b = gl.load(B + pid_b * sbb + (k0 + rk_b) * sbk + rn * sbn)
        acc = tl_dot(a, b, acc=acc, input_precision="tf32")
    cm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, c_blk))[:, None]
    cn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, c_blk))[None, :]
    gl.store(C + pid_b * scb + cm * scm + cn * scn, acc)


def gluon_gemm(A, B, BM=128, BN=128, BK=128, nw=8):
    Bb, M, K = A.shape
    N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = (Bb, triton.cdiv(M, BM), triton.cdiv(N, BN))
    gemm[grid](A, B, C, grid[1], grid[2], M, N, K, *A.stride(), *B.stride(), *C.stride(),
               BM=BM, BN=BN, BK=BK, num_warps=nw)
    return C


def clear_l2(): torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)
def time_fn(fn, it=40):
    for _ in range(5):
        try: fn()
        except Exception as e: return ("ERR:" + repr(e)[:120])
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


PEAK = 1100.0
def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    for name, Bb, M, K, N in [("n512 fat 640,512,128,384", 640, 512, 128, 384),
                              ("n1024 fat 60,1024,256,768", 60, 1024, 256, 768)]:
        A = torch.randn(Bb, M, K, device="cuda"); B = torch.randn(Bb, K, N, device="cuda")
        flops = 2.0 * Bb * M * N * K
        ref = torch.matmul(A.double(), B.double()).float()
        out = gluon_gemm(A, B)
        if isinstance(out, str):
            print(f"\n{name}\n  gluon tcgen05: {out}"); continue
        rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        t = time_fn(lambda: gluon_gemm(A, B))
        tcu = time_fn(lambda: torch.matmul(A, B))
        print(f"\n{name}")
        if isinstance(t, str):
            print(f"  gluon tcgen05 1xtf32: {t}")
        else:
            tf = flops / (t * 1e-6) / 1e12
            print(f"  gluon tcgen05 1xtf32: {t:8.1f} us  {tf:6.0f} TF/s  {100*tf/PEAK:4.0f}% peak  relerr={rel:.1e}")
            print(f"    -> implied 3-pass tf32x3 ~ {tf/3:.0f} TF/s (vs our tl.dot tf32x3 38/41)")
        tfcu = flops / (tcu * 1e-6) / 1e12
        print(f"  cuBLAS 1xtf32       : {tcu:8.1f} us  {tfcu:6.0f} TF/s  {100*tfcu/PEAK:4.0f}% peak")
    print("\nGATE2 DONE")


if __name__ == "__main__":
    main()
