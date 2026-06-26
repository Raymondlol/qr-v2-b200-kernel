"""Gluon G1: minimal tcgen05 GEMM via the in-tree tl_dot helper (deployable, no nvcc).
Validates the 3.6.0 Gluon tcgen05 path compiles + runs + is correct on B200 sm_100.
Single tile, one matrix: C[M,N] = A[M,K] @ B[K,N], tf32 precision (~1e-3 vs fp64).
No banned substrings.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import tl_dot, default_blocked_layout


@gluon.jit
def gemm1(A, B, C, sam, sak, sbk, sbn, scm, scn,
          M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    a_blk: gl.constexpr = default_blocked_layout([M, K], gl.num_warps())
    b_blk: gl.constexpr = default_blocked_layout([K, N], gl.num_warps())
    c_blk: gl.constexpr = default_blocked_layout([M, N], gl.num_warps())
    am = gl.arange(0, M, layout=gl.SliceLayout(1, a_blk))[:, None]
    ak = gl.arange(0, K, layout=gl.SliceLayout(0, a_blk))[None, :]
    a = gl.load(A + am * sam + ak * sak)
    bk = gl.arange(0, K, layout=gl.SliceLayout(1, b_blk))[:, None]
    bn = gl.arange(0, N, layout=gl.SliceLayout(0, b_blk))[None, :]
    b = gl.load(B + bk * sbk + bn * sbn)
    acc = tl_dot(a, b, input_precision="tf32")            # tcgen05 mma
    acc = gl.convert_layout(acc, c_blk)
    cm = gl.arange(0, M, layout=gl.SliceLayout(1, c_blk))[:, None]
    cn = gl.arange(0, N, layout=gl.SliceLayout(0, c_blk))[None, :]
    gl.store(C + cm * scm + cn * scn, acc)


def run(M, N, K, nw):
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.empty(M, N, device="cuda")
    gemm1[(1,)](A, B, C, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                C.stride(0), C.stride(1), M=M, N=N, K=K, num_warps=nw)
    ref = (A.double() @ B.double()).float()
    rel = (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    print(f"  M={M} N={N} K={K} nw={nw}: relerr={rel:.2e} -> {'OK (tf32)' if rel < 5e-3 else 'WRONG'}")


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    print("=== Gluon tcgen05 GEMM correctness ===")
    run(128, 128, 64, 4)
    run(128, 128, 64, 8)
    run(128, 256, 64, 4)
    print("GATE1 DONE")


if __name__ == "__main__":
    main()
