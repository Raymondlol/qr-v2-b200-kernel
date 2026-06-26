"""STAGE 1 increment B — compact-WY trailing APPLY via async tcgen05 (the trailing partition's
real job), built from a K-looped tf32xN async GEMM.

  C := C - V @ (T^T @ (V^T @ C))     (block Householder apply, T = compact-WY T-factor)
     W1 = V^T @ C    (K = m, large -> K-loop, wait-per-chunk; worker-private bar = no coupling)
     W2 = T^T @ W1   (K = b)
     C -= V @ W2     (K = b, subtract epilogue)

Generalize the Stage-1A GEMM: K-loop (BK chunks, wait per chunk to reuse smem), NTERM in {3,4}
(4 adds Al@Bl to lift naive 3-term tf32x3 ~19-20bit toward ~22bit), optional SUB epilogue.
Validate: (1) apply relerr vs torch compact-WY; (2) NTERM=4 precision vs NTERM=3; (3) time vs
the current _bmm3+solve+_bmm3_sub path. No banned substrings.
"""
import torch, triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory, tcgen05_mma, TensorMemoryLayout, mbarrier,
    fence_async_shared, get_tmem_reg_layout,
)
from triton.tools.triton_to_gluon_translater.translator_helpers import (
    get_shared_memory_mma_operand, default_blocked_layout,
)


@gluon.jit
def _hi(x):
    return ((x.to(gl.int32, bitcast=True)) & -8192).to(gl.float32, bitcast=True)


@gluon.jit
def _tf32xN_gemm(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                 M, N, K, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                 NTERM: gl.constexpr, SUB: gl.constexpr):
    pid_b = gl.program_id(0); pid_m = gl.program_id(1); pid_n = gl.program_id(2)
    a_blk: gl.constexpr = default_blocked_layout([BM, BK], gl.num_warps())
    b_blk: gl.constexpr = default_blocked_layout([BK, BN], gl.num_warps())
    m_: gl.constexpr = 128 if BM >= 128 else 64
    n_: gl.constexpr = 256 if BN >= 256 else BN
    col_stride: gl.constexpr = 32 // gl.float32.primitive_bitwidth
    acc_layout: gl.constexpr = TensorMemoryLayout([m_, n_], col_stride=col_stride)
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BM, BN), acc_layout, gl.num_warps())
    acc0 = gl.zeros([BM, BN], gl.float32, layout=reg_layout)
    acc_tmem = allocate_tensor_memory(gl.float32, [BM, BN], acc_layout, acc0)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    rm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
    rka0 = gl.arange(0, BK, layout=gl.SliceLayout(0, a_blk))[None, :]
    rkb0 = gl.arange(0, BK, layout=gl.SliceLayout(1, b_blk))[:, None]
    rn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_blk))[None, :]
    nchunk = (K + BK - 1) // BK
    reg_acc = gl.zeros([BM, BN], gl.float32, layout=reg_layout)
    for kc in range(nchunk):
        k0 = kc * BK
        rka = rka0 + k0; rkb = rkb0 + k0
        a = gl.load(A + pid_b * sab + rm * sam + rka * sak, mask=(rm < M) & (rka < K), other=0.0)
        b = gl.load(B + pid_b * sbb + rkb * sbk + rn * sbn, mask=(rkb < K) & (rn < N), other=0.0)
        ah = _hi(a); al = a - ah; bh = _hi(b); bl = b - bh
        ah_s = get_shared_memory_mma_operand(ah, 0, False)
        al_s = get_shared_memory_mma_operand(al, 0, False)
        bh_s = get_shared_memory_mma_operand(bh, 1, False)
        bl_s = get_shared_memory_mma_operand(bl, 1, False)
        fence_async_shared()
        mbarrier.init(bar, count=NTERM)
        # each chunk computes its FULL partial product into acc_tmem (use_acc=False resets),
        # then we add it to the register accumulator -> no cross-chunk TMEM dependency.
        tcgen05_mma(ah_s, bh_s, acc_tmem, use_acc=False, mbarriers=[bar])
        tcgen05_mma(ah_s, bl_s, acc_tmem, use_acc=True, mbarriers=[bar])
        tcgen05_mma(al_s, bh_s, acc_tmem, use_acc=True, mbarriers=[bar])
        if NTERM == 4:
            tcgen05_mma(al_s, bl_s, acc_tmem, use_acc=True, mbarriers=[bar])
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        reg_acc += acc_tmem.load(reg_layout)
    out = reg_acc
    c_blk: gl.constexpr = default_blocked_layout([BM, BN], gl.num_warps())
    out = gl.convert_layout(out, c_blk)
    cm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, c_blk))[:, None]
    cn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, c_blk))[None, :]
    cmask = (cm < M) & (cn < N)
    cptr = C + pid_b * scb + cm * scm + cn * scn
    if SUB:
        prev = gl.load(cptr, mask=cmask, other=0.0)
        gl.store(cptr, prev - out, mask=cmask)
    else:
        gl.store(cptr, out, mask=cmask)


def gemm(A, B, C=None, sub=False, nterm=4, BM=64, BN=128, BK=128, nw=8):
    Bb, M, K = A.shape; N = B.shape[2]
    if C is None:
        C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = (Bb, triton.cdiv(M, BM), triton.cdiv(N, BN))
    k = _tf32xN_gemm[grid](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                           M, N, K, BM=BM, BN=BN, BK=BK, NTERM=nterm, SUB=sub, num_warps=nw)
    return C, k


def apply_async(V, T, C, nterm=4):
    # C -= V @ (T^T @ (V^T @ C)).  V:[B,m,b]  T:[B,b,b]  C:[B,m,Wd]
    Vt = V.transpose(1, 2).contiguous()          # [B,b,m]
    W1, _ = gemm(Vt, C, nterm=nterm)             # [B,b,Wd] = V^T @ C
    Tt = T.transpose(1, 2).contiguous()          # [B,b,b]
    W2, _ = gemm(Tt, W1, nterm=nterm)            # [B,b,Wd] = T^T @ W1
    gemm(V, W2, C=C, sub=True, nterm=nterm)      # C -= V @ W2
    return C


# ----- torch references -----
def apply_ref(V, T, C):
    return C - V @ (T.transpose(1, 2) @ (V.transpose(1, 2) @ C))


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=40):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:" + repr(e)[:200]
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    torch.manual_seed(0)

    # --- precision of the K-looped GEMM: NTERM 3 vs 4 (large K) ---
    print("\n=== GEMM precision NTERM 3 vs 4 (relerr vs fp64) ===")
    for Bb, M, K, N in [(8, 128, 512, 384), (8, 128, 128, 128)]:
        A = torch.randn(Bb, M, K, device="cuda"); Bm = torch.randn(Bb, K, N, device="cuda")
        ref = (A.double() @ Bm.double()).float()
        for nt in (3, 4):
            try:
                C, k = gemm(A, Bm, nterm=nt); torch.cuda.synchronize()
            except Exception as e:
                import traceback; print(f"  K{K} nt{nt}: FAIL\n" + traceback.format_exc()[-1500:]); return
            err = (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            print(f"  {Bb}x{M}x{K}x{N} NTERM={nt}: relerr={err:.2e}  n_regs={getattr(k,'n_regs','?')} sp={getattr(k,'n_spills','?')}")

    # --- compact-WY apply correctness (vs torch) using real unit reflectors ---
    print("\n=== compact-WY apply  C -= V@(T^T@(V^T@C))  (relerr vs torch) ===")
    for Bb, m, b, Wd in [(8, 512, 128, 384), (8, 256, 128, 256)]:
        A = torch.randn(Bb, m, b, device="cuda")
        V = torch.tril(A, -1); V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        G = V.transpose(1, 2) @ V
        T = torch.linalg.inv(torch.triu(G))          # arbitrary well-cond T for the math test
        C = torch.randn(Bb, m, Wd, device="cuda")
        ref = apply_ref(V, T, C.clone())
        for nt in (3, 4):
            out = apply_async(V, T, C.clone(), nterm=nt); torch.cuda.synchronize()
            err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            print(f"  m{m} b{b} Wd{Wd} NTERM={nt}: apply relerr={err:.2e}  {'OK' if err < 1e-4 else 'CHECK'}")

    # --- speed: apply vs current path (B=640, first super-panel: m=512 b=128 Wd=384) ---
    print("\n=== speed: compact-WY apply (B=640 m=512 b=128 Wd=384) ===")
    Bb, m, b, Wd = 640, 512, 128, 384
    A = torch.randn(Bb, m, b, device="cuda")
    V = torch.tril(A, -1); V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    G = V.transpose(1, 2) @ V; T = torch.linalg.inv(torch.triu(G))
    C = torch.randn(Bb, m, Wd, device="cuda")
    t4 = time_fn(lambda: apply_async(V, T, C.clone(), nterm=4))
    t3 = time_fn(lambda: apply_async(V, T, C.clone(), nterm=3))
    print(f"  apply_async NTERM=4: {t4 if isinstance(t4,str) else f'{t4:.1f} us'}")
    print(f"  apply_async NTERM=3: {t3 if isinstance(t3,str) else f'{t3:.1f} us'}")
    print("\nSTAGE 1B DONE (need: apply relerr<1e-4, NTERM=4 tightens precision, time sane)")


if __name__ == "__main__":
    main()
