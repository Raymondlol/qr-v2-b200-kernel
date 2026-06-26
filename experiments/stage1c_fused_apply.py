"""STAGE 1C step 1 — RESIDENT-W1 fused compact-WY apply (kill the W1 global round-trip).

The 1B apply was ~1.4x slow partly because W1 = V^T@C round-trips through HBM. Here W1 stays
in SMEM: per Wd-tile, one CTA computes W1_tile[b,BN_W] = Vt @ C_tile (K-loop over m, async
tcgen05), keeps it in smem, then updates C_tile -= VT @ W1_tile (K=b). VT = V@T^T precomputed.
b=128 (fat block) -> both GEMMs are tensor-core-efficient (M=b=128, K=b=128).

Test: (1) correctness vs torch C - VT@(Vt@C); (2) time vs the UNFUSED 2x async-GEMM (global W1)
and vs the current-path _bmm3 2-GEMM. Goal: fused ~<= 1.0-1.1x the current 2-GEMM apply -> the
trailing worker is efficient enough that overlap nets a clear win. No banned substrings.
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

_SW = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])


@gluon.jit
def _hi(x):
    return ((x.to(gl.int32, bitcast=True)) & -8192).to(gl.float32, bitcast=True)


@gluon.jit
def _mma_tf32x3(a_s_h, a_s_l, b_s_h, b_s_l, acc_tmem, bar, first: gl.constexpr):
    # 3-term tf32x3 into acc_tmem (use_acc=first? no -> caller controls reset via 'first')
    mbarrier.init(bar, count=3)
    tcgen05_mma(a_s_h, b_s_h, acc_tmem, use_acc=(not first), mbarriers=[bar])
    tcgen05_mma(a_s_h, b_s_l, acc_tmem, use_acc=True, mbarriers=[bar])
    tcgen05_mma(a_s_l, b_s_h, acc_tmem, use_acc=True, mbarriers=[bar])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)


@gluon.jit
def _fused_apply(Vt, VT, C, svtb, svtr, svtc, sVTb, sVTr, sVTc, scb, scr, scc,
                 M, Wd, BB: gl.constexpr, BNW: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr):
    # grid = (Bb, cdiv(Wd, BNW)).  one CTA owns one [*, BNW] column-tile of C. BB = block width b.
    bid = gl.program_id(0); wt = gl.program_id(1)
    nw: gl.constexpr = gl.num_warps()
    # ---- Phase 1: W1[BB, BNW] = Vt[BB, M] @ C[M, wt*BNW : +BNW], K-loop over M ----
    a_blk: gl.constexpr = default_blocked_layout([BB, BK], nw)      # Vt tile [b, BK]
    c_blk: gl.constexpr = default_blocked_layout([BK, BNW], nw)     # C  tile [BK, BNW]
    m_: gl.constexpr = 128 if BB >= 128 else 64
    n_: gl.constexpr = 256 if BNW >= 256 else BNW
    col_stride: gl.constexpr = 32 // gl.float32.primitive_bitwidth
    acc_layout: gl.constexpr = TensorMemoryLayout([m_, n_], col_stride=col_stride)
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BB, BNW), acc_layout, nw)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    acc0 = gl.zeros([BB, BNW], gl.float32, layout=reg_layout)
    acc_tmem = allocate_tensor_memory(gl.float32, [BB, BNW], acc_layout, acc0)
    rb = gl.arange(0, BB, layout=gl.SliceLayout(1, a_blk))[:, None]
    rk0a = gl.arange(0, BK, layout=gl.SliceLayout(0, a_blk))[None, :]
    rk0c = gl.arange(0, BK, layout=gl.SliceLayout(1, c_blk))[:, None]
    cn = wt * BNW + gl.arange(0, BNW, layout=gl.SliceLayout(0, c_blk))[None, :]
    w1_reg = gl.zeros([BB, BNW], gl.float32, layout=reg_layout)
    nchunk = (M + BK - 1) // BK
    for kc in range(nchunk):
        rka = rk0a + kc * BK; rkc = rk0c + kc * BK
        a = gl.load(Vt + bid * svtb + rb * svtr + rka * svtc, mask=(rka < M), other=0.0)
        c = gl.load(C + bid * scb + rkc * scr + cn * scc, mask=(rkc < M) & (cn < Wd), other=0.0)
        ah = _hi(a); al = a - ah; ch = _hi(c); cl = c - ch
        ah_s = get_shared_memory_mma_operand(ah, 0, False)
        al_s = get_shared_memory_mma_operand(al, 0, False)
        ch_s = get_shared_memory_mma_operand(ch, 1, False)
        cl_s = get_shared_memory_mma_operand(cl, 1, False)
        fence_async_shared()
        _mma_tf32x3(ah_s, al_s, ch_s, cl_s, acc_tmem, bar, first=True)
        w1_reg += acc_tmem.load(reg_layout)
    # stash W1 [BB, BNW] into smem for phase 2
    w1_blk: gl.constexpr = default_blocked_layout([BB, BNW], nw)
    w1_sm = gl.allocate_shared_memory(gl.float32, [BB, BNW], _SW)
    w1_sm.store(gl.convert_layout(w1_reg, w1_blk))
    fence_async_shared()
    # ---- Phase 2: C_tile[M, BNW] -= VT[M, BB] @ W1[BB, BNW], K = BB, loop M-tiles ----
    va_blk: gl.constexpr = default_blocked_layout([BM, BB], nw)      # VT tile [BM, b]
    wb_blk: gl.constexpr = default_blocked_layout([BB, BNW], nw)     # W1 [b, BNW]
    m2_: gl.constexpr = 128 if BM >= 128 else 64
    acc2_layout: gl.constexpr = TensorMemoryLayout([m2_, n_], col_stride=col_stride)
    reg2_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BM, BNW), acc2_layout, nw)
    w1h = _hi(w1_sm.load(wb_blk)); w1l = w1_sm.load(wb_blk) - w1h
    w1h_s = get_shared_memory_mma_operand(w1h, 1, False)
    w1l_s = get_shared_memory_mma_operand(w1l, 1, False)
    nmt = (M + BM - 1) // BM
    for mt in range(nmt):
        rm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, va_blk))[:, None]
        rkv = gl.arange(0, BB, layout=gl.SliceLayout(0, va_blk))[None, :]
        vt2 = gl.load(VT + bid * sVTb + rm * sVTr + rkv * sVTc, mask=(rm < M), other=0.0)
        vh = _hi(vt2); vl = vt2 - vh
        vh_s = get_shared_memory_mma_operand(vh, 0, False)
        vl_s = get_shared_memory_mma_operand(vl, 0, False)
        acc20 = gl.zeros([BM, BNW], gl.float32, layout=reg2_layout)
        acc2 = allocate_tensor_memory(gl.float32, [BM, BNW], acc2_layout, acc20)
        bar2 = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        fence_async_shared()
        mbarrier.init(bar2, count=3)
        tcgen05_mma(vh_s, w1h_s, acc2, use_acc=False, mbarriers=[bar2])
        tcgen05_mma(vh_s, w1l_s, acc2, use_acc=True, mbarriers=[bar2])
        tcgen05_mma(vl_s, w1h_s, acc2, use_acc=True, mbarriers=[bar2])
        mbarrier.wait(bar2, phase=0); mbarrier.invalidate(bar2)
        delta = acc2.load(reg2_layout)
        o_blk: gl.constexpr = default_blocked_layout([BM, BNW], nw)
        delta = gl.convert_layout(delta, o_blk)
        cm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, o_blk))[:, None]
        co = wt * BNW + gl.arange(0, BNW, layout=gl.SliceLayout(0, o_blk))[None, :]
        cmask = (cm < M) & (co < Wd)
        cptr = C + bid * scb + cm * scr + co * scc
        prev = gl.load(cptr, mask=cmask, other=0.0)
        gl.store(cptr, prev - delta, mask=cmask)


def fused_apply(V, T, C, BNW=128, BM=64, BK=128, nw=8):
    Bb, m, b = V.shape; Wd = C.shape[2]
    Vt = V.transpose(1, 2).contiguous()                 # [B,b,m]
    VT = (V @ T.transpose(1, 2)).contiguous()            # [B,m,b]
    Cw = C.clone()
    grid = (Bb, triton.cdiv(Wd, BNW))
    k = _fused_apply[grid](Vt, VT, Cw, *Vt.stride(), *VT.stride(), *Cw.stride(),
                           m, Wd, BB=b, BNW=BNW, BM=BM, BK=BK, num_warps=nw)
    return Cw, k


# ---- current-path 2-GEMM apply baseline (tf32x3 _bmm3, global W1) ----
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


def apply_current(V, T, C):
    Vt = V.transpose(1, 2).contiguous()
    W1 = bmm3(Vt, C)                                      # [B,b,Wd]
    Vt2 = (V @ T.transpose(1, 2)).contiguous()
    Cw = C.clone()
    bmm3(Vt2, W1, C=Cw, sub=True)
    return Cw


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
    print("\n=== correctness: resident-W1 fused apply (relerr vs torch) ===")
    for Bb, m, b, Wd in [(8, 512, 128, 256), (8, 512, 128, 384), (8, 384, 128, 128)]:
        A = torch.randn(Bb, m, b, device="cuda")
        V = torch.tril(A, -1); V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        G = V.transpose(1, 2) @ V; T = torch.linalg.inv(torch.triu(G))
        C = torch.randn(Bb, m, Wd, device="cuda")
        ref = C - V @ (T.transpose(1, 2) @ (V.transpose(1, 2) @ C))
        try:
            out, k = fused_apply(V, T, C); torch.cuda.synchronize()
        except Exception as e:
            import traceback; print(f"  m{m} Wd{Wd}: FAIL\n" + traceback.format_exc()[-1800:]); return
        err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        print(f"  m{m} b{b} Wd{Wd}: relerr={err:.2e}  n_regs={getattr(k,'n_regs','?')} sp={getattr(k,'n_spills','?')}  {'OK' if err<1e-4 else 'CHECK'}")

    print("\n=== speed: fat apply (B=640 m=512 b=128 Wd=256) ===")
    Bb, m, b, Wd = 640, 512, 128, 256
    A = torch.randn(Bb, m, b, device="cuda")
    V = torch.tril(A, -1); V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    G = V.transpose(1, 2) @ V; T = torch.linalg.inv(torch.triu(G))
    C = torch.randn(Bb, m, Wd, device="cuda")
    t_cur = time_fn(lambda: apply_current(V, T, C))
    print(f"  current 2x _bmm3 (global W1): {t_cur if isinstance(t_cur,str) else f'{t_cur:.1f} us'}  (BASELINE)")
    for BNW, BM in [(128, 64), (256, 64), (128, 128)]:
        t = time_fn(lambda BNW=BNW, BM=BM: fused_apply(V, T, C, BNW=BNW, BM=BM))
        if isinstance(t, str): print(f"  fused BNW={BNW} BM={BM}: {t}"); continue
        r = t / t_cur if not isinstance(t_cur, str) else float('nan')
        print(f"  fused resident-W1 BNW={BNW} BM={BM}: {t:7.1f} us  {r:.2f}x current")
    print("\nSTAGE 1C-1 DONE (need: relerr<1e-4 AND fused <= ~1.1x current 2-GEMM -> worker efficient)")


if __name__ == "__main__":
    main()
