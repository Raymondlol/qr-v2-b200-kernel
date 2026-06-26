"""STAGE 1D — the BUILDABLE design-A core: warp_specialize( rowmagma 1 sub-panel ||
far-trailing 2-GEMM slice ).  Avoids the hard in-kernel super-panel factor: within-applies /
gram / T stay torch; only the latency-bound rowmagma panel (CUDA-core) overlaps the far-trailing
GEMM (tensor-core). W1 lives in SMEM (slice SLW small -> no OOM). Worker uses async tcgen05
(NOT tl_dot, which couples partitions).

Measure: (1) panel + worker both CORRECT under WS; (2) eff = (seq-ws)/(seq-max) and net vs the
sequential equivalent, at real n=512 sub-step shapes. GO -> wire into _factor_custom + lab. No
banned substrings.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory, tcgen05_mma, TensorMemoryLayout, mbarrier,
    fence_async_shared, get_tmem_reg_layout,
)
from triton.tools.triton_to_gluon_translater.translator_helpers import (
    get_shared_memory_mma_operand, default_blocked_layout,
)
from stage0_regfile_panel import _panel_rowmagma, panel_ref

_SW = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])


@gluon.jit
def _hi(x):
    return ((x.to(gl.int32, bitcast=True)) & -8192).to(gl.float32, bitcast=True)


@gluon.jit
def _trail_slice_worker(Vt, VT, C, svtb, svtr, svtc, sVTb, sVTr, sVTc, scb, scr, scc,
                        M, BB: gl.constexpr, SLW: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr):
    # one CTA, one matrix (bid). C is a [M, SLW] slice (in place). Vt=[BB,M], VT=[M,BB].
    # Phase1: W1[BB,SLW] = Vt @ C (K-loop over M, reg-accumulate). Phase2: C -= VT @ W1 (K=BB).
    bid = gl.program_id(0)
    nw: gl.constexpr = gl.num_warps()
    a_blk: gl.constexpr = default_blocked_layout([BB, BK], nw)
    c_blk: gl.constexpr = default_blocked_layout([BK, SLW], nw)
    m_: gl.constexpr = 128 if BB >= 128 else 64
    col_stride: gl.constexpr = 32 // gl.float32.primitive_bitwidth
    acc_layout: gl.constexpr = TensorMemoryLayout([m_, SLW], col_stride=col_stride)
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BB, SLW), acc_layout, nw)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    acc0 = gl.zeros([BB, SLW], gl.float32, layout=reg_layout)
    acc_tmem = allocate_tensor_memory(gl.float32, [BB, SLW], acc_layout, acc0)
    rb = gl.arange(0, BB, layout=gl.SliceLayout(1, a_blk))[:, None]
    rk0a = gl.arange(0, BK, layout=gl.SliceLayout(0, a_blk))[None, :]
    rk0c = gl.arange(0, BK, layout=gl.SliceLayout(1, c_blk))[:, None]
    cn = gl.arange(0, SLW, layout=gl.SliceLayout(0, c_blk))[None, :]
    w1_reg = gl.zeros([BB, SLW], gl.float32, layout=reg_layout)
    nchunk = (M + BK - 1) // BK
    for kc in range(nchunk):
        rka = rk0a + kc * BK; rkc = rk0c + kc * BK
        a = gl.load(Vt + bid * svtb + rb * svtr + rka * svtc, mask=(rka < M), other=0.0)
        c = gl.load(C + bid * scb + rkc * scr + cn * scc, mask=(rkc < M), other=0.0)
        ah = _hi(a); al = a - ah; ch = _hi(c); cl = c - ch
        ah_s = get_shared_memory_mma_operand(ah, 0, False)
        al_s = get_shared_memory_mma_operand(al, 0, False)
        ch_s = get_shared_memory_mma_operand(ch, 1, False)
        cl_s = get_shared_memory_mma_operand(cl, 1, False)
        fence_async_shared()
        mbarrier.init(bar, count=3)
        tcgen05_mma(ah_s, ch_s, acc_tmem, use_acc=False, mbarriers=[bar])
        tcgen05_mma(ah_s, cl_s, acc_tmem, use_acc=True, mbarriers=[bar])
        tcgen05_mma(al_s, ch_s, acc_tmem, use_acc=True, mbarriers=[bar])
        mbarrier.wait(bar, phase=0); mbarrier.invalidate(bar)
        w1_reg += acc_tmem.load(reg_layout)
    w1_blk: gl.constexpr = default_blocked_layout([BB, SLW], nw)
    w1_sm = gl.allocate_shared_memory(gl.float32, [BB, SLW], _SW)
    w1_sm.store(gl.convert_layout(w1_reg, w1_blk))
    fence_async_shared()
    # Phase 2
    va_blk: gl.constexpr = default_blocked_layout([BM, BB], nw)
    wb_blk: gl.constexpr = default_blocked_layout([BB, SLW], nw)
    m2_: gl.constexpr = 128 if BM >= 128 else 64
    acc2_layout: gl.constexpr = TensorMemoryLayout([m2_, SLW], col_stride=col_stride)
    reg2_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BM, SLW), acc2_layout, nw)
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
        acc20 = gl.zeros([BM, SLW], gl.float32, layout=reg2_layout)
        acc2 = allocate_tensor_memory(gl.float32, [BM, SLW], acc2_layout, acc20)
        bar2 = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        fence_async_shared()
        mbarrier.init(bar2, count=3)
        tcgen05_mma(vh_s, w1h_s, acc2, use_acc=False, mbarriers=[bar2])
        tcgen05_mma(vh_s, w1l_s, acc2, use_acc=True, mbarriers=[bar2])
        tcgen05_mma(vl_s, w1h_s, acc2, use_acc=True, mbarriers=[bar2])
        mbarrier.wait(bar2, phase=0); mbarrier.invalidate(bar2)
        delta = acc2.load(reg2_layout)
        o_blk: gl.constexpr = default_blocked_layout([BM, SLW], nw)
        delta = gl.convert_layout(delta, o_blk)
        cm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, o_blk))[:, None]
        co = gl.arange(0, SLW, layout=gl.SliceLayout(0, o_blk))[None, :]
        cmask = (cm < M)
        cptr = C + bid * scb + cm * scr + co * scc
        prev = gl.load(cptr, mask=cmask, other=0.0)
        gl.store(cptr, prev - delta, mask=cmask)


@gluon.jit
def _fused(Hptr, tauptr, SB, SR, STAU, MP,
           Vt, VT, C, svtb, svtr, svtc, sVTb, sVTr, sVTc, scb, scr, scc, MT,
           PBN: gl.constexpr, PBCOLS: gl.constexpr, RPT: gl.constexpr, NW: gl.constexpr,
           BB: gl.constexpr, SLW: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr,
           MODE: gl.constexpr, WK_WARPS: gl.constexpr, WK_REGS: gl.constexpr):
    if MODE == 1:
        gl.warp_specialize(
            [(_panel_rowmagma, (Hptr, tauptr, SB, SR, STAU, MP, PBN, PBCOLS, RPT, PBCOLS, 32, 1, NW, 1)),
             (_trail_slice_worker, (Vt, VT, C, svtb, svtr, svtc, sVTb, sVTr, sVTc, scb, scr, scc,
                                    MT, BB, SLW, BM, BK))],
            [WK_WARPS], [WK_REGS])
    elif MODE == 0:
        _panel_rowmagma(Hptr, tauptr, SB, SR, STAU, MP, PBN, PBCOLS, RPT, PBCOLS, 32, 1, NW, 1)
        _trail_slice_worker(Vt, VT, C, svtb, svtr, svtc, sVTb, sVTr, sVTc, scb, scr, scc,
                            MT, BB, SLW, BM, BK)
    elif MODE == 2:
        _panel_rowmagma(Hptr, tauptr, SB, SR, STAU, MP, PBN, PBCOLS, RPT, PBCOLS, 32, 1, NW, 1)
    else:
        _trail_slice_worker(Vt, VT, C, svtb, svtr, svtc, sVTb, sVTr, sVTc, scb, scr, scc,
                            MT, BB, SLW, BM, BK)


def run(P, Vk, VTk, C, MODE, nw=8, wk_warps=8, wk_regs=128, SLW=32, BM=64, BK=128):
    Bb, MP, b = P.shape           # panel sub-panel tile [MP, 16]
    _, MTr, BB = Vk.shape         # block reflectors [MT, 128]
    BN = triton.next_power_of_2(MP); BCOLS = triton.next_power_of_2(b); RPT = BN // (32 * nw)
    H = P.clone().contiguous(); tau = torch.zeros(Bb, b, device="cuda")
    Vt = Vk.transpose(1, 2).contiguous()
    Cw = C.clone()
    sb, sr, _ = H.stride()
    k = _fused[(Bb,)](H, tau, sb, sr, tau.stride(0), MP,
                      Vt, VTk, Cw, *Vt.stride(), *VTk.stride(), *Cw.stride(), MTr,
                      PBN=BN, PBCOLS=BCOLS, RPT=RPT, NW=nw, BB=BB, SLW=SLW, BM=BM, BK=BK,
                      MODE=MODE, WK_WARPS=wk_warps, WK_REGS=wk_regs, num_warps=nw)
    return H, tau, Cw, k


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
    Bb = 640
    # panel sub-panel = [M, 16]; far-trailing block = [M, 128] applied to a [M, SLW] slice
    print("\n=== correctness (M=512 panel-subpanel, BB=128 block, SLW=32 slice) ===")
    MP = 512; SLW = 32
    P = torch.randn(Bb, MP, 16, device="cuda")
    A = torch.randn(Bb, MP, 128, device="cuda")
    Vk = torch.tril(A, -1); Vk.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    G = Vk.transpose(1, 2) @ Vk; T = torch.linalg.inv(torch.triu(G)); VTk = (Vk @ T.transpose(1, 2)).contiguous()
    C = torch.randn(Bb, MP, SLW, device="cuda")
    ref_apply = C - VTk @ (Vk.transpose(1, 2) @ C)
    Hr, taur = panel_ref(P)
    for MODE, nm in [(2, "panel"), (3, "worker"), (0, "SEQ"), (1, "WS")]:
        try:
            H, tau, Cw, k = run(P, Vk, VTk, C, MODE, SLW=SLW); torch.cuda.synchronize()
        except Exception as e:
            import traceback; print(f"  {nm} FAIL\n" + traceback.format_exc()[-2000:]); continue
        perr = (H - Hr).abs().max().item() / (Hr.abs().max().item() + 1e-9) if MODE in (0, 1, 2) else 0
        cerr = (Cw - ref_apply).abs().max().item() / (ref_apply.abs().max().item() + 1e-9) if MODE in (0, 1, 3) else 0
        print(f"  {nm:6s}: panel relerr={perr:.2e} apply relerr={cerr:.2e}  "
              f"n_regs={getattr(k,'n_regs','?')} sp={getattr(k,'n_spills','?')}")

    print("\n=== overlap eff + net (grid=640, M=512) ===")
    print(f"  {'panel':>8}{'worker':>8}{'sum':>8}{'max':>8} | {'SEQ':>8}{'WS':>8} | {'spd':>6}{'eff':>6}")
    t_p = time_fn(lambda: run(P, Vk, VTk, C, 2, SLW=SLW))
    t_w = time_fn(lambda: run(P, Vk, VTk, C, 3, SLW=SLW))
    for wkw, wkr in [(8, 128), (8, 160)]:
        t_s = time_fn(lambda: run(P, Vk, VTk, C, 0, wk_warps=wkw, wk_regs=wkr, SLW=SLW))
        t_ws = time_fn(lambda: run(P, Vk, VTk, C, 1, wk_warps=wkw, wk_regs=wkr, SLW=SLW))
        if any(isinstance(t, str) for t in (t_p, t_w, t_s, t_ws)):
            print(f"  p={t_p} w={t_w} s={t_s} ws={t_ws}"); continue
        s = t_p + t_w; mx = max(t_p, t_w); spd = t_s / t_ws; eff = (t_s - t_ws) / (t_s - mx + 1e-9)
        print(f"  {t_p:8.1f}{t_w:8.1f}{s:8.1f}{mx:8.1f} | {t_s:8.1f}{t_ws:8.1f} | {spd:6.2f}{eff:6.2f}  wk={wkw}/{wkr}")
    print("\nSTAGE 1D DONE (GO if WS correct AND eff>~0.4 AND net>1.0 -> wire into _factor_custom + lab)")


if __name__ == "__main__":
    main()
