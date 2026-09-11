"""M2.1b — NARROW-NB engine: NB=16/32 super-panel (cheap physical panel tile) + a TMEM-padded
tcgen05 apply (pads the NB-dim to NBP=max(NB,64) so TMEM's 64x32 minimum is met; the extra rows are
zero-V -> zero gram/W rows, harmless). The apply is M-poor (NBP=64 with NB real cols) but the PANEL
is 4-16x cheaper. Decisive question: does panel(NB=16, ~2993us) + this padded apply land the FULL
kernel UNDER V10 (9805us)? If yes, the narrow-NB engine is the path and the overlap is gravy.

Reuses gfused._larft_t / _mma3 / _hi. Padded apply is a fresh fn (NBP pad). No banned substrings.
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
from gfused import _hi, _mma3, _larft_t, _gpanel, clear_l2, time_fn


@gluon.jit
def _narrow_apply(base, tauptr, bid, stb, sr, sc, k, t0, t1, M,
                  NB: gl.constexpr, NBP: gl.constexpr,
                  BM: gl.constexpr, BW: gl.constexpr, BK: gl.constexpr, nw: gl.constexpr):
    # Same compact-WY apply as gfused but with all NB-dim tiles PADDED to NBP (>=64) so TMEM is legal.
    # V columns [NB:NBP] are forced 0 (unit-lower mask uses car<NB). gram/W rows [NB:NBP] = 0.
    _CS: gl.constexpr = 32 // gl.float32.primitive_bitwidth
    m_: gl.constexpr = 64
    a_blk: gl.constexpr = default_blocked_layout([NBP, BK], nw)
    g_blk: gl.constexpr = default_blocked_layout([BK, NBP], nw)
    gacc_l: gl.constexpr = TensorMemoryLayout([m_, NBP], col_stride=_CS)
    greg_l: gl.constexpr = get_tmem_reg_layout(gl.float32, (NBP, NBP), gacc_l, nw)
    nchunk = (M + BK - 1) // BK
    rb = gl.arange(0, NBP, layout=gl.SliceLayout(1, a_blk))[:, None]
    rka = gl.arange(0, BK, layout=gl.SliceLayout(0, a_blk))[None, :]
    rkc = gl.arange(0, BK, layout=gl.SliceLayout(1, g_blk))[:, None]
    cb = gl.arange(0, NBP, layout=gl.SliceLayout(0, g_blk))[None, :]
    gbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    g_tmem = allocate_tensor_memory(gl.float32, [NBP, NBP], gacc_l,
                                    gl.zeros([NBP, NBP], gl.float32, layout=greg_l))
    G = gl.zeros([NBP, NBP], gl.float32, layout=greg_l)
    for kc in range(nchunk):
        lra = kc * BK + rka
        lrc = kc * BK + rkc
        # Vt[NBP,BK]: row r = reflector col (k+r); rows r>=NB are padded 0.
        vt_raw = gl.load(base + (k + rb) * sc + (k + lra) * sr, mask=(lra < M) & (rb < NB), other=0.0)
        vt = gl.where(lra > rb, vt_raw, gl.where((lra == rb) & (rb < NB), 1.0, 0.0))
        v_raw = gl.load(base + (k + lrc) * sr + (k + cb) * sc, mask=(lrc < M) & (cb < NB), other=0.0)
        vv = gl.where(lrc > cb, v_raw, gl.where((lrc == cb) & (cb < NB), 1.0, 0.0))
        _mma3(vt, vv, g_tmem, gbar)
        G += g_tmem.load(greg_l)
    nb_blk: gl.constexpr = default_blocked_layout([NBP, NBP], nw)
    Gb = gl.convert_layout(G, nb_blk)
    iic = gl.arange(0, NBP, layout=gl.SliceLayout(0, nb_blk))
    tau_vec = gl.load(tauptr + bid * stb + k + iic, mask=(iic < NB), other=0.0)
    # LARFT over NBP (padded rows have tau=0 -> identity T rows, harmless on zero W).
    Tt = _larft_t(Gb, tau_vec, NBP, nb_blk)
    wreg_l: gl.constexpr = get_tmem_reg_layout(gl.float32, (NBP, BW),
                                               TensorMemoryLayout([m_, BW], col_stride=_CS), nw)
    wacc_l: gl.constexpr = TensorMemoryLayout([m_, BW], col_stride=_CS)
    c_blk: gl.constexpr = default_blocked_layout([BK, BW], nw)
    rkc2 = gl.arange(0, BK, layout=gl.SliceLayout(1, c_blk))[:, None]
    cnq = gl.arange(0, BW, layout=gl.SliceLayout(0, c_blk))[None, :]
    Tb = gl.convert_layout(Tt, nb_blk)
    nb = t0
    while nb < t1:
        wbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        w_tmem = allocate_tensor_memory(gl.float32, [NBP, BW], wacc_l,
                                        gl.zeros([NBP, BW], gl.float32, layout=wreg_l))
        W1 = gl.zeros([NBP, BW], gl.float32, layout=wreg_l)
        for kc in range(nchunk):
            lra = kc * BK + rka
            lrc = kc * BK + rkc2
            vt_raw = gl.load(base + (k + rb) * sc + (k + lra) * sr, mask=(lra < M) & (rb < NB), other=0.0)
            vt = gl.where(lra > rb, vt_raw, gl.where((lra == rb) & (rb < NB), 1.0, 0.0))
            cc = gl.load(base + (k + lrc) * sr + (nb + cnq) * sc,
                         mask=(lrc < M) & (nb + cnq < t1), other=0.0)
            _mma3(vt, cc, w_tmem, wbar)
            W1 += w_tmem.load(wreg_l)
        w2bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        w2_tmem = allocate_tensor_memory(gl.float32, [NBP, BW], wacc_l,
                                         gl.zeros([NBP, BW], gl.float32, layout=wreg_l))
        nb_blkW: gl.constexpr = default_blocked_layout([NBP, BW], nw)
        W1b = gl.convert_layout(W1, nb_blkW)
        _mma3(Tb, W1b, w2_tmem, w2bar)
        W2 = w2_tmem.load(wreg_l)
        W2b = gl.convert_layout(W2, nb_blkW)
        va_blk: gl.constexpr = default_blocked_layout([BM, NBP], nw)
        m2_: gl.constexpr = 128 if BM >= 128 else 64
        acc2_l: gl.constexpr = TensorMemoryLayout([m2_, BW], col_stride=_CS)
        reg2_l: gl.constexpr = get_tmem_reg_layout(gl.float32, (BM, BW), acc2_l, nw)
        w2h = _hi(W2b); w2l = W2b - w2h
        w2h_s = get_shared_memory_mma_operand(w2h, 1, False)
        w2l_s = get_shared_memory_mma_operand(w2l, 1, False)
        nmt = (M + BM - 1) // BM
        for mt in range(nmt):
            rm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, va_blk))[:, None]
            rkv = gl.arange(0, NBP, layout=gl.SliceLayout(0, va_blk))[None, :]
            v_raw = gl.load(base + (k + rm) * sr + (k + rkv) * sc, mask=(rm < M) & (rkv < NB), other=0.0)
            vv = gl.where(rm > rkv, v_raw, gl.where((rm == rkv) & (rkv < NB), 1.0, 0.0))
            vh = _hi(vv); vl = vv - vh
            vh_s = get_shared_memory_mma_operand(vh, 0, False)
            vl_s = get_shared_memory_mma_operand(vl, 0, False)
            acc2 = allocate_tensor_memory(gl.float32, [BM, BW], acc2_l,
                                          gl.zeros([BM, BW], gl.float32, layout=reg2_l))
            bar2 = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
            fence_async_shared(); mbarrier.init(bar2, count=3)
            tcgen05_mma(vh_s, w2h_s, acc2, use_acc=False, mbarriers=[bar2])
            tcgen05_mma(vh_s, w2l_s, acc2, use_acc=True, mbarriers=[bar2])
            tcgen05_mma(vl_s, w2h_s, acc2, use_acc=True, mbarriers=[bar2])
            mbarrier.wait(bar2, phase=0); mbarrier.invalidate(bar2)
            delta = acc2.load(reg2_l)
            o_blk: gl.constexpr = default_blocked_layout([BM, BW], nw)
            delta = gl.convert_layout(delta, o_blk)
            cm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, o_blk))[:, None]
            cq = nb + gl.arange(0, BW, layout=gl.SliceLayout(0, o_blk))[None, :]
            cmask = (cm < M) & (cq < t1)
            cptr = base + (k + cm) * sr + cq * sc
            prev = gl.load(cptr, mask=cmask, other=0.0)
            gl.store(cptr, prev - delta, mask=cmask)
        nb += BW


@gluon.jit
def _narrow_fused(Hptr, tauptr, sb, sr, sc, stb,
                  N: gl.constexpr, NB: gl.constexpr, NBP: gl.constexpr, BN: gl.constexpr,
                  RPT: gl.constexpr, BM: gl.constexpr, BW: gl.constexpr, BK: gl.constexpr,
                  nw: gl.constexpr):
    bid = gl.program_id(0)
    base = Hptr + bid * sb
    k = 0
    while k < N:
        M = N - k
        _gpanel(base, tauptr, bid, stb, sr, sc, k, M, BN, NB, RPT, nw)
        gl.thread_barrier()
        if k + NB < N:
            _narrow_apply(base, tauptr, bid, stb, sr, sc, k, k + NB, N, M, NB, NBP, BM, BW, BK, nw)
        gl.thread_barrier()
        k += NB


def narrow_qr(A, NB=16, NBP=64, BM=64, BW=64, BK=64, nw=8):
    B, n, _ = A.shape
    H = A.clone().contiguous()
    tau = torch.zeros(B, n, device=A.device, dtype=A.dtype)
    sb, sr, sc = H.stride()
    BN = triton.next_power_of_2(n)
    RPT = BN // (32 * nw)
    k = _narrow_fused[(B,)](H, tau, sb, sr, sc, tau.stride(0),
                            N=n, NB=NB, NBP=NBP, BN=BN, RPT=RPT, BM=BM, BW=BW, BK=BK, nw=nw, num_warps=nw)
    return H, tau, k


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    torch.manual_seed(0)
    print("\n=== M2.1b correctness (grid=1) vs torch.geqrf ===")
    for N in [256, 512]:
        for NB in [16, 32]:
            A = torch.randn(1, N, N, device="cuda")
            try:
                H, tau, k = narrow_qr(A, NB=NB, NBP=64); torch.cuda.synchronize()
            except Exception as e:
                import traceback; print(f"  N={N} NB={NB}: FAIL\n" + traceback.format_exc()[-1800:]); continue
            Hg, taug = torch.geqrf(A[0])
            herr = (H[0] - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
            terr = (tau[0] - taug).abs().max().item() / (taug.abs().max().item() + 1e-30)
            print(f"  N={N:4d} NB={NB}: H={herr:.2e} tau={terr:.2e} regs={getattr(k,'n_regs','?')} "
                  f"sp={getattr(k,'n_spills','?')} {'OK' if herr<1e-4 and terr<1e-4 else 'BAD'}")

    print("\n=== M2.1b WHOLE n=512 b=640 vs V10 (9805us) ===")
    A = torch.randn(640, 512, 512, device="cuda")
    Hg, _ = torch.geqrf(A)
    for NB, NBP, BM, BW, BK, nw in [(16, 64, 64, 64, 64, 8), (16, 64, 64, 128, 64, 8),
                                    (32, 64, 64, 128, 64, 8), (32, 64, 64, 256, 64, 8)]:
        try:
            H, tau, k = narrow_qr(A, NB=NB, NBP=NBP, BM=BM, BW=BW, BK=BK, nw=nw); torch.cuda.synchronize()
            herr = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
        except Exception as e:
            print(f"  NB={NB} BW={BW}: ERR {repr(e)[:90]}"); continue
        t = time_fn(lambda NB=NB, NBP=NBP, BM=BM, BW=BW, BK=BK, nw=nw:
                    narrow_qr(A, NB=NB, NBP=NBP, BM=BM, BW=BW, BK=BK, nw=nw))
        ts = t if isinstance(t, str) else f"{t:8.1f}us"
        rr = t / 9805.0 if not isinstance(t, str) else float('nan')
        print(f"  NB={NB} NBP={NBP} BM={BM} BW={BW:3d} BK={BK} nw={nw}: {ts} {rr:.3f}x V10 "
              f"regs={getattr(k,'n_regs','?')} sp={getattr(k,'n_spills','?')} herr={herr:.0e}")
    print("\nM2.1b DONE.")


if __name__ == "__main__":
    main()
