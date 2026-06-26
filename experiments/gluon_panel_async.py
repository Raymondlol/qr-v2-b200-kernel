"""PHASE 1.5 — design-A RESURRECTION test: the REAL register-heavy Gluon panel (default 8w)
co-resident with a LOW-LEVEL ASYNC tcgen05 trailing worker (8w, private mbarrier, low-register
— NOT tl_dot, which CTA-barrier-couples the partitions and kills overlap). Does the real panel
overlap the async trailing in one CTA at grid=640? GO -> design A is salvageable. No banned subs.
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


@gluon.jit
def _panel_part(Hptr, tauptr, SB, SR, STAU, M, BN: gl.constexpr, BCOLS: gl.constexpr):
    bid = gl.program_id(0)
    t_blk: gl.constexpr = default_blocked_layout([BN, BCOLS], gl.num_warps())
    rl: gl.constexpr = gl.SliceLayout(1, t_blk)
    cl: gl.constexpr = gl.SliceLayout(0, t_blk)
    rar = gl.arange(0, BN, layout=rl)
    car = gl.arange(0, BCOLS, layout=cl)
    ptr = Hptr + bid * SB + rar[:, None] * SR + car[None, :]
    tmask = (rar[:, None] < M)
    tile = gl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        colj = gl.sum(gl.where(car[None, :] == j, tile, 0.0), axis=1)
        alpha = gl.sum(gl.where(rar == j, colj, 0.0))
        xnorm2 = gl.sum(gl.where(rar > j, colj * colj, 0.0))
        normfull = gl.sqrt(alpha * alpha + xnorm2)
        sgn = gl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -sgn * normfull
        need = xnorm2 > 0.0
        scale = gl.where(need, 1.0 / (alpha - beta), 0.0)
        tau_j = gl.where(need, (beta - alpha) / beta, 0.0)
        v = gl.where(rar > j, colj * scale, 0.0)
        v = gl.where(rar == j, 1.0, v)
        w = gl.sum(v[:, None] * tile, axis=0)
        upd = tile - tau_j * (v[:, None] * w[None, :])
        tile = gl.where(car[None, :] > j, upd, tile)
        diagval = gl.where(need, beta, alpha)
        newcol = gl.where(rar > j, colj * scale, colj)
        newcol = gl.where(rar == j, diagval, newcol)
        tile = gl.where(car[None, :] == j, newcol[:, None], tile)
        gl.store(tauptr + bid * STAU + j, tau_j)
    gl.store(ptr, tile, mask=tmask)


# Low-level async tcgen05 trailing worker: tiles the [TM,TN] output (K=TK), each output tile
# = NPASS async tcgen05 mmas into a private TMEM acc with a private mbarrier; wait per tile.
@gluon.jit
def _async_trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                      TK: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
                      MT: gl.constexpr, NT: gl.constexpr, NPASS: gl.constexpr):
    bid = gl.program_id(0)
    a_blk: gl.constexpr = default_blocked_layout([BM, TK], gl.num_warps())
    b_blk: gl.constexpr = default_blocked_layout([TK, BN], gl.num_warps())
    m: gl.constexpr = 128 if BM >= 128 else 64
    n: gl.constexpr = 256 if BN >= 256 else BN
    col_stride: gl.constexpr = 32 // gl.float32.primitive_bitwidth
    acc_layout: gl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BM, BN), acc_layout, gl.num_warps())
    for mt in range(MT):
        for nt in range(NT):
            rm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
            rka = gl.arange(0, TK, layout=gl.SliceLayout(0, a_blk))[None, :]
            rkb = gl.arange(0, TK, layout=gl.SliceLayout(1, b_blk))[:, None]
            rn = nt * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_blk))[None, :]
            a = gl.load(A + bid * sab + rm * sam + rka * sak)
            b = gl.load(B + bid * sbb + rkb * sbk + rn * sbn)
            a_smem = get_shared_memory_mma_operand(a, 0, False)
            b_smem = get_shared_memory_mma_operand(b, 1, False)
            acc0 = gl.zeros([BM, BN], gl.float32, layout=reg_layout)
            acc_tmem = allocate_tensor_memory(gl.float32, [BM, BN], acc_layout, acc0)
            fence_async_shared()
            bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
            mbarrier.init(bar, count=NPASS)
            for _p in range(NPASS):
                tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=(_p > 0), mbarriers=[bar])
            mbarrier.wait(bar, phase=0)
            mbarrier.invalidate(bar)
            out = acc_tmem.load(reg_layout)
            c_blk: gl.constexpr = default_blocked_layout([BM, BN], gl.num_warps())
            out = gl.convert_layout(out, c_blk)
            cm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, c_blk))[:, None]
            cn = nt * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, c_blk))[None, :]
            gl.store(C + bid * scb + cm * scm + cn * scn, out)


@gluon.jit
def fused(Hptr, tauptr, SB, SR, STAU, M,
          A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
          PBN: gl.constexpr, PBCOLS: gl.constexpr,
          TK: gl.constexpr, TBM: gl.constexpr, TBN: gl.constexpr,
          TMT: gl.constexpr, TNT: gl.constexpr, NPASS: gl.constexpr,
          MODE: gl.constexpr, WK_WARPS: gl.constexpr, WK_REGS: gl.constexpr):
    if MODE == 1:    # WS: real panel (default) || async tcgen05 trailing (worker)
        gl.warp_specialize(
            [(_panel_part, (Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)),
             (_async_trail_part, (A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                                  TK, TBM, TBN, TMT, TNT, NPASS))],
            [WK_WARPS], [WK_REGS])
    elif MODE == 0:  # SEQ
        _panel_part(Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)
        _async_trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                          TK, TBM, TBN, TMT, TNT, NPASS)
    elif MODE == 2:  # panel only
        _panel_part(Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)
    else:            # trailing only
        _async_trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                          TK, TBM, TBN, TMT, TNT, NPASS)


def panel_ref(P):
    P = P.clone(); B, m, b = P.shape
    tau = torch.zeros(B, b, dtype=P.dtype, device=P.device)
    for j in range(b):
        alpha = P[:, j, j]; x = P[:, j + 1:, j]
        xn = torch.linalg.vector_norm(x, dim=1)
        nf = torch.sqrt(alpha * alpha + xn * xn)
        sgn = torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        beta = -sgn * nf; mask = xn > 0
        sd = torch.where(mask, alpha - beta, torch.ones_like(alpha))
        sbq = torch.where(mask, beta, torch.ones_like(beta))
        tj = torch.where(mask, (beta - alpha) / sbq, torch.zeros_like(alpha))
        vt = torch.where(mask.unsqueeze(1), x / sd.unsqueeze(1), torch.zeros_like(x))
        P[:, j + 1:, j] = vt; P[:, j, j] = torch.where(mask, beta, alpha); tau[:, j] = tj
        if j < b - 1:
            o = torch.ones(B, 1, dtype=P.dtype, device=P.device)
            V = torch.cat([o, vt], dim=1); sub = P[:, j:, j + 1:]
            w = torch.einsum('bm,bmt->bt', V, sub)
            sub.sub_(tj.view(B, 1, 1) * torch.einsum('bm,bt->bmt', V, w))
    return P, tau


def run(P, A, Bm, MODE, nw=8, wk_warps=8, wk_regs=128, npass=3, TBM=128, TBN=128):
    B, M, b = P.shape
    _, TM, TK = A.shape; TN = Bm.shape[2]
    H = P.clone().contiguous(); tau = torch.zeros(B, b, device="cuda")
    C = torch.empty(B, TM, TN, device="cuda")
    sb, sr, _ = H.stride()
    fused[(B,)](H, tau, sb, sr, tau.stride(0), M,
                A, Bm, C, *A.stride(), *Bm.stride(), *C.stride(),
                PBN=triton.next_power_of_2(M), PBCOLS=triton.next_power_of_2(b),
                TK=TK, TBM=TBM, TBN=TBN, TMT=triton.cdiv(TM, TBM), TNT=triton.cdiv(TN, TBN),
                NPASS=npass, MODE=MODE, WK_WARPS=wk_warps, WK_REGS=wk_regs, num_warps=nw)
    return H, tau, C


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
    B = 640
    print("\n=== correctness (M=384) ===")
    P = torch.randn(B, 384, 64, device="cuda")
    A = torch.randn(B, 384, 128, device="cuda")
    Bm = torch.randn(B, 128, 256, device="cuda")
    ref_ab = 3 * (A.double() @ Bm.double()).float()
    for MODE, nm in [(0, "SEQ"), (1, "WS")]:
        try:
            H, tau, C = run(P, A, Bm, MODE)
            torch.cuda.synchronize()
        except Exception as e:
            import traceback; print(f"  {nm} FAIL"); traceback.print_exc(); return
        Hr, _ = panel_ref(P)
        herr = (H - Hr).abs().max().item() / (Hr.abs().max().item() + 1e-9)
        cerr = (C - ref_ab).abs().max().item() / (ref_ab.abs().max().item() + 1e-9)
        print(f"  {nm}: panel relerr={herr:.2e} trailing relerr={cerr:.2e} "
              f"{'OK' if herr < 1e-4 and cerr < 5e-3 else 'BAD'}")

    print("\n=== RESURRECTION: real panel(8w) || async tcgen05 trailing(worker) by height ===")
    print(f"  {'M':>5}{'panel':>8}{'trail':>8}{'sum':>8}{'max':>8} | {'SEQ':>8}{'WS':>8} | {'spdup':>6}{'eff':>6}")
    for M in [384, 256, 128]:
        TM = M; TN = max(128, 512 - M)
        P = torch.randn(B, M, 64, device="cuda")
        A = torch.randn(B, TM, 128, device="cuda")
        Bm = torch.randn(B, 128, TN, device="cuda")
        t_panel = time_fn(lambda: run(P, A, Bm, 2))
        t_trail = time_fn(lambda: run(P, A, Bm, 3))
        for wkw, wkr in [(8, 128), (4, 128)]:
            t_seq = time_fn(lambda: run(P, A, Bm, 0, wk_warps=wkw, wk_regs=wkr))
            t_ws = time_fn(lambda: run(P, A, Bm, 1, wk_warps=wkw, wk_regs=wkr))
            if any(isinstance(t, str) for t in (t_panel, t_trail, t_seq, t_ws)):
                print(f"  M={M} wk={wkw} panel={t_panel} trail={t_trail} seq={t_seq} ws={t_ws}"); continue
            s = t_panel + t_trail; mx = max(t_panel, t_trail)
            spd = t_seq / t_ws; eff = (t_seq - t_ws) / (t_seq - mx + 1e-9)
            print(f"  {M:>5}{t_panel:8.1f}{t_trail:8.1f}{s:8.1f}{mx:8.1f} | {t_seq:8.1f}{t_ws:8.1f} | {spd:6.2f}{eff:6.2f}  wk={wkw}w/{wkr}r")
    print("\nRESURRECTION DONE (GO if eff>~0.5 -> design A salvageable with async-tcgen05 worker)")


if __name__ == "__main__":
    main()
