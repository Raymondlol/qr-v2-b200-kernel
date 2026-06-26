"""PHASE 1.5 step 2 — THE de-risk: the REAL Gluon panel reduction (default partition,
register-heavy [512,64] tile) co-resident in ONE CTA with a real tcgen05 trailing GEMM
(worker partition) via gl.warp_specialize, at n=512 grid=640. Does overlap eff stay ~0.8
with the real panel's register/smem footprint? GO -> commit to the full Phase 2 build.

Worker uses blocking tl_dot x3 (tf32x3-equiv): warp_specialize gives the concurrency, so the
worker's blocking MMA does NOT stall the panel (separate warps). No banned substrings.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import tl_dot, default_blocked_layout


# ----- panel partition (ported _panel_kernel; returns None, writes H+tau) -----
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


# ----- trailing partition: single-CTA tiled tcgen05 GEMM C=A@B, 3 passes (tf32x3-equiv) -----
@gluon.jit
def _trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                MT: gl.constexpr, NT: gl.constexpr, NPASS: gl.constexpr):
    bid = gl.program_id(0)
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
                a = gl.load(A + bid * sab + rm * sam + (k0 + rka) * sak)
                b = gl.load(B + bid * sbb + (k0 + rkb) * sbk + rn * sbn)
                for _p in range(NPASS):
                    acc = tl_dot(a, b, acc=acc, input_precision="tf32")
            cm = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, c_blk))[:, None]
            cn = nt * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, c_blk))[None, :]
            gl.store(C + bid * scb + cm * scm + cn * scn, acc)


@gluon.jit
def fused(Hptr, tauptr, SB, SR, STAU, M,
          A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
          PBN: gl.constexpr, PBCOLS: gl.constexpr,
          TK: gl.constexpr, TBM: gl.constexpr, TBN: gl.constexpr, TBK: gl.constexpr,
          TMT: gl.constexpr, TNT: gl.constexpr, NPASS: gl.constexpr,
          MODE: gl.constexpr, WK_WARPS: gl.constexpr, WK_REGS: gl.constexpr):
    if MODE == 0:  # SEQ: panel then trailing (no overlap)
        _panel_part(Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)
        _trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                    TK, TBM, TBN, TBK, TMT, TNT, NPASS)
    elif MODE == 1:  # WS: panel (default) || trailing (worker)
        gl.warp_specialize(
            [(_panel_part, (Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)),
             (_trail_part, (A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                            TK, TBM, TBN, TBK, TMT, TNT, NPASS))],
            [WK_WARPS], [WK_REGS])
    elif MODE == 2:  # panel only
        _panel_part(Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)
    else:            # trailing only
        _trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                    TK, TBM, TBN, TBK, TMT, TNT, NPASS)


def panel_ref(P):
    P = P.clone(); B, m, b = P.shape
    tau = torch.zeros(B, b, dtype=P.dtype, device=P.device)
    for j in range(b):
        alpha = P[:, j, j]; x = P[:, j + 1:, j]
        xnorm = torch.linalg.vector_norm(x, dim=1)
        normfull = torch.sqrt(alpha * alpha + xnorm * xnorm)
        sign = torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        beta = -sign * normfull; mask = xnorm > 0
        sd = torch.where(mask, alpha - beta, torch.ones_like(alpha))
        sbq = torch.where(mask, beta, torch.ones_like(beta))
        tau_j = torch.where(mask, (beta - alpha) / sbq, torch.zeros_like(alpha))
        vtail = torch.where(mask.unsqueeze(1), x / sd.unsqueeze(1), torch.zeros_like(x))
        P[:, j + 1:, j] = vtail; P[:, j, j] = torch.where(mask, beta, alpha); tau[:, j] = tau_j
        if j < b - 1:
            ones = torch.ones(B, 1, dtype=P.dtype, device=P.device)
            V = torch.cat([ones, vtail], dim=1); sub = P[:, j:, j + 1:]
            w = torch.einsum('bm,bmt->bt', V, sub)
            sub.sub_(tau_j.view(B, 1, 1) * torch.einsum('bm,bt->bmt', V, w))
    return P, tau


def make(B, M, b, TM, TK, TN):
    P = torch.randn(B, M, b, device="cuda")
    A = torch.randn(B, TM, TK, device="cuda")
    Bm = torch.randn(B, TK, TN, device="cuda")
    return P, A, Bm


def run(P, A, Bm, MODE, nw=8, wk_warps=4, wk_regs=64, npass=3,
        TBM=128, TBN=128, TBK=128):
    B, M, b = P.shape
    _, TM, TK = A.shape; TN = Bm.shape[2]
    H = P.clone().contiguous(); tau = torch.zeros(B, b, device="cuda")
    C = torch.empty(B, TM, TN, device="cuda")
    sb, sr, _ = H.stride()
    fused[(B,)](H, tau, sb, sr, tau.stride(0), M,
                A, Bm, C, *A.stride(), *Bm.stride(), *C.stride(),
                PBN=triton.next_power_of_2(M), PBCOLS=triton.next_power_of_2(b),
                TK=TK, TBM=TBM, TBN=TBN, TBK=TBK,
                TMT=triton.cdiv(TM, TBM), TNT=triton.cdiv(TN, TBN), NPASS=npass,
                MODE=MODE, WK_WARPS=wk_warps, WK_REGS=wk_regs, num_warps=nw)
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
    B, b = 640, 64
    NPASS = 3
    # In the look-ahead pipeline the panel that co-resides with trailing(k) is panel(k+1),
    # whose HEIGHT shrinks (n-c-NB): 512-panel runs alone in the prologue; the real co-resident
    # panels are 384/256/128. Sweep height + a representative trailing.
    print("\n=== correctness (M=384) ===")
    Pc, Ac, Bc = make(B, 384, b, 384, 128, 256)
    ref_ab = NPASS * (Ac.double() @ Bc.double()).float()
    for MODE, nm in [(0, "SEQ"), (1, "WS")]:
        try:
            H, tau, C = run(Pc, Ac, Bc, MODE, npass=NPASS)
            torch.cuda.synchronize()
        except Exception as e:
            import traceback; print(f"  {nm} FAIL"); traceback.print_exc(); return
        Hr, _ = panel_ref(Pc)
        herr = (H - Hr).abs().max().item() / (Hr.abs().max().item() + 1e-9)
        cerr = (C - ref_ab).abs().max().item() / (ref_ab.abs().max().item() + 1e-9)
        print(f"  {nm}: panel relerr={herr:.2e}  trailing(x{NPASS}) relerr={cerr:.2e}  "
              f"{'OK' if herr < 1e-4 and cerr < 5e-3 else 'BAD'}")

    print("\n=== overlap by panel HEIGHT (grid=640, panel default nw=8, worker 4w/64r) ===")
    print(f"  {'M':>5}{'panel':>8}{'trail':>8}{'sum':>8}{'max':>8} | {'SEQ':>8}{'WS':>8} | {'spdup':>6}{'eff':>6}")
    for M in [384, 256, 128]:
        TM = M; TN = max(128, 512 - M)   # trailing width grows as panel shrinks (rough real trend)
        P, A, Bm = make(B, M, b, TM, 128, TN)
        t_panel = time_fn(lambda: run(P, A, Bm, 2))
        t_trail = time_fn(lambda: run(P, A, Bm, 3, wk_warps=4))
        t_seq = time_fn(lambda: run(P, A, Bm, 0, wk_warps=4, wk_regs=64))
        t_ws = time_fn(lambda: run(P, A, Bm, 1, wk_warps=4, wk_regs=64))
        if any(isinstance(t, str) for t in (t_panel, t_trail, t_seq, t_ws)):
            print(f"  M={M} panel={t_panel} trail={t_trail} seq={t_seq} ws={t_ws}"); continue
        s = t_panel + t_trail; mx = max(t_panel, t_trail)
        spd = t_seq / t_ws; eff = (t_seq - t_ws) / (t_seq - mx + 1e-9)
        print(f"  {M:>5}{t_panel:8.1f}{t_trail:8.1f}{s:8.1f}{mx:8.1f} | {t_seq:8.1f}{t_ws:8.1f} | {spd:6.2f}{eff:6.2f}")
    print("\nPHASE1.5 DONE (GO if eff>~0.6 for the real co-resident panel heights 384/256/128)")


if False:
    pass


if __name__ == "__main__":
    main()
