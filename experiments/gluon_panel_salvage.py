"""PHASE 1.5 salvage DIAGNOSTIC — before rewriting the panel to be shared-memory-resident,
decompose the design-A regression into its two possible causes:
  (1) warp_specialize FIXED overhead (fork/join + co-residence) — probe: TRIVIAL default
      partition (near-zero registers) || real trailing worker, vs trailing-only.
  (2) PANEL REGISTER pressure — probe: real panel default partition, sweep BCOLS
      {64,32,16,8} (register footprint ∝ BCOLS) || trailing worker, watch eff vs BCOLS.
If (1) is small AND (2) eff recovers as BCOLS shrinks -> a smem-resident panel (which cuts the
register footprint) is worth building. If (1) is large -> warp_specialize itself is the wall.
No banned substrings.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import tl_dot, default_blocked_layout


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


@gluon.jit
def _trivial_part(Sptr):  # near-zero-register default partition (just touch one element)
    bid = gl.program_id(0)
    gl.store(Sptr + bid, gl.program_id(0).to(gl.float32))


@gluon.jit
def _redux_part(X, RED, RBM: gl.constexpr, RBR: gl.constexpr, NRED: gl.constexpr):
    # LOW-register reduction proxy for a smem-resident panel: small tile (RBM x RBR) reused,
    # loop NRED -> ~panel-magnitude TIME at LOW register footprint. Tests whether a low-reg
    # default partition overlaps the 8w/128r trailing worker (= the smem-panel viability proxy).
    bid = gl.program_id(0)
    r_blk: gl.constexpr = default_blocked_layout([RBM, RBR], gl.num_warps())
    xm = gl.arange(0, RBM, layout=gl.SliceLayout(1, r_blk))[:, None]
    xr = gl.arange(0, RBR, layout=gl.SliceLayout(0, r_blk))[None, :]
    xt = gl.load(X + xm * RBR + xr)
    red = gl.zeros([RBM], gl.float32, layout=gl.SliceLayout(1, r_blk))
    for j in range(NRED):
        acc = xt + red[:, None] * 1e-30
        red += gl.sum(acc * acc, axis=1)
    gl.store(RED + bid * RBM + gl.arange(0, RBM, layout=gl.SliceLayout(1, r_blk)), red)


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
def fused(Hptr, tauptr, SB, SR, STAU, M, Sptr, Xptr, REDptr,
          A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
          PBN: gl.constexpr, PBCOLS: gl.constexpr,
          TK: gl.constexpr, TBM: gl.constexpr, TBN: gl.constexpr, TBK: gl.constexpr,
          TMT: gl.constexpr, TNT: gl.constexpr, NPASS: gl.constexpr,
          RBM: gl.constexpr, RBR: gl.constexpr, NRED: gl.constexpr,
          MODE: gl.constexpr, WK_WARPS: gl.constexpr, WK_REGS: gl.constexpr):
    if MODE == 1:    # WS: real panel || trailing
        gl.warp_specialize(
            [(_panel_part, (Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)),
             (_trail_part, (A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                            TK, TBM, TBN, TBK, TMT, TNT, NPASS))],
            [WK_WARPS], [WK_REGS])
    elif MODE == 0:  # SEQ: real panel then trailing
        _panel_part(Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)
        _trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                    TK, TBM, TBN, TBK, TMT, TNT, NPASS)
    elif MODE == 2:  # panel only
        _panel_part(Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS)
    elif MODE == 3:  # trailing only
        _trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                    TK, TBM, TBN, TBK, TMT, TNT, NPASS)
    elif MODE == 4:  # WS: TRIVIAL default || trailing (isolates warp_specialize overhead)
        gl.warp_specialize(
            [(_trivial_part, (Sptr,)),
             (_trail_part, (A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                            TK, TBM, TBN, TBK, TMT, TNT, NPASS))],
            [WK_WARPS], [WK_REGS])
    elif MODE == 5:  # SEQ trivial then trailing
        _trivial_part(Sptr)
        _trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                    TK, TBM, TBN, TBK, TMT, TNT, NPASS)
    elif MODE == 6:  # WS: LOW-REG reduction proxy (smem-panel stand-in) || trailing worker
        gl.warp_specialize(
            [(_redux_part, (Xptr, REDptr, RBM, RBR, NRED)),
             (_trail_part, (A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                            TK, TBM, TBN, TBK, TMT, TNT, NPASS))],
            [WK_WARPS], [WK_REGS])
    elif MODE == 7:  # SEQ redux then trailing
        _redux_part(Xptr, REDptr, RBM, RBR, NRED)
        _trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                    TK, TBM, TBN, TBK, TMT, TNT, NPASS)
    elif MODE == 8:  # redux only
        _redux_part(Xptr, REDptr, RBM, RBR, NRED)


def run(P, A, Bm, MODE, nw=8, wk_warps=4, wk_regs=64, npass=3, PBCOLS=None,
        TBM=128, TBN=128, TBK=128, RBM=128, RBR=64, NRED=256, X=None, RED=None):
    B, M, b = P.shape
    _, TM, TK = A.shape; TN = Bm.shape[2]
    if PBCOLS is None:
        PBCOLS = triton.next_power_of_2(b)
    H = P.clone().contiguous(); tau = torch.zeros(B, b, device="cuda")
    C = torch.empty(B, TM, TN, device="cuda"); S = torch.zeros(B, device="cuda")
    if X is None:
        X = torch.randn(RBM, RBR, device="cuda")
    if RED is None:
        RED = torch.empty(B, RBM, device="cuda")
    sb, sr, _ = H.stride()
    fused[(B,)](H, tau, sb, sr, tau.stride(0), M, S, X, RED,
                A, Bm, C, *A.stride(), *Bm.stride(), *C.stride(),
                PBN=triton.next_power_of_2(M), PBCOLS=PBCOLS,
                TK=TK, TBM=TBM, TBN=TBN, TBK=TBK,
                TMT=triton.cdiv(TM, TBM), TNT=triton.cdiv(TN, TBN), NPASS=npass,
                RBM=RBM, RBR=RBR, NRED=NRED,
                MODE=MODE, WK_WARPS=wk_warps, WK_REGS=wk_regs, num_warps=nw)
    return H, tau, C


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=40):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:" + repr(e)[:160]
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
    M = 384                      # a real co-resident panel height
    TM, TK, TN = 384, 128, 256
    P = torch.randn(B, M, 64, device="cuda")
    A = torch.randn(B, TM, TK, device="cuda")
    Bm = torch.randn(B, TK, TN, device="cuda")

    # Decompose the 3.3x worker slowdown: warp-count vs warp_specialize-fundamental.
    print("\n=== trailing-only at different warp counts (no warp_specialize) ===")
    for nw in [8, 4, 2]:
        t = time_fn(lambda: run(P, A, Bm, 3, nw=nw))
        print(f"  trailing-only nw={nw}: {t if isinstance(t,str) else f'{t:.1f} us'}")

    print("\n=== WS(trivial default || trailing worker): worker warp sweep ===")
    print("  (default partition does ~nothing; isolates what warp_specialize does to the worker)")
    for wkw, wkr in [(4, 64), (8, 64), (8, 128), (4, 128)]:
        t_ws = time_fn(lambda: run(P, A, Bm, 4, wk_warps=wkw, wk_regs=wkr))
        print(f"  WS trivial||trail  worker={wkw}w/{wkr}r: {t_ws if isinstance(t_ws,str) else f'{t_ws:.1f} us'}")

    # DECISIVE: real panel (8w default) || trailing worker at the GOOD resourcing (8w/128r).
    print("\n=== DECISIVE: real panel(default 8w) || trailing(worker swept) — does design A work? ===")
    print(f"  {'wkW/R':>9}{'panel':>8}{'trail8':>8} | {'SEQ':>8}{'WS':>8} | {'spdup':>6}{'eff':>6}")
    t_trail8 = time_fn(lambda: run(P, A, Bm, 3, nw=8))
    t_panel = time_fn(lambda: run(P, A, Bm, 2, PBCOLS=64))
    for wkw, wkr in [(8, 128), (8, 96), (4, 128), (8, 64)]:
        t_seq = time_fn(lambda: run(P, A, Bm, 0, wk_warps=wkw, wk_regs=wkr, PBCOLS=64))
        t_ws = time_fn(lambda: run(P, A, Bm, 1, wk_warps=wkw, wk_regs=wkr, PBCOLS=64))
        if any(isinstance(t, str) for t in (t_seq, t_ws)):
            print(f"  {wkw}w/{wkr}r seq={t_seq} ws={t_ws}"); continue
        mx = max(t_panel, t_trail8)
        spd = t_seq / t_ws; eff = (t_seq - t_ws) / (t_seq - mx + 1e-9)
        print(f"  {wkw}w/{wkr}r{t_panel:8.1f}{t_trail8:8.1f} | {t_seq:8.1f}{t_ws:8.1f} | {spd:6.2f}{eff:6.2f}")
    print("\nDIAG DONE (GO if any config gives spdup>1.1 with the REAL panel)")

    # SMEM-PANEL VIABILITY PROXY: a LOW-register reduction default (stand-in for a smem-resident
    # panel) sized to ~panel magnitude, || the trailing worker at 8w/128r. If this overlaps,
    # building the smem panel is worth it; if not, design A is fully dead.
    print("\n=== SMEM-PANEL VIABILITY: low-reg reduction default (8w) || trailing(8w/128r) ===")
    print(f"  {'NRED':>6}{'redux':>8}{'trail8':>8} | {'SEQ':>8}{'WS':>8} | {'spdup':>6}{'eff':>6}")
    Xb = torch.randn(128, 64, device="cuda")
    for NRED in [256, 512, 768]:
        REDb = torch.empty(B, 128, device="cuda")
        t_redux = time_fn(lambda: run(P, A, Bm, 8, nw=8, NRED=NRED, RBM=128, RBR=64, X=Xb, RED=REDb))
        t_seq = time_fn(lambda: run(P, A, Bm, 7, nw=8, wk_warps=8, wk_regs=128, NRED=NRED, RBM=128, RBR=64, X=Xb, RED=REDb))
        t_ws = time_fn(lambda: run(P, A, Bm, 6, nw=8, wk_warps=8, wk_regs=128, NRED=NRED, RBM=128, RBR=64, X=Xb, RED=REDb))
        if any(isinstance(t, str) for t in (t_redux, t_seq, t_ws)):
            print(f"  {NRED}: redux={t_redux} seq={t_seq} ws={t_ws}"); continue
        mx = max(t_redux, t_trail8)
        spd = t_seq / t_ws; eff = (t_seq - t_ws) / (t_seq - mx + 1e-9)
        print(f"  {NRED:>6}{t_redux:8.1f}{t_trail8:8.1f} | {t_seq:8.1f}{t_ws:8.1f} | {spd:6.2f}{eff:6.2f}")
    print("\nSMEM-PANEL PROXY DONE (GO -> build smem panel; NO-GO -> design A fully dead)")


if __name__ == "__main__":
    main()
