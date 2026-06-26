"""STAGE 0.5 — make-or-break co-residence test for the crux.

Swap the register-heavy default panel (255 regs / 310 spills, which made design-A REGRESS
per the handover) for the Stage-0 winner: the MAGMA row-distributed panel at ib=16
(108 regs, 0 spills). Put it as the DEFAULT partition of gl.warp_specialize, co-resident
with the low-level async tcgen05 trailing worker (8w / 128r). Confirm:
  (1) the fused WS kernel COMPILES within the 16-warp / 64K register budget (low n_regs,
      0/low spills) -> co-residence is REAL, not just a hand-computed budget;
  (2) panel + trailing are both CORRECT under WS;
  (3) overlap efficiency eff = (t_seq - t_ws) / (t_seq - max(panel,trail)) -> GO if > ~0.5.

No banned substrings.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from stage0_regfile_panel import _panel_rowmagma
from gluon_panel_async import _async_trail_part, panel_ref


@gluon.jit
def fused_rm(Hptr, tauptr, SB, SR, STAU, M,
             A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
             PBN: gl.constexpr, PBCOLS: gl.constexpr, RPT: gl.constexpr, NW: gl.constexpr,
             TK: gl.constexpr, TBM: gl.constexpr, TBN: gl.constexpr,
             TMT: gl.constexpr, TNT: gl.constexpr, NPASS: gl.constexpr,
             MODE: gl.constexpr, WK_WARPS: gl.constexpr, WK_REGS: gl.constexpr):
    if MODE == 1:    # WS: rowmagma panel (default partition) || async tcgen05 trailing (worker)
        gl.warp_specialize(
            [(_panel_rowmagma, (Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS, RPT, PBCOLS, 32, 1, NW, 1)),
             (_async_trail_part, (A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                                  TK, TBM, TBN, TMT, TNT, NPASS))],
            [WK_WARPS], [WK_REGS])
    elif MODE == 0:  # SEQ
        _panel_rowmagma(Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS, RPT, PBCOLS, 32, 1, NW, 1)
        _async_trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                          TK, TBM, TBN, TMT, TNT, NPASS)
    elif MODE == 2:  # panel only
        _panel_rowmagma(Hptr, tauptr, SB, SR, STAU, M, PBN, PBCOLS, RPT, PBCOLS, 32, 1, NW, 1)
    else:            # trailing only
        _async_trail_part(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                          TK, TBM, TBN, TMT, TNT, NPASS)


def run(P, A, Bm, MODE, nw=8, wk_warps=8, wk_regs=128, npass=3, TBM=128, TBN=128):
    B, M, b = P.shape
    _, TM, TK = A.shape; TN = Bm.shape[2]
    BN = triton.next_power_of_2(M); BCOLS = triton.next_power_of_2(b)
    RPT = BN // (32 * nw)
    assert RPT * 32 * nw == BN
    H = P.clone().contiguous(); tau = torch.zeros(B, b, device="cuda")
    C = torch.empty(B, TM, TN, device="cuda")
    sb, sr, _ = H.stride()
    k = fused_rm[(B,)](H, tau, sb, sr, tau.stride(0), M,
                       A, Bm, C, *A.stride(), *Bm.stride(), *C.stride(),
                       PBN=BN, PBCOLS=BCOLS, RPT=RPT, NW=nw,
                       TK=TK, TBM=TBM, TBN=TBN, TMT=triton.cdiv(TM, TBM), TNT=triton.cdiv(TN, TBN),
                       NPASS=npass, MODE=MODE, WK_WARPS=wk_warps, WK_REGS=wk_regs, num_warps=nw)
    return H, tau, C, k


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
    B = 640

    # ---- (1)+(2) co-residence COMPILE + correctness (M=384, ib=16 panel) ----
    print("\n=== co-residence compile + correctness (M=384, ib=16) ===")
    P = torch.randn(B, 384, 16, device="cuda")
    A = torch.randn(B, 384, 128, device="cuda")
    Bm = torch.randn(B, 128, 256, device="cuda")
    ref_ab = 3 * (A.double() @ Bm.double()).float()
    for MODE, nm in [(2, "panel-only"), (3, "trail-only"), (0, "SEQ"), (1, "WS")]:
        try:
            H, tau, C, k = run(P, A, Bm, MODE)
            torch.cuda.synchronize()
        except Exception as e:
            import traceback; print(f"  {nm} FAIL\n" + traceback.format_exc()[-2500:]); continue
        Hr, _ = panel_ref(P)
        herr = (H - Hr).abs().max().item() / (Hr.abs().max().item() + 1e-9)
        cerr = (C - ref_ab).abs().max().item() / (ref_ab.abs().max().item() + 1e-9)
        nreg = getattr(k, "n_regs", "?"); nsp = getattr(k, "n_spills", "?")
        ok = (herr < 1e-4) and (MODE in (2,) or cerr < 5e-3)
        print(f"  {nm:11s}: panel relerr={herr:.2e} trail relerr={cerr:.2e}  "
              f"n_regs={nreg} spills={nsp}  {'OK' if ok else 'CHECK'}")

    # ---- (3) overlap efficiency, real-ish n=512 sub-step shapes, ib=16 panel ----
    print("\n=== overlap: rowmagma ib=16 panel(8w) || async tcgen05 trailing(worker) ===")
    print(f"  {'M':>5}{'panel':>8}{'trail':>8}{'sum':>8}{'max':>8} | {'SEQ':>8}{'WS':>8} | {'spd':>6}{'eff':>6} {'regs/sp':>9}")
    for M in [512, 384, 256]:
        TM = M; TN = max(128, 512 - M)
        P = torch.randn(B, M, 16, device="cuda")
        A = torch.randn(B, TM, 128, device="cuda")
        Bm = torch.randn(B, 128, TN, device="cuda")
        t_panel = time_fn(lambda: run(P, A, Bm, 2))
        t_trail = time_fn(lambda: run(P, A, Bm, 3))
        for wkw, wkr in [(8, 128), (8, 160), (4, 128)]:
            t_seq = time_fn(lambda: run(P, A, Bm, 0, wk_warps=wkw, wk_regs=wkr))
            t_ws = time_fn(lambda: run(P, A, Bm, 1, wk_warps=wkw, wk_regs=wkr))
            try:
                _, _, _, kws = run(P, A, Bm, 1, wk_warps=wkw, wk_regs=wkr)
                rsp = f"{getattr(kws,'n_regs','?')}/{getattr(kws,'n_spills','?')}"
            except Exception:
                rsp = "?"
            if any(isinstance(t, str) for t in (t_panel, t_trail, t_seq, t_ws)):
                print(f"  M={M} wk={wkw}/{wkr} panel={t_panel} trail={t_trail} seq={t_seq} ws={t_ws}"); continue
            s = t_panel + t_trail; mx = max(t_panel, t_trail)
            spd = t_seq / t_ws; eff = (t_seq - t_ws) / (t_seq - mx + 1e-9)
            print(f"  {M:>5}{t_panel:8.1f}{t_trail:8.1f}{s:8.1f}{mx:8.1f} | {t_seq:8.1f}{t_ws:8.1f} | "
                  f"{spd:6.2f}{eff:6.2f} {rsp:>9}  wk={wkw}w/{wkr}r")
    print("\nSTAGE 0.5 DONE (GO: WS compiles low-reg/0-spill AND eff>~0.5 -> design A revived)")


if __name__ == "__main__":
    main()
