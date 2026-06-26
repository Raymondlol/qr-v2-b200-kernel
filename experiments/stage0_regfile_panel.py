"""STAGE 0 (the crux): MAGMA-style register-FILE Householder panel.

Goal: a panel layout that is BOTH fast (~= the register-resident gluon_panel.py) AND
low-register (<= ~64 regs/thread) so it can co-reside with the 8w/128r async tcgen05
trailing worker inside one 16-warp warp_specialize CTA (design A revival).

The element-count floor: a register-resident [BN,BCOLS] tile over T threads needs
BN*BCOLS/T regs/thread just to HOLD the tile (128 for [512,64] @ 8warps). The current
default_blocked_layout ALSO pays a transient full-tile temporary for the rank-1 update
(v[:,None]*w[None,:]) -> peak ~2-3x resident. The MAGMA trick = own COMPLETE ROWS per
thread ([RPT,BCOLS] size_per_thread, threads split only along rows): then
  - colj / column extraction + the rank-1 update are per-thread IN-PLACE (no transient
    full-tile blowup),
  - the w = sum_rows(v*tile) reduction is a clean cross-thread (warp+smem) reduction,
  - alpha/xnorm2 are cross-thread scalar reductions.
We measure n_regs / n_spills / correctness / speed vs the default-layout register panel,
swept over BCOLS (ib) in {64,32,16} and num_warps in {8,16}.  No banned substrings.

GATE: GO if some config hits regs<=~64/thread AND time<=~1.3x the register panel.
"""
import inspect, torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import default_blocked_layout
from gluon_panel import _panel_gl as _panel_reg, panel_ref


# ---------------- API recon (so a wrong guess is diagnosable in one run) ----------------
def dump_api():
    print("triton", triton.__version__)
    try:
        print("gl.BlockedLayout sig:", inspect.signature(gl.BlockedLayout))
    except Exception as e:
        print("BlockedLayout sig N/A:", e)
    try:
        print("--- default_blocked_layout source ---")
        print(inspect.getsource(default_blocked_layout))
    except Exception as e:
        print("default_blocked_layout src N/A:", e)


# ---------------- MAGMA-style row-distributed register panel ----------------
# Explicit BlockedLayout: each thread owns RPT complete rows x SPT_C columns.
# Pure row-distribution => SPT_C=BCOLS, TPW=[32,1], WPC=[nw,1]; RPT=BN/(32*nw).
@gluon.jit
def _panel_rowmagma(Hptr, tauptr, SB, SR, STAU, M,
                    BN: gl.constexpr, BCOLS: gl.constexpr,
                    RPT: gl.constexpr, SPT_C: gl.constexpr,
                    TPW_R: gl.constexpr, TPW_C: gl.constexpr,
                    WPC_R: gl.constexpr, WPC_C: gl.constexpr):
    bid = gl.program_id(0)
    blk: gl.constexpr = gl.BlockedLayout([RPT, SPT_C], [TPW_R, TPW_C], [WPC_R, WPC_C], [1, 0])
    rl: gl.constexpr = gl.SliceLayout(1, blk)   # [BN] row-vector
    cl: gl.constexpr = gl.SliceLayout(0, blk)   # [BCOLS] col-vector (replicated across row-threads)
    rar = gl.arange(0, BN, layout=rl)
    car = gl.arange(0, BCOLS, layout=cl)
    base = Hptr + bid * SB
    ptr = base + rar[:, None] * SR + car[None, :]
    tmask = (rar[:, None] < M)
    tile = gl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        colj = gl.sum(gl.where(car[None, :] == j, tile, 0.0), axis=1)          # [BN] (within-thread cols)
        alpha = gl.sum(gl.where(rar == j, colj, 0.0))                          # scalar (cross-thread)
        xnorm2 = gl.sum(gl.where(rar > j, colj * colj, 0.0))                   # scalar (cross-thread)
        normfull = gl.sqrt(alpha * alpha + xnorm2)
        sgn = gl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -sgn * normfull
        need = xnorm2 > 0.0
        scale = gl.where(need, 1.0 / (alpha - beta), 0.0)
        tau_j = gl.where(need, (beta - alpha) / beta, 0.0)
        v = gl.where(rar > j, colj * scale, 0.0)
        v = gl.where(rar == j, 1.0, v)                                         # [BN]
        w = gl.sum(v[:, None] * tile, axis=0)                                  # [BCOLS] (cross-thread)
        upd = tile - tau_j * (v[:, None] * w[None, :])                         # per-thread in-place
        tile = gl.where(car[None, :] > j, upd, tile)
        diagval = gl.where(need, beta, alpha)
        newcol = gl.where(rar > j, colj * scale, colj)
        newcol = gl.where(rar == j, diagval, newcol)
        tile = gl.where(car[None, :] == j, newcol[:, None], tile)
        gl.store(tauptr + bid * STAU + j, tau_j)
    gl.store(ptr, tile, mask=tmask)


def run_rowmagma(P, nw, cfg_name):
    B, M, b = P.shape
    BN = triton.next_power_of_2(M)
    BCOLS = triton.next_power_of_2(b)
    RPT = BN // (32 * nw)
    assert RPT * 32 * nw == BN, f"{cfg_name}: BN={BN} not covered by 32*{nw}"
    H = P.clone().contiguous()
    tau = torch.zeros(B, b, dtype=P.dtype, device=P.device)
    sb, sr, _ = H.stride()
    k = _panel_rowmagma[(B,)](H, tau, sb, sr, tau.stride(0), M,
                              BN=BN, BCOLS=BCOLS, RPT=RPT, SPT_C=BCOLS,
                              TPW_R=32, TPW_C=1, WPC_R=nw, WPC_C=1, num_warps=nw)
    return H, tau, k


def run_reg(P, nw):
    B, M, b = P.shape
    H = P.clone().contiguous()
    tau = torch.zeros(B, b, dtype=P.dtype, device=P.device)
    sb, sr, _ = H.stride()
    k = _panel_reg[(B,)](H, tau, sb, sr, tau.stride(0), M,
                         BN=triton.next_power_of_2(M), BCOLS=triton.next_power_of_2(b), num_warps=nw)
    return H, tau, k


# ---------------- timing ----------------
def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=40):
    for _ in range(6):
        try:
            fn()
        except Exception as e:
            return "ERR:" + repr(e)[:200]
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1000


def correctness(label, runner, bcols, m_list):
    print(f"\n=== correctness: {label} ===")
    ok_all = True
    nreg = nspill = "?"
    for M in m_list:
        P = torch.randn(8, M, bcols, device="cuda")
        try:
            Hg, taug, k = runner(P)
            torch.cuda.synchronize()
        except Exception as e:
            import traceback
            print(f"  M={M:4d} b={bcols}: RUN FAIL"); traceback.print_exc()
            ok_all = False
            continue
        Hr, taur = panel_ref(P)
        herr = (Hg - Hr).abs().max().item() / (Hr.abs().max().item() + 1e-9)
        terr = (taug - taur).abs().max().item() / (taur.abs().max().item() + 1e-9)
        ok = herr < 1e-4 and terr < 1e-4
        ok_all = ok_all and ok
        nreg = getattr(k, "n_regs", "?"); nspill = getattr(k, "n_spills", "?")
        print(f"  M={M:4d} b={bcols}: H relerr={herr:.2e} tau relerr={terr:.2e} {'OK' if ok else 'MISMATCH'}"
              f"  n_regs={nreg} spills={nspill}")
    return ok_all, nreg, nspill


def main():
    dump_api()
    print("dev", torch.cuda.get_device_name(0))
    torch.manual_seed(0)

    # --- baseline: default-layout register panel (b=64, nw=8) ---
    correctness("REGISTER panel (default_blocked_layout, nw=8, b=64)",
                lambda P: run_reg(P, 8), 64, [512, 256])

    # --- MAGMA row-distributed variants. m_list chosen so BN divisible by 32*nw. ---
    # CO-RESIDABLE candidates = nw=8 (leaves 8 warps for the trailing worker).
    configs = [
        ("rowmagma b=64 nw=8  (resident128)", 64, 8, [512, 256]),
        ("rowmagma b=32 nw=8  (resident 64)", 32, 8, [512, 256]),
        ("rowmagma b=16 nw=8  (resident 32)", 16, 8, [512, 256]),
        # nw=16 variants are NOT co-residable (panel eats all 16 warps) -> reference only.
        ("rowmagma b=64 nw=16 (resident 64, NOT co-resid)", 64, 16, [512]),
        ("rowmagma b=32 nw=16 (resident 32, NOT co-resid)", 32, 16, [512]),
    ]
    for name, bcols, nw, mlist in configs:
        correctness(name, lambda P, w=nw, nm=name: run_rowmagma(P, w, nm), bcols, mlist)

    # ---------------- SPEED gate (grid=640, M=512) ----------------
    # Work-normalized: per-call time x (64/b) = time to factor 64 columns (the register
    # panel's b=64 super-panel does this in ONE call; a b=32 panel needs 2 calls, etc.).
    print("\n=== SPEED gate (grid=640, M=512). norm = per-call x (64/b) ===")
    base = torch.randn(640, 512, 64, device="cuda")
    t_reg = time_fn(lambda: run_reg(base, 8))
    if isinstance(t_reg, str):
        print(f"  REGISTER baseline FAIL: {t_reg}"); return
    print(f"  REGISTER panel (default, nw=8, b=64): {t_reg:7.1f} us  (BASELINE, norm={t_reg:.1f})")
    for name, bcols, nw, _ in configs:
        Pc = base[:, :, :bcols].contiguous()
        t = time_fn(lambda Pc=Pc, w=nw, nm=name: run_rowmagma(Pc, w, nm))
        if isinstance(t, str):
            print(f"  {name}: {t}")
            continue
        norm = t * (64 / bcols)
        ratio = norm / t_reg
        verdict = "<=1.3x OK" if ratio <= 1.3 else "SLOW"
        print(f"  {name}: per-call {t:7.1f} us  norm {norm:7.1f} us  {ratio:.2f}x reg  {verdict}")
    print("\nSTAGE 0 DONE  (GATE: a CO-RESIDABLE config (nw=8) with regs/spills low AND norm<=~1.3x reg -> GO)")


if __name__ == "__main__":
    main()
