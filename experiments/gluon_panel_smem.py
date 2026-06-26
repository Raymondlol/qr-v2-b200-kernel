"""Stage A.2 — SMEM-RESIDENT STREAMING panel: the tile lives in shared memory; each
j-reflector streams the tile in RBLK-row blocks (only one block in registers at a time) so
the register peak is ~[RBLK,BCOLS] instead of the whole [BN,BCOLS]. Goal: low register count
so panel(8w) + async-trailing-worker(8w) fits one CTA -> design-A overlap.

2 streaming passes per reflector j:
  PASS A (reduce): colj, alpha=col[j], xnorm2=sum_{r>j} col^2, w0=sum_{r>j} col[r]*tile[r,:],
                   tile_j=tile[j,:].  Then v = e_j + scale*col_{>j}, w = tile_j + scale*w0.
  PASS B (update): tile[:,>j] -= tau*v(x)w ; write column j (=v tail + beta diag).
Initial load + final store are ALSO streamed. Bit-faithful vs panel_ref. No banned substrings.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import default_blocked_layout
from gluon_panel import _panel_gl as _panel_reg   # register-resident baseline

_SW = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])


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


@gluon.jit
def _panel_smem(Hptr, tauptr, SB, SR, STAU, M,
                BN: gl.constexpr, BCOLS: gl.constexpr, RBLK: gl.constexpr):
    bid = gl.program_id(0)
    NBLK: gl.constexpr = BN // RBLK
    blkr: gl.constexpr = default_blocked_layout([RBLK, BCOLS], gl.num_warps())
    rl: gl.constexpr = gl.SliceLayout(1, blkr)   # [RBLK] row-vec
    cl: gl.constexpr = gl.SliceLayout(0, blkr)   # [BCOLS] col-vec
    car = gl.arange(0, BCOLS, layout=cl)          # [BCOLS]
    loc = gl.arange(0, RBLK, layout=rl)           # [RBLK] local row idx

    # 3D smem [NBLK,RBLK,BCOLS]; RUNTIME block loops (range) + .index(ib) so the compiler
    # reuses ONE block's registers across iterations -> low register peak.
    smem = gl.allocate_shared_memory(gl.float32, [NBLK, RBLK, BCOLS], _SW)
    # --- streamed initial load global -> smem ---
    for ib in range(NBLK):
        rar = ib * RBLK + loc
        gptr = Hptr + bid * SB + rar[:, None] * SR + car[None, :]
        t = gl.load(gptr, mask=(rar[:, None] < M), other=0.0)
        smem.index(ib).store(t)

    for j in range(BCOLS):
        # ---- PASS A: streaming reduction ----
        alpha = 0.0
        xnorm2 = 0.0
        w0 = gl.zeros([BCOLS], gl.float32, layout=cl)
        tile_j = gl.zeros([BCOLS], gl.float32, layout=cl)
        for ib in range(NBLK):
            rar = ib * RBLK + loc
            t = smem.index(ib).load(blkr)                          # [RBLK,BCOLS]
            colj = gl.sum(gl.where(car[None, :] == j, t, 0.0), axis=1)  # [RBLK]
            alpha += gl.sum(gl.where(rar == j, colj, 0.0))
            xnorm2 += gl.sum(gl.where(rar > j, colj * colj, 0.0))
            w0 += gl.sum(gl.where(rar[:, None] > j, colj[:, None] * t, 0.0), axis=0)
            tile_j += gl.sum(gl.where(rar[:, None] == j, t, 0.0), axis=0)
        normfull = gl.sqrt(alpha * alpha + xnorm2)
        sgn = gl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -sgn * normfull
        need = xnorm2 > 0.0
        scale = gl.where(need, 1.0 / (alpha - beta), 0.0)
        tau_j = gl.where(need, (beta - alpha) / beta, 0.0)
        diag = gl.where(need, beta, alpha)
        w = tile_j + scale * w0                                    # [BCOLS]
        # ---- PASS B: streaming update + writeback ----
        for ib in range(NBLK):
            rar = ib * RBLK + loc
            t = smem.index(ib).load(blkr)
            colj = gl.sum(gl.where(car[None, :] == j, t, 0.0), axis=1)  # recompute (col j unchanged)
            v = gl.where(rar > j, colj * scale, 0.0)
            v = gl.where(rar == j, 1.0, v)
            upd = t - tau_j * (v[:, None] * w[None, :])
            t = gl.where(car[None, :] > j, upd, t)
            newcol = gl.where(rar > j, colj * scale, colj)
            newcol = gl.where(rar == j, diag, newcol)
            t = gl.where(car[None, :] == j, newcol[:, None], t)
            smem.index(ib).store(t)
        gl.store(tauptr + bid * STAU + j, tau_j)

    # --- streamed final store smem -> global ---
    for ib in range(NBLK):
        rar = ib * RBLK + loc
        t = smem.index(ib).load(blkr)
        gptr = Hptr + bid * SB + rar[:, None] * SR + car[None, :]
        gl.store(gptr, t, mask=(rar[:, None] < M))


def run(P, RBLK=128, nw=8):
    B, M, b = P.shape
    BN = triton.next_power_of_2(M)
    BCOLS = triton.next_power_of_2(b)
    assert BN == M and BCOLS == b
    H = P.clone().contiguous(); tau = torch.zeros(B, b, device="cuda")
    sb, sr, _ = H.stride()
    k = _panel_smem[(B,)](H, tau, sb, sr, tau.stride(0), M, BN=BN, BCOLS=BCOLS, RBLK=RBLK, num_warps=nw)
    return H, tau, k


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    torch.manual_seed(0)
    for B, M, b, RBLK in [(4, 128, 64, 64), (8, 256, 64, 64), (8, 512, 64, 128),
                          (8, 512, 64, 64), (8, 512, 64, 32), (8, 512, 64, 256)]:
        P = torch.randn(B, M, b, device="cuda")
        try:
            Hg, taug, k = run(P, RBLK=RBLK)
            torch.cuda.synchronize()
        except Exception as e:
            import traceback; print(f"  B{B} M{M} b{b} RBLK{RBLK}: FAIL"); traceback.print_exc(); return
        Hr, taur = panel_ref(P)
        herr = (Hg - Hr).abs().max().item() / (Hr.abs().max().item() + 1e-9)
        terr = (taug - taur).abs().max().item() / (taur.abs().max().item() + 1e-9)
        flag = "OK" if (herr < 1e-4 and terr < 1e-4) else "MISMATCH"
        nreg = getattr(k, "n_regs", "?"); nspill = getattr(k, "n_spills", "?")
        print(f"  B{B} M{M:4d} b{b} RBLK{RBLK}: H relerr={herr:.2e} tau relerr={terr:.2e} {flag} | n_regs={nreg} spills={nspill}")

    # --- Stage A.3: SPEED vs register-resident panel at grid=640 (the other half of the gate) ---
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

    def reg_panel(P, nw=8):
        B, M, b = P.shape
        H = P.clone().contiguous(); tau = torch.zeros(B, b, device="cuda")
        sb, sr, _ = H.stride()
        _panel_reg[(B,)](H, tau, sb, sr, tau.stride(0), M,
                         BN=triton.next_power_of_2(M), BCOLS=triton.next_power_of_2(b), num_warps=nw)

    def smem_panel(P, RBLK, nw=8):
        B, M, b = P.shape
        H = P.clone().contiguous(); tau = torch.zeros(B, b, device="cuda")
        sb, sr, _ = H.stride()
        _panel_smem[(B,)](H, tau, sb, sr, tau.stride(0), M,
                          BN=triton.next_power_of_2(M), BCOLS=triton.next_power_of_2(b), RBLK=RBLK, num_warps=nw)

    print("\n=== SPEED gate (grid=640, M=512, b=64) — smem panel vs register-resident ===")
    P = torch.randn(640, 512, 64, device="cuda")
    t_reg = time_fn(lambda: reg_panel(P))
    print(f"  register-resident panel: {t_reg:.1f} us  (baseline)")
    for RBLK in [256, 128, 64, 32]:
        t = time_fn(lambda: smem_panel(P, RBLK))
        if isinstance(t, str): print(f"  smem RBLK={RBLK}: {t}"); continue
        print(f"  smem RBLK={RBLK:3d}: {t:8.1f} us  {t/t_reg:.2f}x reg  {'<=1.5x OK' if t/t_reg <= 1.5 else 'SLOW'}")
    print("\nSMEM PANEL DONE  (GATE: need regs<=64 AND time<=1.5x reg)")


if __name__ == "__main__":
    main()
