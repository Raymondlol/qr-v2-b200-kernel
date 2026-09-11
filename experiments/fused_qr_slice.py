"""M1 vertical slice: ONE CTA factors ONE [N,N] matrix end-to-end, in-kernel, serial.

This is the FUSED one-shot kernel (one launch, all sub-panels in a kernel loop, NO host
round-trip, NO per-op relaunch). Implements the M0-validated right-looking ib=16 blocked
Householder QR: per sub-panel { factor 16 cols (rowmagma-style) -> in-kernel LARFT T16 ->
compact-WY apply C -= V @ T^T @ (V^T C) to the trailing }. tf32x3 GEMMs (tl.dot).

Unoptimized on purpose (plain Triton, tl.dot, full-row tiles, generous barriers): M1's job is
a CORRECT whole-kernel measured WHOLE. Gluon/tcgen05/warp_specialize/TMA come at M3+. Per the
methodology shift: never judge a piece in isolation; the gate is the whole kernel vs V9-whole.

No banned substrings.
"""
import torch, triton
import triton.language as tl


@triton.jit
def _fused_qr(Hptr, tauptr, sb, sr, sc, stb,
              N: tl.constexpr, IB: tl.constexpr, BN: tl.constexpr,
              BM: tl.constexpr, BW: tl.constexpr, APPLY: tl.constexpr = 1):
    bid = tl.program_id(0)
    base = Hptr + bid * sb
    rar = tl.arange(0, BN)            # local panel-row index (0..BN)
    car = tl.arange(0, IB)           # panel-col index (0..IB)
    ii = tl.arange(0, IB)            # generic IB index

    c0 = 0
    while c0 < N:
        M = N - c0                    # active rows of this sub-panel
        # ---------- (a) PANEL FACTOR: H[c0:N, c0:c0+IB] -> reflectors + R, tau ----------
        prow = c0 + rar
        pcol = c0 + car
        tmask = prow < N
        tptr = base + prow[:, None] * sr + pcol[None, :] * sc
        tile = tl.load(tptr, mask=tmask[:, None], other=0.0)
        for jj in range(IB):
            colj = tl.sum(tl.where(car[None, :] == jj, tile, 0.0), axis=1)     # [BN] = tile[:,jj]
            alpha = tl.sum(tl.where(rar == jj, colj, 0.0))
            xnorm2 = tl.sum(tl.where(rar > jj, colj * colj, 0.0))
            normfull = tl.sqrt(alpha * alpha + xnorm2)
            sgn = tl.where(alpha >= 0.0, 1.0, -1.0)
            beta = -sgn * normfull
            need = xnorm2 > 0.0
            scale = tl.where(need, 1.0 / (alpha - beta), 0.0)
            tau_jj = tl.where(need, (beta - alpha) / beta, 0.0)
            v = tl.where(rar > jj, colj * scale, 0.0)
            v = tl.where(rar == jj, 1.0, v)
            w = tl.sum(v[:, None] * tile, axis=0)                              # [IB]
            upd = tile - tau_jj * (v[:, None] * w[None, :])
            tile = tl.where(car[None, :] > jj, upd, tile)
            diagval = tl.where(need, beta, alpha)
            newcol = tl.where(rar > jj, colj * scale, colj)
            newcol = tl.where(rar == jj, diagval, newcol)
            tile = tl.where(car[None, :] == jj, newcol[:, None], tile)
            tl.store(tauptr + bid * stb + c0 + jj, tau_jj)
        tl.store(tptr, tile, mask=tmask[:, None])
        tl.debug_barrier()

        if (c0 + IB < N) and (APPLY == 1):
            # ---------- (b) gram G = V^T V  [IB,IB] (K = M), then LARFT -> T [IB,IB] ----------
            tau_vec = tl.load(tauptr + bid * stb + c0 + ii)                    # [IB]
            G = tl.zeros((IB, IB), dtype=tl.float32)
            mb = 0
            while mb < M:
                lp = mb + tl.arange(0, BM)                                     # local rows
                gr = c0 + lp
                rmask = gr < N
                # Vt [IB, BM] (unit-lower, transposed)
                vtptr = base + (c0 + car)[:, None] * sc + gr[None, :] * sr
                vt_raw = tl.load(vtptr, mask=rmask[None, :], other=0.0)
                vt = tl.where(lp[None, :] > car[:, None], vt_raw,
                              tl.where(lp[None, :] == car[:, None], 1.0, 0.0))
                # V [BM, IB] (unit-lower)
                vptr = base + gr[:, None] * sr + (c0 + car)[None, :] * sc
                v_raw = tl.load(vptr, mask=rmask[:, None], other=0.0)
                vv = tl.where(lp[:, None] > car[None, :], v_raw,
                              tl.where(lp[:, None] == car[None, :], 1.0, 0.0))
                G += tl.dot(vt, vv, input_precision="tf32x3")
                mb += BM
            # LARFT forward recurrence: T upper-tri, T[0,0]=tau0
            t0 = tl.sum(tl.where(ii == 0, tau_vec, 0.0))
            T = tl.where((ii[:, None] == 0) & (ii[None, :] == 0), t0, 0.0)     # [IB,IB]
            for i in range(1, IB):
                ti = tl.sum(tl.where(ii == i, tau_vec, 0.0))
                gcol = tl.sum(tl.where(ii[None, :] == i, G, 0.0), axis=1)      # [IB] = G[:,i]
                z = tl.where(ii < i, gcol, 0.0)
                m = tl.sum(T * z[None, :], axis=1)                             # [IB] = T@z
                newc = tl.where(ii < i, -ti * m, tl.where(ii == i, ti, 0.0))
                T = tl.where(ii[None, :] == i, newc[:, None], T)
            Tt = tl.trans(T)                                                   # T^T (lower-tri)

            # ---------- (c) APPLY to trailing cols [c0+IB, N): C -= V @ (T^T (V^T C)) ----------
            nb = c0 + IB
            while nb < N:
                qcol = nb + tl.arange(0, BW)
                cmask_n = qcol < N
                # phase 1: W1 [IB,BW] = V^T @ C
                W1 = tl.zeros((IB, BW), dtype=tl.float32)
                mb = 0
                while mb < M:
                    lp = mb + tl.arange(0, BM)
                    gr = c0 + lp
                    rmask = gr < N
                    vtptr = base + (c0 + car)[:, None] * sc + gr[None, :] * sr
                    vt_raw = tl.load(vtptr, mask=rmask[None, :], other=0.0)
                    vt = tl.where(lp[None, :] > car[:, None], vt_raw,
                                  tl.where(lp[None, :] == car[:, None], 1.0, 0.0))
                    cptr = base + gr[:, None] * sr + qcol[None, :] * sc
                    Ct = tl.load(cptr, mask=rmask[:, None] & cmask_n[None, :], other=0.0)
                    W1 += tl.dot(vt, Ct, input_precision="tf32x3")
                    mb += BM
                W2 = tl.dot(Tt, W1, input_precision="tf32x3")                  # [IB,BW]
                # phase 2: C -= V @ W2
                mb = 0
                while mb < M:
                    lp = mb + tl.arange(0, BM)
                    gr = c0 + lp
                    rmask = gr < N
                    vptr = base + gr[:, None] * sr + (c0 + car)[None, :] * sc
                    v_raw = tl.load(vptr, mask=rmask[:, None], other=0.0)
                    vv = tl.where(lp[:, None] > car[None, :], v_raw,
                                  tl.where(lp[:, None] == car[None, :], 1.0, 0.0))
                    delta = tl.dot(vv, W2, input_precision="tf32x3")           # [BM,BW]
                    cptr = base + gr[:, None] * sr + qcol[None, :] * sc
                    full = rmask[:, None] & cmask_n[None, :]
                    Cold = tl.load(cptr, mask=full, other=0.0)
                    tl.store(cptr, Cold - delta, mask=full)
                    mb += BM
                nb += BW
        tl.debug_barrier()
        c0 += IB


def fused_qr(A, IB=16, BM=64, BW=64, nw=4, APPLY=1):
    """A: [B, N, N] contiguous. Returns (H, tau) compact-Householder."""
    B, N, _ = A.shape
    H = A.clone().contiguous()
    tau = torch.zeros(B, N, device=A.device, dtype=A.dtype)
    sb, sr, sc = H.stride()
    BN = triton.next_power_of_2(N)
    k = _fused_qr[(B,)](H, tau, sb, sr, sc, tau.stride(0),
                        N=N, IB=IB, BN=BN, BM=BM, BW=BW, APPLY=APPLY, num_warps=nw)
    return H, tau, k


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=30):
    for _ in range(5):
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
    ts.sort(); return ts[len(ts) // 2] * 1000


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    torch.manual_seed(0)

    print("\n=== correctness on REAL benchmark shapes (vs torch.geqrf) ===")
    for N, B in [(32, 20), (176, 40), (352, 40), (512, 640)]:
        A = torch.randn(B, N, N, device="cuda")
        try:
            H, tau, k = fused_qr(A); torch.cuda.synchronize()
        except Exception as e:
            import traceback; print(f"  n={N}: FAIL\n" + traceback.format_exc()[-2200:]); return
        Hg, taug = torch.geqrf(A)
        herr = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
        terr = (tau - taug).abs().max().item() / (taug.abs().max().item() + 1e-30)
        ok = herr < 1e-4 and terr < 1e-4
        print(f"  n={N:4d} b={B:4d}: H relerr={herr:.2e} tau relerr={terr:.2e}  "
              f"regs={getattr(k,'n_regs','?')} sp={getattr(k,'n_spills','?')}  {'OK' if ok else 'MISMATCH'}")

    print("\n=== PROFILE: panel-only (APPLY=0) vs full (APPLY=1), n=512 b=640, nw=4/64/64 ===")
    Bb, N = 640, 512
    A = torch.randn(Bb, N, N, device="cuda")
    t_full = time_fn(lambda: fused_qr(A, APPLY=1))
    t_panel = time_fn(lambda: fused_qr(A, APPLY=0))
    if not isinstance(t_full, str) and not isinstance(t_panel, str):
        print(f"  full={t_full:.1f}us  panel-only={t_panel:.1f}us  -> apply~{t_full-t_panel:.1f}us "
              f"(panel {100*t_panel/t_full:.0f}% / apply {100*(t_full-t_panel)/t_full:.0f}%)")

    print("\n=== M3 occupancy/tiling sweep on n=512 b=640 (V9 ~12252us) ===")
    Bb, N = 640, 512
    A = torch.randn(Bb, N, N, device="cuda")
    for BM, BW, nw in [(128, 128, 8), (64, 64, 8), (64, 64, 4), (128, 64, 8),
                       (64, 128, 8), (64, 64, 16), (64, 96, 8), (96, 96, 8)]:
        try:
            H, tau, k = fused_qr(A, BM=BM, BW=BW, nw=nw); torch.cuda.synchronize()
            Hg, _ = torch.geqrf(A)
            herr = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
        except Exception as e:
            print(f"  BM={BM} BW={BW} nw={nw}: ERR {repr(e)[:90]}"); continue
        t = time_fn(lambda BM=BM, BW=BW, nw=nw: fused_qr(A, BM=BM, BW=BW, nw=nw))
        ts = t if isinstance(t, str) else f"{t:8.1f}us"
        rr = t / 12252.0 if not isinstance(t, str) else float('nan')
        print(f"  BM={BM:3d} BW={BW:3d} nw={nw:2d}: {ts}  {rr:.3f}x V9  "
              f"regs={getattr(k,'n_regs','?')} sp={getattr(k,'n_spills','?')} herr={herr:.0e}")
    print("\nM3-sweep DONE.")


if __name__ == "__main__":
    main()
