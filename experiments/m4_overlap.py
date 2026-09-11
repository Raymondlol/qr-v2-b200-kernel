"""M4 — CORRECT look-ahead overlap PROTOTYPE (panel hidden behind the GEMM).
Sidesteps m2_overlap's broken Tt-smem-staging: each apply RECOMPUTES gram+LARFT internally (reuse the
CORRECT m2_narrowapply._narrow_apply), so NO Tt smem round-trip. warp_specialize(far-trailing[k] on
tensor warps || panel[k+1] on CUDA warps). Correctness FIRST (grid=1 SEQ vs WS), then whole vs V10.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from gfused import _gpanel, clear_l2, time_fn
from m2_narrowapply import _narrow_apply


@gluon.jit
def _trail_part(base, tauptr, bid, stb, sr, sc, k, t0, t1, M,
                NB: gl.constexpr, NBP: gl.constexpr, BM: gl.constexpr, BW: gl.constexpr,
                BK: gl.constexpr, nw: gl.constexpr):
    _narrow_apply(base, tauptr, bid, stb, sr, sc, k, t0, t1, M, NB, NBP, BM, BW, BK, nw)


@gluon.jit
def _panel_part(base, tauptr, bid, stb, sr, sc, knext, Mnext,
                BN: gl.constexpr, NB: gl.constexpr, RPT_WK: gl.constexpr, WK_WARPS: gl.constexpr):
    _gpanel(base, tauptr, bid, stb, sr, sc, knext, Mnext, BN, NB, RPT_WK, WK_WARPS)


@gluon.jit
def _m4_fused(Hptr, tauptr, sb, sr, sc, stb,
              N: gl.constexpr, NB: gl.constexpr, NBP: gl.constexpr, BN: gl.constexpr,
              RPT: gl.constexpr, BM: gl.constexpr, BW: gl.constexpr, BK: gl.constexpr,
              WK_WARPS: gl.constexpr, WK_REGS: gl.constexpr, OVERLAP: gl.constexpr, nw: gl.constexpr):
    bid = gl.program_id(0)
    base = Hptr + bid * sb
    _gpanel(base, tauptr, bid, stb, sr, sc, 0, N, BN, NB, RPT, nw)        # panel[0]
    gl.thread_barrier()
    k = 0
    while k < N:
        M = N - k
        if k + NB < N:
            knext = k + NB
            Mnext = N - knext
            la1 = knext + NB if knext + NB < N else N
            # LOOK-AHEAD: apply block[k] to next-panel cols [knext:la1] (breaks the dep). all warps.
            _narrow_apply(base, tauptr, bid, stb, sr, sc, k, knext, la1, M, NB, NBP, BM, BW, BK, nw)
            gl.thread_barrier()
            far0 = la1
            RPT_WK: gl.constexpr = BN // (32 * WK_WARPS)
            if OVERLAP == 1 and far0 < N and knext + NB < N:
                # OVERLAP: far-apply[k] over [far0:N] (tensor) || factor panel[k+1] (CUDA)
                gl.warp_specialize(
                    [(_trail_part, (base, tauptr, bid, stb, sr, sc, k, far0, N, M,
                                    NB, NBP, BM, BW, BK, nw)),
                     (_panel_part, (base, tauptr, bid, stb, sr, sc, knext, Mnext,
                                    BN, NB, RPT_WK, WK_WARPS))],
                    [WK_WARPS], [WK_REGS])
            else:
                if far0 < N:
                    _narrow_apply(base, tauptr, bid, stb, sr, sc, k, far0, N, M, NB, NBP, BM, BW, BK, nw)
                gl.thread_barrier()
                _gpanel(base, tauptr, bid, stb, sr, sc, knext, Mnext, BN, NB, RPT, nw)  # panel[knext] (knext<N guaranteed)
            gl.thread_barrier()
        k += NB


def m4_qr(A, NB=64, NBP=64, BM=64, BW=128, BK=64, nw=8, wk_warps=8, wk_regs=160, overlap=1):
    B, n, _ = A.shape
    H = A.clone().contiguous()
    tau = torch.zeros(B, n, device=A.device, dtype=A.dtype)
    sb, sr, sc = H.stride()
    BN = triton.next_power_of_2(n)
    RPT = BN // (32 * nw)
    k = _m4_fused[(B,)](H, tau, sb, sr, sc, tau.stride(0),
                        N=n, NB=NB, NBP=NBP, BN=BN, RPT=RPT, BM=BM, BW=BW, BK=BK,
                        WK_WARPS=wk_warps, WK_REGS=wk_regs, OVERLAP=overlap, nw=nw, num_warps=nw)
    return H, tau, k


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    torch.manual_seed(0)
    print("\n=== M4 correctness (grid=1): SEQ (ov=0) vs WS overlap (ov=1) ===")
    for N in [256, 512]:
        A = torch.randn(1, N, N, device="cuda")
        Hg, taug = torch.geqrf(A[0])
        for ov in [0, 1]:
            try:
                H, tau, k = m4_qr(A, NB=64, overlap=ov); torch.cuda.synchronize()
            except Exception as e:
                import traceback; print(f"  N={N} ov={ov}: FAIL\n" + traceback.format_exc()[-1900:]); continue
            herr = (H[0] - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
            terr = (tau[0] - taug).abs().max().item() / (taug.abs().max().item() + 1e-30)
            print(f"  N={N:4d} ov={ov}: H={herr:.2e} tau={terr:.2e} regs={getattr(k,'n_regs','?')} "
                  f"sp={getattr(k,'n_spills','?')} {'OK' if herr<1e-4 and terr<1e-4 else 'BAD'}")

    print("\n=== M4 WHOLE n=512 b=640 vs V10 (9805us): SEQ vs WS (panel hidden?) ===")
    A = torch.randn(640, 512, 512, device="cuda")
    Hg, _ = torch.geqrf(A)
    for ov, wkw, wkr in [(0, 8, 160), (1, 8, 160), (1, 4, 200), (1, 8, 128)]:
        try:
            H, tau, k = m4_qr(A, NB=64, BM=64, BW=128, wk_warps=wkw, wk_regs=wkr, overlap=ov)
            torch.cuda.synchronize()
            herr = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
        except Exception as e:
            print(f"  ov={ov} wk={wkw}/{wkr}: ERR {repr(e)[:90]}"); continue
        t = time_fn(lambda ov=ov, wkw=wkw, wkr=wkr:
                    m4_qr(A, NB=64, BM=64, BW=128, wk_warps=wkw, wk_regs=wkr, overlap=ov))
        ts = t if isinstance(t, str) else f"{t:8.1f}us"
        rr = t / 9805.0 if not isinstance(t, str) else float('nan')
        tag = "SEQ" if ov == 0 else f"WS wk={wkw}/{wkr}"
        print(f"  {tag:13s}: {ts} {rr:.3f}x V10 regs={getattr(k,'n_regs','?')} "
              f"sp={getattr(k,'n_spills','?')} herr={herr:.0e}")
    print("\nM4 DONE.")


if __name__ == "__main__":
    main()
