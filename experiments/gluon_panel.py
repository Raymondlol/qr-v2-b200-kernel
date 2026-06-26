"""PHASE 1.5 step 1 — port submission.py::_panel_kernel (the latency-bound Householder
reflector reduction, pure gl.sum + FMA, no dot) to Gluon @gluon.jit, and verify it is
bit-faithful to the torch reference _panel_factor. This is the panel that must run as the
default partition co-resident with the tcgen05 trailing worker. No banned substrings.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import default_blocked_layout


# ----- torch reference (mirrors submission.py::_panel_factor on a [B,M,b] tile) -----
def panel_ref(P):
    P = P.clone()
    B, m, b = P.shape
    tau = torch.zeros(B, b, dtype=P.dtype, device=P.device)
    for j in range(b):
        alpha = P[:, j, j]
        x = P[:, j + 1:, j]
        xnorm = torch.linalg.vector_norm(x, dim=1)
        normfull = torch.sqrt(alpha * alpha + xnorm * xnorm)
        sign = torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        beta = -sign * normfull
        mask = xnorm > 0
        safe_denom = torch.where(mask, alpha - beta, torch.ones_like(alpha))
        safe_beta = torch.where(mask, beta, torch.ones_like(beta))
        tau_j = torch.where(mask, (beta - alpha) / safe_beta, torch.zeros_like(alpha))
        vtail = torch.where(mask.unsqueeze(1), x / safe_denom.unsqueeze(1), torch.zeros_like(x))
        P[:, j + 1:, j] = vtail
        P[:, j, j] = torch.where(mask, beta, alpha)
        tau[:, j] = tau_j
        if j < b - 1:
            ones = torch.ones(B, 1, dtype=P.dtype, device=P.device)
            V = torch.cat([ones, vtail], dim=1)
            sub = P[:, j:, j + 1:]
            w = torch.einsum('bm,bmt->bt', V, sub)
            sub.sub_(tau_j.view(B, 1, 1) * torch.einsum('bm,bt->bmt', V, w))
    return P, tau


# ----- Gluon port of _panel_kernel -----
@gluon.jit
def _panel_gl(Hptr, tauptr, SB, SR, STAU, M,
              BN: gl.constexpr, BCOLS: gl.constexpr):
    bid = gl.program_id(0)
    t_blk: gl.constexpr = default_blocked_layout([BN, BCOLS], gl.num_warps())
    rl: gl.constexpr = gl.SliceLayout(1, t_blk)   # row-vector layout (length BN)
    cl: gl.constexpr = gl.SliceLayout(0, t_blk)   # col-vector layout (length BCOLS)
    rar = gl.arange(0, BN, layout=rl)
    car = gl.arange(0, BCOLS, layout=cl)
    base = Hptr + bid * SB
    ptr = base + rar[:, None] * SR + car[None, :]
    tmask = (rar[:, None] < M)
    tile = gl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        colj = gl.sum(gl.where(car[None, :] == j, tile, 0.0), axis=1)          # [BN]
        alpha = gl.sum(gl.where(rar == j, colj, 0.0))                          # scalar
        xnorm2 = gl.sum(gl.where(rar > j, colj * colj, 0.0))                   # scalar
        normfull = gl.sqrt(alpha * alpha + xnorm2)
        sgn = gl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -sgn * normfull
        need = xnorm2 > 0.0
        scale = gl.where(need, 1.0 / (alpha - beta), 0.0)
        tau_j = gl.where(need, (beta - alpha) / beta, 0.0)
        v = gl.where(rar > j, colj * scale, 0.0)
        v = gl.where(rar == j, 1.0, v)                                         # [BN]
        w = gl.sum(v[:, None] * tile, axis=0)                                  # [BCOLS]
        upd = tile - tau_j * (v[:, None] * w[None, :])
        tile = gl.where(car[None, :] > j, upd, tile)
        diagval = gl.where(need, beta, alpha)
        newcol = gl.where(rar > j, colj * scale, colj)
        newcol = gl.where(rar == j, diagval, newcol)
        tile = gl.where(car[None, :] == j, newcol[:, None], tile)
        gl.store(tauptr + bid * STAU + j, tau_j)
    gl.store(ptr, tile, mask=tmask)


def run_gl(P, nw=8):
    B, M, b = P.shape
    BN = triton.next_power_of_2(M)
    BCOLS = triton.next_power_of_2(b)
    assert BN == M and BCOLS == b, "test uses power-of-2 M,b (no padding) for a clean ref match"
    H = P.clone().contiguous()
    tau = torch.zeros(B, b, dtype=P.dtype, device=P.device)
    sb, sr, _ = H.stride()
    _panel_gl[(B,)](H, tau, sb, sr, tau.stride(0), M, BN=BN, BCOLS=BCOLS, num_warps=nw)
    return H, tau


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    torch.manual_seed(0)
    for B, M, b in [(4, 64, 64), (4, 128, 64), (8, 512, 64), (8, 256, 32)]:
        P = torch.randn(B, M, b, device="cuda")
        try:
            Hg, taug = run_gl(P)
        except Exception as e:
            import traceback; print(f"  B={B} M={M} b={b}: COMPILE/RUN FAIL"); traceback.print_exc(); return
        Hr, taur = panel_ref(P)
        torch.cuda.synchronize()
        herr = (Hg - Hr).abs().max().item() / (Hr.abs().max().item() + 1e-9)
        terr = (taug - taur).abs().max().item() / (taur.abs().max().item() + 1e-9)
        flag = "OK" if (herr < 1e-4 and terr < 1e-4) else "MISMATCH"
        print(f"  B={B} M={M:4d} b={b:3d}: H relerr={herr:.2e}  tau relerr={terr:.2e}  {flag}")
    print("\nGLUON PANEL PORT DONE")


if __name__ == "__main__":
    main()
