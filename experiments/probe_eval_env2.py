"""DECISIVE K0 probe v2 — make PASS/FAIL ENCODE whether cute-dsl is deployable on the qr_v2 board.

probe v1 printed the env but the board shows Debug Info = N/A on PASS (stdout/stderr only surface on
error), so v1 was inconclusive. v2 gates correctness on the FULL cute-dsl path: lazily (on the first
custom_kernel call, when a GPU is guaranteed present) it imports cutlass.cute and JIT-compiles + launches
the EXACT add-one kernel already validated maxerr=0 on our eval-replica image. If that works, every call
returns torch.geqrf (PASS, ~131000us, 22/22). If cute-dsl is absent/broken on the board, custom_kernel
RAISES — the submission FAILS and the reason lands in Debug Info (which populates on errors).

  => PASS  ⟺  cute-dsl import+JIT+launch+correct works on the qr_v2 board (K0 confirmed, deployable).
  => FAIL  ⟺  not deployable; the exception message tells us why.

Zero risk to the leaderboard standing (this is a throwaway probe). No banned substrings.
"""
import torch

_CHECKED = False
_OK = False
_MSG = ""


def _check_cutedsl():
    global _CHECKED, _OK, _MSG
    if _CHECKED:
        return
    _CHECKED = True
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        @cute.kernel
        def _add_one_k(gX: cute.Tensor, gY: cute.Tensor, cC: cute.Tensor, shape: cute.Shape,
                       thr_layout: cute.Layout, val_layout: cute.Layout):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            blk = ((None, None), bidx)
            blkX = gX[blk]; blkY = gY[blk]; blkCrd = cC[blk]
            ld = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
            st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type)
            tcX = cute.make_tiled_copy_tv(ld, thr_layout, val_layout)
            tcY = cute.make_tiled_copy_tv(st, thr_layout, val_layout)
            thrX = tcX.get_slice(tidx).partition_S(blkX)
            thrY = tcY.get_slice(tidx).partition_S(blkY)
            thrCrd = tcY.get_slice(tidx).partition_S(blkCrd)
            frgX = cute.make_fragment_like(thrX)
            frgY = cute.make_fragment_like(thrY)
            frgP = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
            for i in range(0, cute.size(frgP), 1):
                frgP[i] = cute.elem_less(thrCrd[i], shape)
            cute.copy(ld, thrX, frgX, pred=frgP)
            frgY.store(frgX.load() + 1.0)
            cute.copy(st, frgY, thrY, pred=frgP)

        @cute.jit
        def _add_one_host(mX, mY):
            thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
            val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
            tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
            gX = cute.zipped_divide(mX, tiler_mn)
            gY = cute.zipped_divide(mY, tiler_mn)
            idC = cute.make_identity_tensor(mY.shape)
            cC = cute.zipped_divide(idC, tiler=tiler_mn)
            _add_one_k(gX, gY, cC, mY.shape, thr_layout, val_layout).launch(
                grid=[cute.size(gY, mode=[1]), 1, 1],
                block=[cute.size(tv_layout, mode=[0]), 1, 1])

        x = torch.randn(256, 128, device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)
        xt = from_dlpack(x).mark_layout_dynamic()
        yt = from_dlpack(y).mark_layout_dynamic()
        cute.compile(_add_one_host, xt, yt)(xt, yt)
        torch.cuda.synchronize()
        _OK = (y - (x + 1.0)).abs().max().item() < 1e-5
        _MSG = f"cute_v{getattr(cutlass, '__version__', '?')}_ok={_OK}"
    except Exception as e:
        _MSG = "EXC:" + repr(e)[:300]
        _OK = False


def custom_kernel(data):
    _check_cutedsl()
    if not _OK:
        raise RuntimeError("CUTEDSL_PROBE_FAIL :: " + _MSG)
    return torch.geqrf(data)


if __name__ == "__main__":
    # local self-test on the eval-replica image: confirm the PASS path works end-to-end here, so a
    # board FAIL would mean a real board issue (not a probe bug).
    A = torch.randn(8, 64, 64, device="cuda", dtype=torch.float32)
    try:
        H, tau = custom_kernel(A)
        print(f"PROBE v2 self-test: PASS-path OK :: {_MSG} :: H{tuple(H.shape)} tau{tuple(tau.shape)}")
    except Exception as e:
        print(f"PROBE v2 self-test: RAISED :: {e}")
