"""Isolate the in-register tf32 hi/lo split: bitcast+AND variants, compare hi to torch truncation."""
import cutlass
import cutlass.cute as cute
import cutlass.cute.core as ccore
from cutlass.cute.runtime import from_dlpack

MASK = -8192   # 0xFFFFE000  clears low 13 mantissa bits


@cute.kernel
def k_bitcast_and(mx, mhi):                       # variant 1: x.bitcast(Int32); cutlass.and_
    bid, _, _ = cute.arch.block_idx()
    x = mx[bid]
    xi = x.bitcast(cutlass.Int32)
    hi_i = cutlass.and_(xi, cutlass.Int32(MASK))
    mhi[bid] = hi_i.bitcast(cutlass.Float32)


@cute.kernel
def k_core_and(mx, mhi):                           # variant 2: cute.core.and_
    bid, _, _ = cute.arch.block_idx()
    x = mx[bid]
    xi = x.bitcast(cutlass.Int32)
    hi_i = ccore.and_(xi, cutlass.Int32(MASK))
    mhi[bid] = hi_i.bitcast(cutlass.Float32)


@cute.kernel
def k_op_and(mx, mhi):                             # variant 3: python & operator
    bid, _, _ = cute.arch.block_idx()
    x = mx[bid]
    xi = x.bitcast(cutlass.Int32)
    hi_i = xi & cutlass.Int32(MASK)
    mhi[bid] = hi_i.bitcast(cutlass.Float32)


@cute.jit
def host(mx, mhi, which: cutlass.Constexpr):
    B = cute.size(mx, mode=[0])
    if which == 1:
        k_bitcast_and(mx, mhi).launch(grid=[B, 1, 1], block=[1, 1, 1])
    elif which == 2:
        k_core_and(mx, mhi).launch(grid=[B, 1, 1], block=[1, 1, 1])
    else:
        k_op_and(mx, mhi).launch(grid=[B, 1, 1], block=[1, 1, 1])


def run():
    import torch, traceback
    torch.manual_seed(0)
    x = torch.randn(64, device="cuda", dtype=torch.float32)
    ref = (x.view(torch.int32) & MASK).view(torch.float32)   # the intended tf32 truncation
    for which, name in [(1, "cutlass.and_"), (2, "cute.core.and_"), (3, "python &")]:
        hi = torch.zeros_like(x)
        try:
            host(from_dlpack(x).mark_layout_dynamic(), from_dlpack(hi).mark_layout_dynamic(),
                 which, no_cache=True)
            torch.cuda.synchronize()
            err = (hi - ref).abs().max().item()
            print(f"  [{name:16s}] hi-vs-truncation maxabs = {err:.3e}   {'OK' if err < 1e-30 else 'WRONG'}")
        except Exception:
            print(f"  [{name:16s}] EXC: " + traceback.format_exc().strip().split(chr(10))[-1][:120])
    print("DONE")


if __name__ == "__main__":
    run()
