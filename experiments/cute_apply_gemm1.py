"""M3c probe #2 — apply GEMM-1: W1 = V^T @ C via tcgen05 tf32x3, REAL apply orientation + in-reg split.

Validates the hardest correctness pieces of the M3c-0 apply, isolated from the K-loop ring (M=32 = ONE
k-tile, no overwrite race):
  * transposed orientation: A = V^T [IB,M] (K-major over rows r), B = C^T [BW,M] (K-major over r) -> W1[IB,BW]
  * unit-lower V mask applied on load: Vmask[r,i] = V[r,i] if r>i; 1 if r==i; 0 if r<i
  * IN-REGISTER tf32x3 split (Dekker, pure arithmetic — no bitcast). If the compiler reassociates
    `c-(c-x)`->x the split collapses to 1x tf32 and rel err ~8e-4 instead of ~1e-6 -> then switch to bitcast.
  * 3-pass tf32x3 (hi*hi + hi*lo + lo*hi) into a small [16,64]-live TMEM acc + read-back of the corner.
Reuses the VALIDATED hand-fill idiom from cute_apply_probe.py (sX[(row,ki),0,kb,0], k=kb*8+ki).

Run: modal run modal_cute_lab.py::run_candidate --script cute_apply_gemm1.py   (static-scan clean.)
"""
import argparse
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float32
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 8)
mma_tiler_mnk = (128, 256, 32)        # ONE k-tile when M=32
threads_per_cta = 128
ab_stages = 1
acc_stage = 1
IB = 16
BW = 64
MAGIC = 8193.0                        # 2^13 + 1  (Dekker split -> ~tf32 hi, residual lo)


@cute.struct
class SharedStorage:
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.jit
def _dek_hi(x: cutlass.Float32) -> cutlass.Float32:
    # tf32 hi = clear low 13 mantissa bits via bitcast f32<->i32 + AND 0xFFFFE000. Bit ops can't be
    # folded (the arithmetic Dekker `c-(c-x)` AND the f32->tf32->f32 round-trip both collapsed to x).
    xi = x.bitcast(cutlass.Int32)
    hi_i = xi & cutlass.Int32(-8192)                    # ~0x1FFF = 0xFFFFE000 (python & = bitwise; and_ is logical)
    return hi_i.bitcast(cutlass.Float32)


@cute.kernel
def kernel(tiled_mma: cute.TiledMma, mV: cute.Tensor, mC: cute.Tensor, mW: cute.Tensor,
           a_sl: cute.ComposedLayout, b_sl: cute.ComposedLayout,
           M: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sVh = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sVl = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sCh = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    sCl = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr()).make_participants()

    mma_coord = (0, 0, None)
    gW = cute.local_tile(mW, mma_tiler_mnk, mma_coord, proj=(1, 1, None))   # (128,256)
    thr_mma = tiled_mma.get_slice(0)
    tCgW = thr_mma.partition_C(gW)
    tVh = tiled_mma.make_fragment_A(sVh); tVl = tiled_mma.make_fragment_A(sVl)
    tCh = tiled_mma.make_fragment_B(sCh); tCl = tiled_mma.make_fragment_B(sCl)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gW_epi = cute.zipped_divide(tCgW, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gW_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrW = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    # ---- fill 4 swizzled smem tiles (single k-tile, M<=32 rows) ----
    # A = V^T : A[i, kk] = Vmask[r=kk, i],  i in [0,IB), kk in [0,32) (r=kk, pad kk>=M -> 0)
    i = tidx
    while i < IB * 32:
        ii = i // 32
        kk = i - ii * 32
        kb = kk // 8
        ki = kk - kb * 8
        r = kk
        v = cutlass.Float32(0.0)
        if r < M:
            vraw = mV[r, ii]
            v = vraw if r > ii else (cutlass.Float32(1.0) if r == ii else cutlass.Float32(0.0))
        hi = _dek_hi(v)
        sVh[(ii, ki), 0, kb, 0] = hi
        sVl[(ii, ki), 0, kb, 0] = v - hi
        i = i + threads_per_cta
    # B = C^T : B[w, kk] = C[r=kk, w],  w in [0,BW), kk in [0,32)
    i = tidx
    while i < BW * 32:
        ww = i // 32
        kk = i - ww * 32
        kb = kk // 8
        ki = kk - kb * 8
        r = kk
        cval = cutlass.Float32(0.0)
        if r < M:
            cval = mC[r, ww]
        hi = _dek_hi(cval)
        sCh[(ww, ki), 0, kb, 0] = hi
        sCl[(ww, ki), 0, kb, 0] = cval - hi
        i = i + threads_per_cta
    cute.arch.barrier()

    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        nkb = cute.size(tVh, mode=[2])
        for kb in cutlass.range_constexpr(nkb):
            kc = (None, None, kb, 0)
            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
            cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    for j in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, j], tCrAcc)
        tCrW.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrW, tDgC[None, None, j])
    acc_full.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(v: cute.Tensor, c: cute.Tensor, w: cute.Tensor, M: cutlass.Constexpr):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    kernel(tiled_mma, v, c, w, a_sl, b_sl, M).launch(grid=[1, 1, 1], block=(threads_per_cta, 1, 1))


def run(M=32, tol=1e-5):
    import torch
    print(f"=== probe#2 apply GEMM-1 W1=V^T C  tf32x3 in-reg split  M={M} IB={IB} BW={BW} ===")
    torch.manual_seed(3)
    V = torch.randn(M, IB, device="cuda", dtype=torch.float32)
    C = torch.randn(M, BW, device="cuda", dtype=torch.float32)
    W = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    T = lambda x, d: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                      .mark_compact_shape_dynamic(mode=1, divisibility=d))
    host_function(T(V, IB), T(C, BW), T(W, 256), M, no_cache=True)
    torch.cuda.synchronize()
    # reference: unit-lower mask V, then W1 = Vmask^T @ C
    rows = torch.arange(M, device="cuda").view(M, 1)
    cols = torch.arange(IB, device="cuda").view(1, IB)
    Vm = torch.where(rows > cols, V, torch.where(rows == cols, torch.ones_like(V), torch.zeros_like(V)))
    ref = Vm.t() @ C                       # [IB, BW]
    got = W[:IB, :BW]
    rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    tag = "PASS tf32x3 (~22-bit)" if rel < tol else ("1xtf32-only (split collapsed?)" if rel < 2e-3 else "FAIL")
    print(f"  max rel err = {rel:.2e}   -> {tag}")
    return rel


if __name__ == "__main__":
    import traceback
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=32)
    args = p.parse_args()
    try:
        run(args.M)
    except Exception:
        print("EXC\n" + traceback.format_exc()[-3000:])
    print("DONE")
