"""M3a — tcgen05 TF32x3 GEMM in cute-DSL (3-limb emulated-fp32, the precision the n<=512 apply needs).

Extends cute_gemm_tf32_m1.py (1x tf32, rel 8.34e-4) to tf32x3: split each fp32 operand into hi+lo
tf32 limbs and accumulate 3 products  A_hi*B_hi + A_hi*B_lo + A_lo*B_hi  into one Float32 TMEM acc
(drop the lo*lo cross term). There is NO native 3xtf32 op (MmaTF32Op fixes inputs=TFloat32); we issue
3 MmaTF32Op passes per K-tile. Limbs are pre-split on the host (clean; in-kernel split is an M3+ opt).

Validates the rel err drops from ~8e-4 (1x tf32) toward ~1e-6/1e-7 (~22-bit) — the mixed@640 floor.
Run: modal run modal_cute_lab.py::run_candidate --script cute_gemm_tf32x3.py
(static-scan clean.)
"""
import argparse
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float32
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 8)
mma_tiler_mnk = (128, 256, 32)
threads_per_cta = 128
ab_stages = 2                      # 4 operands (a_hi,a_lo,b_hi,b_lo) -> 2 stages to fit SMEM
acc_stage = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(tiled_mma: cute.TiledMma,
           ta_hi: cute.CopyAtom, mAhi: cute.Tensor, ta_lo: cute.CopyAtom, mAlo: cute.Tensor,
           tb_hi: cute.CopyAtom, mBhi: cute.Tensor, tb_lo: cute.CopyAtom, mBlo: cute.Tensor,
           mC: cute.Tensor, a_smem_layout: cute.ComposedLayout, b_smem_layout: cute.ComposedLayout):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord = (bidx, bidy, None)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sAhi = smem.allocate_tensor(io_dtype, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner)
    sAlo = smem.allocate_tensor(io_dtype, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner)
    sBhi = smem.allocate_tensor(io_dtype, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner)
    sBlo = smem.allocate_tensor(io_dtype, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(ta_hi); cpasync.prefetch_descriptor(ta_lo)
        cpasync.prefetch_descriptor(tb_hi); cpasync.prefetch_descriptor(tb_lo)

    one_a = cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
    one_b = cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    tx = 2 * one_a + 2 * one_b
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=tx, barrier_storage=storage.ab_mbar_ptr.data_ptr()).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr()).make_participants()

    gAhi = cute.local_tile(mAhi, mma_tiler_mnk, mma_coord, proj=(1, None, 1))
    gAlo = cute.local_tile(mAlo, mma_tiler_mnk, mma_coord, proj=(1, None, 1))
    gBhi = cute.local_tile(mBhi, mma_tiler_mnk, mma_coord, proj=(None, 1, 1))
    gBlo = cute.local_tile(mBlo, mma_tiler_mnk, mma_coord, proj=(None, 1, 1))
    gC = cute.local_tile(mC, mma_tiler_mnk, mma_coord, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgAhi = thr_mma.partition_A(gAhi); tCgAlo = thr_mma.partition_A(gAlo)
    tCgBhi = thr_mma.partition_B(gBhi); tCgBlo = thr_mma.partition_B(gBlo)
    tCgC = thr_mma.partition_C(gC)
    tCrAhi = tiled_mma.make_fragment_A(sAhi); tCrAlo = tiled_mma.make_fragment_A(sAlo)
    tCrBhi = tiled_mma.make_fragment_B(sBhi); tCrBlo = tiled_mma.make_fragment_B(sBlo)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tAsAhi, tAgAhi = cute.nvgpu.cpasync.tma_partition(ta_hi, 0, cute.make_layout(1),
        cute.group_modes(sAhi, 0, 3), cute.group_modes(tCgAhi, 0, 3))
    tAsAlo, tAgAlo = cute.nvgpu.cpasync.tma_partition(ta_lo, 0, cute.make_layout(1),
        cute.group_modes(sAlo, 0, 3), cute.group_modes(tCgAlo, 0, 3))
    tBsBhi, tBgBhi = cute.nvgpu.cpasync.tma_partition(tb_hi, 0, cute.make_layout(1),
        cute.group_modes(sBhi, 0, 3), cute.group_modes(tCgBhi, 0, 3))
    tBsBlo, tBgBlo = cute.nvgpu.cpasync.tma_partition(tb_lo, 0, cute.make_layout(1),
        cute.group_modes(sBlo, 0, 3), cute.group_modes(tCgBlo, 0, 3))

    tmem.wait_for_alloc()
    tCtAcc = cute.make_tensor(tmem.retrieve_ptr(acc_dtype), tCtAcc.layout)

    subtile = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    num_k_tiles = cute.size(gAhi, mode=[2])
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for k in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 1):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(ta_hi, tAgAhi[(None, ab_empty.count)], tAsAhi[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
            cute.copy(ta_lo, tAgAlo[(None, ab_empty.count)], tAsAlo[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
            cute.copy(tb_hi, tBgBhi[(None, ab_empty.count)], tBsBhi[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
            cute.copy(tb_lo, tBgBlo[(None, ab_empty.count)], tBsBlo[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
            ab_full = ab_consumer.wait_and_advance()
            nkb = cute.size(tCrAhi, mode=[2])
            for kb in cutlass.range_constexpr(nkb):
                kc = (None, None, kb, ab_full.index)
                # 3 tf32 passes into the SAME acc: hi*hi, hi*lo, lo*hi
                cute.gemm(tiled_mma, tCtAcc, tCrAhi[kc], tCrBhi[kc], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc, tCrAhi[kc], tCrBlo[kc], tCtAcc)
                cute.gemm(tiled_mma, tCtAcc, tCrAlo[kc], tCrBhi[kc], tCtAcc)
            ab_full.release()
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem.retrieve_ptr(acc_dtype))


@cute.jit
def host_function(a_hi, a_lo, b_hi, b_lo, c):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_hi.element_type, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_hi.element_type, ab_stages)
    a1 = cute.select(a_sl, mode=[0, 1, 2]); b1 = cute.select(b_sl, mode=[0, 1, 2])
    g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    ta_hi, ma_hi = cute.nvgpu.make_tiled_tma_atom_A(g2s, a_hi, a1, mma_tiler_mnk, tiled_mma)
    ta_lo, ma_lo = cute.nvgpu.make_tiled_tma_atom_A(g2s, a_lo, a1, mma_tiler_mnk, tiled_mma)
    tb_hi, mb_hi = cute.nvgpu.make_tiled_tma_atom_B(g2s, b_hi, b1, mma_tiler_mnk, tiled_mma)
    tb_lo, mb_lo = cute.nvgpu.make_tiled_tma_atom_B(g2s, b_lo, b1, mma_tiler_mnk, tiled_mma)
    grid = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(tiled_mma, ta_hi, ma_hi, ta_lo, ma_lo, tb_hi, mb_hi, tb_lo, mb_lo, c, a_sl, b_sl).launch(
        grid=grid, block=(threads_per_cta, 1, 1))


def _split_tf32(x):
    import torch
    hi = (x.view(torch.int32) & (~0x1FFF)).view(torch.float32)   # truncate low 13 mantissa bits -> tf32
    lo = (x - hi)
    return hi.contiguous(), lo.contiguous()


def run(mnk, tol=1e-5):
    import torch
    m, n, k = mnk
    print(f"=== cute-DSL TF32x3 GEMM  mnk={mnk}  tol={tol} ===")
    torch.manual_seed(1)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(n, k, device="cuda", dtype=torch.float32)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    a_hi, a_lo = _split_tf32(a); b_hi, b_lo = _split_tf32(b)
    T = lambda x: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                   .mark_compact_shape_dynamic(mode=1, divisibility=(k if x.shape[1] == k else n)))
    host_function(T(a_hi), T(a_lo), T(b_hi), T(b_lo), T(c), no_cache=True)
    torch.cuda.synchronize()
    ref = torch.einsum("mk,nk->mn", a, b)
    rel = (c - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    print(f"  TF32x3 max rel err = {rel:.2e}   {'PASS ✅ (~22-bit)' if rel < tol else 'CHECK'}")
    print(f"  (1x tf32 baseline was ~8.3e-4; expect tf32x3 ~1e-6/1e-7)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mnk", default="512,512,256")
    a = p.parse_args(); run(tuple(int(x) for x in a.mnk.split(",")))
    print("DONE")
