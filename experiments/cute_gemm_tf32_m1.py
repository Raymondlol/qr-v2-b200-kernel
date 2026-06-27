"""M1' step 1 — single TF32 tcgen05 GEMM in cute-dsl (mutation of the shipped fp16_gemm_0.py).

Goal: get tcgen05 *TF32* working end-to-end on our B200 eval-replica image. This is the foundation
for the tf32x3 trailing apply (the perf-critical, decisive piece of the cute-dsl QR port). Computes
C[m,n] = A[m,k] @ B[n,k]^T (NT, K-major operands) in 1x TF32, validates vs torch fp32.

Changes vs fp16_gemm_0: MmaF16BF16Op->MmaTF32Op (fixes inputs=TFloat32, acc=Float32, NO io/acc dtype
args, inst-K MUST be 8); io_dtype fp16->fp32 (so SMEM/stage doubles -> shrink tiler-K + stages to fit
228KB); tolerance loosened to tf32's ~1e-2. Run: modal run modal_cute_lab.py::run_candidate --script cute_gemm_tf32_m1.py
(static-scan clean: no banned launch-plumbing substrings.)
"""
import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float32              # tf32 operands are fed from fp32 storage
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 8)     # TF32: instruction K MUST be 8
mma_tiler_mnk = (128, 256, 32)         # K-tile 32 (fp32 SMEM is 2x fp16 -> keep tile-K modest)
threads_per_cta = 128

ab_stages = 3                          # A:128*32*4 + B:256*32*4 = 48KB/stage * 3 = 144KB < 228KB
acc_stage = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(element_type=io_dtype, layout=a_smem_layout.outer,
                              byte_alignment=128, swizzle=a_smem_layout.inner)
    sB = smem.allocate_tensor(element_type=io_dtype, layout=b_smem_layout.outer,
                              byte_alignment=128, swizzle=b_smem_layout.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    num_tma_copy_bytes = cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])) \
        + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(tma_atom_a, tAgA[(None, ab_empty.count)], tAsA[(None, ab_empty.index)],
                      tma_bar_ptr=ab_empty.barrier)
            cute.copy(tma_atom_b, tBgB[(None, ab_empty.count)], tBsB[(None, ab_empty.index)],
                      tma_bar_ptr=ab_empty.barrier)
            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(tiled_mma, tCtAcc, tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
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
    tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaTF32Op(
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a.element_type, ab_stages)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b.element_type, ab_stages)
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    op2 = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op2, a, a_smem_layout_one_stage, mma_tiler_mnk, tiled_mma)
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op2, b, b_smem_layout_one_stage, mma_tiler_mnk, tiled_mma)

    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor, c,
           a_smem_layout, b_smem_layout).launch(
        grid=grid_shape, block=(threads_per_cta, 1, 1))


def run_tf32_gemm(mnk, tolerance=2e-2):
    import torch
    import cutlass.torch as cutlass_torch
    m, n, k = mnk
    print(f"=== cute-dsl TF32 GEMM  mnk={mnk}  tol={tolerance} ===")
    torch.manual_seed(1111)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(n, k, device="cuda", dtype=torch.float32)   # K-major operands (NT)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    a_t = (from_dlpack(a, assumed_align=16).mark_layout_dynamic(leading_dim=1)
           .mark_compact_shape_dynamic(mode=1, divisibility=k))
    b_t = (from_dlpack(b, assumed_align=16).mark_layout_dynamic(leading_dim=1)
           .mark_compact_shape_dynamic(mode=1, divisibility=k))
    c_t = (from_dlpack(c, assumed_align=16).mark_layout_dynamic(leading_dim=1)
           .mark_compact_shape_dynamic(mode=1, divisibility=n))
    host_function(a_t, b_t, c_t, no_cache=True)
    torch.cuda.synchronize()
    ref = torch.einsum("mk,nk->mn", a, b)
    rel = (c - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    print(f"  max rel err = {rel:.2e}   {'PASS ✅' if rel < tolerance else 'FAIL ❌'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mnk", type=str, default="512,512,256")
    a = p.parse_args()
    mnk = tuple(int(x) for x in a.mnk.split(","))
    run_tf32_gemm(mnk)
    print("DONE")
