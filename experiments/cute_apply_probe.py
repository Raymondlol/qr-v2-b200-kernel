"""M3c probe #1 — HAND-FILLED swizzled SMEM → tcgen05 1x TF32 GEMM (NO TMA).

THE decisive de-risk for M3c-0: the build guide wants the per-matrix apply to stage V/C operands into
SMEM by DIRECT manual loads (each thread writes sA[m,k]=value, letting the composed swizzled layout
apply the swizzle), NOT TMA — because the apply operands need arbitrary K-major orientations we control.
This probe is the smallest mutation of the validated cute_gemm_tf32_m1.py (TMA, rel 8.34e-4) that
answers ONLY that question: replace the TMA input ring with a manual gmem->smem fill, keep the MMA loop
+ TMEM read-back identical. K=32 = ONE k-tile => no ring, no overwrite race. If C matches torch, then
"index a swizzled SMEM operand by logical [m,k] coords and feed make_fragment_A" is VALIDATED and the
whole apply can be built on direct loads.

Run: modal run modal_cute_lab.py::run_candidate --script cute_apply_probe.py   (static-scan clean.)
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
mma_inst_shape_mnk = (128, 256, 8)     # TF32 inst-K MUST be 8
mma_tiler_mnk = (128, 256, 32)         # ONE k-tile (K=32) => no input ring needed
threads_per_cta = 128
ab_stages = 1
acc_stage = 1


@cute.struct
class SharedStorage:
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(tiled_mma: cute.TiledMma, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
           a_smem_layout: cute.ComposedLayout, b_smem_layout: cute.ComposedLayout):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord = (bidx, bidy, None)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(io_dtype, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner)
    sB = smem.allocate_tensor(io_dtype, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr()).make_participants()

    gC = cute.local_tile(mC, mma_tiler_mnk, mma_coord, proj=(1, 1, None))   # (128, 256)
    thr_mma = tiled_mma.get_slice(0)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

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

    # ----- MANUAL gmem->smem fill (the thing under test), single k-tile kt=0 -----
    # SMEM operand layout (from probe#1 error) = ((M,8),1,4,1): K=32 chunked as Kblock=4 x Kinner=8.
    # Index hierarchically: sA[(m, ki), 0, kb, 0] with k = kb*8 + ki. Read inputs from the 2D mA/mB.
    TM = 128
    TK = 32
    TN = 256
    i = tidx
    while i < TM * TK:
        m = i // TK
        k = i - m * TK
        kb = k // 8
        ki = k - kb * 8
        sA[(m, ki), 0, kb, 0] = mA[m, k]
        i = i + threads_per_cta
    i = tidx
    while i < TN * TK:
        nn = i // TK
        k = i - nn * TK
        kb = k // 8
        ki = k - kb * 8
        sB[(nn, ki), 0, kb, 0] = mB[nn, k]
        i = i + threads_per_cta
    cute.arch.barrier()              # fill visible to the MMA-issuing warp

    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for kb in cutlass.range_constexpr(num_k_blocks):
            kc = (None, None, kb, 0)
            cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB[kc], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    for j in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, j], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, j])
    acc_full.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a.element_type, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b.element_type, ab_stages)
    print("A smem layout outer =", a_sl.outer)
    print("B smem layout outer =", b_sl.outer)
    grid = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(tiled_mma, a, b, c, a_sl, b_sl).launch(grid=grid, block=(threads_per_cta, 1, 1))


def run(mnk=(128, 256, 32), tol=2e-2):
    import torch
    m, n, k = mnk
    print(f"=== probe#1 HAND-FILL swizzled SMEM -> tcgen05 1x TF32  mnk={mnk}  tol={tol} ===")
    torch.manual_seed(7)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(n, k, device="cuda", dtype=torch.float32)   # NT: C[m,n]=A[m,k]B[n,k]
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    T = lambda x, d: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                      .mark_compact_shape_dynamic(mode=1, divisibility=d))
    host_function(T(a, k), T(b, k), T(c, n), no_cache=True)
    torch.cuda.synchronize()
    ref = torch.einsum("mk,nk->mn", a, b)
    rel = (c - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    ok = rel < tol
    print(f"  max rel err = {rel:.2e}   {'PASS (hand-fill swizzled SMEM WORKS)' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import traceback
    p = argparse.ArgumentParser(); p.add_argument("--mnk", default="128,256,32")
    args = p.parse_args()
    try:
        run(tuple(int(x) for x in args.mnk.split(",")))
    except Exception:
        print("EXC\n" + traceback.format_exc()[-3000:])
    print("DONE")
