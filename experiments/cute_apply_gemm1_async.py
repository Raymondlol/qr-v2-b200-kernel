"""M6 slice — G1 (W1=V^T C) K-loop via an ASYNC WARP-SPEC RING (PipelineAsyncUmma), NO per-tile CTA barrier.

The decisive M6 proof-of-concept: replace the per-tile `cute.arch.barrier()` + drain (the wall, per the
ablation) with a 2-stage smem ring where PRODUCER warps fill stage s+1 while the MMA (CONSUMER) warp
issues tcgen05 on stage s, synced by mbarriers (PipelineAsyncUmma = manual/async-fill producer + UMMA
consumer with UMMA-completion-tracked release). Accumulate across K-tiles in TMEM (ACC=True), read back
ONCE after the loop. If correct + faster than the barrier K-loop (cute_apply_gemm1_kloop), the per-tile
barrier wall is broken -> the M6 engine path is validated.

  WARPS 0-2 (96 thr) = PRODUCER: per K-tile, fill sV/sC[stage] (masked V^T + C^T, tf32x3-split).
  WARP 3   (32 thr) = MMA CONSUMER: per K-tile, 3-pass tcgen05 into TMEM acc.
  after the ring: all 128 threads read back the acc -> W1.

Run: modal run modal_cute_lab.py::run_candidate --script cute_apply_gemm1_async.py   (static-scan clean.)
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
mma_tiler_mnk = (128, 256, 32)
threads_per_cta = 128
ab_stages = 2
acc_stage = 1
IB = 16
BW = 64
BM = 32
MMA_WARP = 3                 # warp 3 = MMA consumer; warps 0-2 = producer (fill)


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.jit
def _hi(x: cutlass.Float32) -> cutlass.Float32:
    return (x.bitcast(cutlass.Int32) & cutlass.Int32(-8192)).bitcast(cutlass.Float32)


@cute.kernel
def kernel(tiled_mma: cute.TiledMma, mV, mC, mW, a_sl, b_sl, M: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sVh = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sVl = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sCh = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    sCl = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)

    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)

    # ab ring: producer = warps 0-2 (96 threads), consumer = warp 3 (32 threads, the MMA)
    ab_producer, ab_consumer = pipeline.PipelineAsyncUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 96),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
        barrier_storage=storage.ab_mbar_ptr.data_ptr()).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr()).make_participants()

    thr_mma = tiled_mma.get_slice(0)
    gW = cute.local_tile(mW, mma_tiler_mnk, (0, 0, None), proj=(1, 1, None))
    tCgW = thr_mma.partition_C(gW)
    tVh = tiled_mma.make_fragment_A(sVh); tVl = tiled_mma.make_fragment_A(sVl)
    tCh = tiled_mma.make_fragment_B(sCh); tCl = tiled_mma.make_fragment_B(sCl)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tmem.wait_for_alloc()
    tCtAcc = cute.make_tensor(tmem.retrieve_ptr(acc_dtype), tCtAcc.layout)

    subtile = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gW_epi = cute.zipped_divide(tCgW, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gW_epi)
    nkb = cute.size(tVh, mode=[2])
    num_kt = (M + BM - 1) // BM

    # ---------------- PRODUCER (warps 0-2): fill the ring ----------------
    if warp < MMA_WARP:
        kt = 0
        while kt < num_kt:
            ab_empty = ab_producer.acquire_and_advance()
            st = ab_empty.index
            r0 = kt * BM
            i = tidx
            while i < IB * 32:
                ii = i // 32; kk = i - ii * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
                v = cutlass.Float32(0.0)
                if r < M:
                    vraw = mV[r, ii]
                    v = vraw if r > ii else (cutlass.Float32(1.0) if r == ii else cutlass.Float32(0.0))
                h = _hi(v); sVh[(ii, ki), 0, kb, st] = h; sVl[(ii, ki), 0, kb, st] = v - h
                i = i + 96
            i = tidx
            while i < BW * 32:
                ww = i // 32; kk = i - ww * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
                cv = mC[r, ww] if r < M else cutlass.Float32(0.0)
                h = _hi(cv); sCh[(ww, ki), 0, kb, st] = h; sCl[(ww, ki), 0, kb, st] = cv - h
                i = i + 96
            cute.arch.fence_view_async_shared()           # manual st.shared visible before commit
            ab_empty.commit()
            kt = kt + 1

    # ---------------- MMA CONSUMER (warp 3): drain the ring into TMEM acc ----------------
    if warp == MMA_WARP:
        acc_empty = acc_producer.acquire_and_advance()    # the acc producer is the MMA warp ONLY
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        kt = 0
        while kt < num_kt:
            ab_full = ab_consumer.wait_and_advance()
            st = ab_full.index
            for kb in cutlass.range_constexpr(nkb):
                kc = (None, None, kb, st)
                cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
                cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
            ab_full.release()
            kt = kt + 1
        acc_empty.commit()                                # signal acc ready to all threads

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    rb = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    rbio = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)
    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
    rbio.store(rb.load().to(io_dtype))
    cute.autovec_copy(rbio, tDgC[None, None, 0])
    acc_full.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem.retrieve_ptr(acc_dtype))


@cute.jit
def host_function(v, c, w, M: cutlass.Constexpr):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    kernel(tiled_mma, v, c, w, a_sl, b_sl, M).launch(grid=[1, 1, 1], block=(threads_per_cta, 1, 1))


def run(M, tol=1e-5):
    import torch
    print(f"=== M6 slice: G1 async-ring (PipelineAsyncUmma)  M={M}  ({(M+BM-1)//BM} k-tiles) ===")
    torch.manual_seed(3)
    V = torch.randn(M, IB, device="cuda", dtype=torch.float32)
    C = torch.randn(M, BW, device="cuda", dtype=torch.float32)
    W = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    T = lambda x, d: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                      .mark_compact_shape_dynamic(mode=1, divisibility=d))
    host_function(T(V, IB), T(C, BW), T(W, 256), M, no_cache=True)
    torch.cuda.synchronize()
    rows = torch.arange(M, device="cuda").view(M, 1); cols = torch.arange(IB, device="cuda").view(1, IB)
    Vm = torch.where(rows > cols, V, torch.where(rows == cols, torch.ones_like(V), torch.zeros_like(V)))
    ref = Vm.t() @ C
    rel = (W[:IB, :BW] - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    print(f"  M={M:4d}: rel={rel:.2e}  -> {'PASS (async ring works!)' if rel < tol else 'FAIL'}")
    return rel


if __name__ == "__main__":
    import traceback
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=0)
    a = p.parse_args()
    for M in ([a.M] if a.M else [32, 128, 512]):
        try:
            run(M)
        except Exception:
            print(f"  M={M}: EXC\n" + traceback.format_exc()[-2600:])
    print("DONE")
