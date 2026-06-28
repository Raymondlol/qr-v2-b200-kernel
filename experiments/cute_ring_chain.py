"""M6 full-apply ring — dV = V(T^T(V^T C)) with G1 as the async warp-spec RING, G2/G3 simple.

Extends the validated G1 ring (cute_ring_g1.py, 1.58x) to the FULL 3-GEMM apply chain (cute_apply_chain.py):
G1 (the dominant K-loop) runs as the producer(warps0-2)‖MMA-consumer(warp3) mbarrier ring with NO per-tile
barrier; after it, a CTA barrier, then G2/G3 run the simple all-128 path (warp3 MMA). Proves the ring
scales to the whole apply before the per-matrix + panel-overlap integration. W1/W2 staged via gmem scratch.

Run: modal run modal_cute_lab.py::run_candidate_quick --script cute_ring_chain.py   (150s kill on hang).
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
IB = 16
BW = 64
BM = 32
NBP = 64
NPROD = 96


@cute.struct
class SharedStorage:
    full_mbar: cute.struct.MemRange[cutlass.Int64, ab_stages]
    empty_mbar: cute.struct.MemRange[cutlass.Int64, ab_stages]
    acc_done: cute.struct.MemRange[cutlass.Int64, 1]
    g23_full: cute.struct.MemRange[cutlass.Int64, 1]    # G2/G3: producer(all)->MMA-consumer ready
    g23_done: cute.struct.MemRange[cutlass.Int64, 1]    # G2/G3: UMMA done -> readback
    tmem_holding_buf: cutlass.Int32


@cute.jit
def _hi(x: cutlass.Float32) -> cutlass.Float32:
    return (x.bitcast(cutlass.Int32) & cutlass.Int32(-8192)).bitcast(cutlass.Float32)


@cute.kernel
def kernel(tiled_mma: cute.TiledMma, mV, mT, mC, mW1, mW2, mDV, a_sl, b_sl, M: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(SharedStorage)
    sVh = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sVl = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sCh = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    sCl = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    full0 = st.full_mbar.data_ptr(); empty0 = st.empty_mbar.data_ptr(); accd = st.acc_done.data_ptr()
    g23f = st.g23_full.data_ptr(); g23d = st.g23_done.data_ptr()

    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(st.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)

    if tidx == 0:
        cute.arch.mbarrier_init(full0 + 0, NPROD); cute.arch.mbarrier_init(full0 + 1, NPROD)
        cute.arch.mbarrier_init(empty0 + 0, 1); cute.arch.mbarrier_init(empty0 + 1, 1)
        cute.arch.mbarrier_init(accd + 0, 1)
        cute.arch.mbarrier_init(g23f + 0, NPROD); cute.arch.mbarrier_init(g23d + 0, 1)
        cute.arch.mbarrier_arrive(empty0 + 0); cute.arch.mbarrier_arrive(empty0 + 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    thr_mma = tiled_mma.get_slice(0)
    tVh = tiled_mma.make_fragment_A(sVh); tVl = tiled_mma.make_fragment_A(sVl)
    tCh = tiled_mma.make_fragment_B(sCh); tCl = tiled_mma.make_fragment_B(sCl)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tmem.wait_for_alloc()
    tCtAcc = cute.make_tensor(tmem.retrieve_ptr(acc_dtype), tCtAcc.layout)
    subtile = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    nkb = cute.size(tVh, mode=[2])

    def dst_part(mOut):
        g = cute.local_tile(mOut, mma_tiler_mnk, (0, 0, None), proj=(1, 1, None))
        return tmem_thr_copy.partition_D(cute.zipped_divide(thr_mma.partition_C(g), epi_tiler))
    tDgW1 = dst_part(mW1); tDgW2 = dst_part(mW2); tDgDV = dst_part(mDV)
    rb = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, acc_dtype)
    rbio = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, io_dtype)
    num_kt = (M + BM - 1) // BM

    # ============ G1 RING: W1 = V^T C (producer warps0-2 ‖ MMA consumer warp3) ============
    if warp < 3:
        kt = 0
        while kt < num_kt:
            s = kt % 2; ph = (kt // 2) % 2
            cute.arch.mbarrier_wait(empty0 + s, ph)
            r0 = kt * BM
            i = tidx
            while i < IB * 32:
                ii = i // 32; kk = i - ii * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
                v = cutlass.Float32(0.0)
                if r < M:
                    vraw = mV[r, ii]
                    v = vraw if r > ii else (cutlass.Float32(1.0) if r == ii else cutlass.Float32(0.0))
                h = _hi(v); sVh[(ii, ki), 0, kb, s] = h; sVl[(ii, ki), 0, kb, s] = v - h
                i = i + NPROD
            i = tidx
            while i < BW * 32:
                ww = i // 32; kk = i - ww * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
                cv = mC[r, ww] if r < M else cutlass.Float32(0.0)
                h = _hi(cv); sCh[(ww, ki), 0, kb, s] = h; sCl[(ww, ki), 0, kb, s] = cv - h
                i = i + NPROD
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(full0 + s)
            kt = kt + 1
    if warp == 3:
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        kt = 0
        while kt < num_kt:
            s = kt % 2; ph = (kt // 2) % 2
            cute.arch.mbarrier_wait(full0 + s, ph)
            for kb in cutlass.range_constexpr(nkb):
                kc = (None, None, kb, s)
                cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
                cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
            if lane == 0:
                tcgen05.commit(empty0 + s)
            kt = kt + 1
        if lane == 0:
            tcgen05.commit(accd + 0)
    cute.arch.mbarrier_wait(accd + 0, 0)
    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
    rbio.store(rb.load().to(io_dtype))
    cute.autovec_copy(rbio, tDgW1[None, None, 0])     # W1 -> gmem
    cute.arch.barrier()                               # end of G1 ring; reset to simple path

    # ============ G2: W2 = T^T W1 (simple: warps0-2 fill stage0 ‖ warp3 MMA) ============
    if warp < 3:
        i = tidx
        while i < IB * 32:
            ii = i // 32; k = i - ii * 32; kb = k // 8; ki = k - kb * 8
            a = mT[k, ii] if k < IB else cutlass.Float32(0.0)
            h = _hi(a); sVh[(ii, ki), 0, kb, 0] = h; sVl[(ii, ki), 0, kb, 0] = a - h
            i = i + NPROD
        i = tidx
        while i < BW * 32:
            ww = i // 32; k = i - ww * 32; kb = k // 8; ki = k - kb * 8
            b = mW1[k, ww] if k < IB else cutlass.Float32(0.0)
            h = _hi(b); sCh[(ww, ki), 0, kb, 0] = h; sCl[(ww, ki), 0, kb, 0] = b - h
            i = i + NPROD
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(g23f + 0)
    if warp == 3:
        cute.arch.mbarrier_wait(g23f + 0, 0)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkb):
            kc = (None, None, kb, 0)
            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
            cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
        if lane == 0:
            tcgen05.commit(g23d + 0)
    cute.arch.mbarrier_wait(g23d + 0, 0)
    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
    rbio.store(rb.load().to(io_dtype))
    cute.autovec_copy(rbio, tDgW2[None, None, 0])     # W2 -> gmem
    cute.arch.barrier()

    # ============ G3: dV = V W2 (single output tile rows[0,M<=NBP)) ============
    if warp < 3:
        i = tidx
        while i < NBP * 32:
            rr = i // 32; k = i - rr * 32; kb = k // 8; ki = k - kb * 8
            a = cutlass.Float32(0.0)
            if k < IB and rr < M:
                a = mV[rr, k] if rr > k else (cutlass.Float32(1.0) if rr == k else cutlass.Float32(0.0))
            h = _hi(a); sVh[(rr, ki), 0, kb, 0] = h; sVl[(rr, ki), 0, kb, 0] = a - h
            i = i + NPROD
        i = tidx
        while i < BW * 32:
            ww = i // 32; k = i - ww * 32; kb = k // 8; ki = k - kb * 8
            b = mW2[k, ww] if k < IB else cutlass.Float32(0.0)
            h = _hi(b); sCh[(ww, ki), 0, kb, 0] = h; sCl[(ww, ki), 0, kb, 0] = b - h
            i = i + NPROD
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(g23f + 0)
    if warp == 3:
        cute.arch.mbarrier_wait(g23f + 0, 1)          # 2nd use of g23f -> phase 1
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkb):
            kc = (None, None, kb, 0)
            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
            cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
        if lane == 0:
            tcgen05.commit(g23d + 0)
    cute.arch.mbarrier_wait(g23d + 0, 1)              # 2nd use -> phase 1
    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
    rbio.store(rb.load().to(io_dtype))
    cute.autovec_copy(rbio, tDgDV[None, None, 0])     # dV -> gmem
    pipeline.sync(barrier_id=1)
    tmem.free(tmem.retrieve_ptr(acc_dtype))


@cute.jit
def host_function(v, t, c, w1, w2, dv, M: cutlass.Constexpr):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    kernel(tiled_mma, v, t, c, w1, w2, dv, a_sl, b_sl, M).launch(grid=[1, 1, 1], block=(threads_per_cta, 1, 1))


def run(M=64, tol=1e-4):
    import torch
    print(f"=== M6 full-apply ring (G1-ring + G2/G3) dV=V(T^T(V^T C))  M={M} ===")
    torch.manual_seed(5)
    V = torch.randn(M, IB, device="cuda", dtype=torch.float32)
    T = torch.triu(torch.randn(IB, IB, device="cuda", dtype=torch.float32))
    C = torch.randn(M, BW, device="cuda", dtype=torch.float32)
    W1 = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    W2 = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    DV = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    Tn = lambda x, d: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                       .mark_compact_shape_dynamic(mode=1, divisibility=d))
    host_function(Tn(V, IB), Tn(T, IB), Tn(C, BW), Tn(W1, 256), Tn(W2, 256), Tn(DV, 256), M, no_cache=True)
    torch.cuda.synchronize()
    rows = torch.arange(M, device="cuda").view(M, 1); cols = torch.arange(IB, device="cuda").view(1, IB)
    Vm = torch.where(rows > cols, V, torch.where(rows == cols, torch.ones_like(V), torch.zeros_like(V)))
    ref = Vm @ (T.t() @ (Vm.t() @ C))
    rel = (DV[:M, :BW] - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    print(f"  M={M:4d}: dV rel={rel:.2e}  -> {'PASS (full ring chain!)' if rel < tol else 'FAIL'}")
    return rel


if __name__ == "__main__":
    import traceback
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=64)
    a = p.parse_args()
    try:
        run(a.M)
    except Exception:
        print("EXC\n" + traceback.format_exc()[-2800:])
    print("DONE")
