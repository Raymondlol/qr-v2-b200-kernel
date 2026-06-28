"""M6 ring (HAND-ROLLED mbarriers) — G1 W1=V^T C K-loop, producer warps ‖ MMA consumer, NO per-tile CTA barrier.

The deadlock fix: cute-DSL's Umma pipelines arrive the full-barrier via the async-copy instruction
(TMA/cp.async), which our MANUAL st.shared fill can't trigger. So we hand-roll the ring with the low-level
mbarrier API + the UMMA-completion primitive `tcgen05.commit(empty[s])` (the UMMA arrives on empty[s] when
it finishes reading smem[s]). 2-stage smem ring:
  PRODUCER (warps 0-2, 96 thr): per K-tile: wait empty[s]; fill smem[s] (masked Vt+Ct, tf32x3-split);
                                fence_view_async_shared; arrive full[s].
  CONSUMER (warp 3, MMA):       per K-tile: wait full[s]; 3-pass tcgen05 into TMEM (ACC accumulate);
                                tcgen05.commit(empty[s])  <- UMMA-done -> stage free.
  after the loop: consumer tcgen05.commit(acc_done); all 128 wait; readback W1.
Run: modal run modal_cute_lab.py::run_candidate_quick --script cute_ring_g1.py   (150s kill if deadlock).
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
NPROD = 96               # warps 0-2 produce; warp 3 consumes


@cute.struct
class SharedStorage:
    full_mbar: cute.struct.MemRange[cutlass.Int64, ab_stages]   # producer -> consumer (data ready)
    empty_mbar: cute.struct.MemRange[cutlass.Int64, ab_stages]  # consumer(UMMA) -> producer (stage free)
    acc_done: cute.struct.MemRange[cutlass.Int64, 1]            # last UMMA done -> readback
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
    full0 = storage.full_mbar.data_ptr()
    empty0 = storage.empty_mbar.data_ptr()
    accd = storage.acc_done.data_ptr()

    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)

    if tidx == 0:
        cute.arch.mbarrier_init(full0 + 0, NPROD); cute.arch.mbarrier_init(full0 + 1, NPROD)
        cute.arch.mbarrier_init(empty0 + 0, 1); cute.arch.mbarrier_init(empty0 + 1, 1)
        cute.arch.mbarrier_init(accd + 0, 1)
        cute.arch.mbarrier_arrive(empty0 + 0)        # prime: both stages start free
        cute.arch.mbarrier_arrive(empty0 + 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()                              # one-time init barrier

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

    # ---------------- PRODUCER (warps 0-2) ----------------
    if warp < 3:
        kt = 0
        while kt < num_kt:
            s = kt % 2
            ph = (kt // 2) % 2
            cute.arch.mbarrier_wait(empty0 + s, ph)        # wait stage s free
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
            cute.arch.fence_view_async_shared()            # st.shared visible before arrive
            cute.arch.mbarrier_arrive(full0 + s)           # each of 96 -> count NPROD
            kt = kt + 1

    # ---------------- CONSUMER (warp 3, MMA) ----------------
    if warp == 3:
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        kt = 0
        while kt < num_kt:
            s = kt % 2
            ph = (kt // 2) % 2
            cute.arch.mbarrier_wait(full0 + s, ph)         # wait stage s data ready
            for kb in cutlass.range_constexpr(nkb):
                kc = (None, None, kb, s)
                cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
                cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
            if cute.arch.lane_idx() == 0:                  # elect-one: 1 UMMA commit per stage (count=1)
                tcgen05.commit(empty0 + s)                 # UMMA done reading smem[s] -> stage free
            kt = kt + 1
        if cute.arch.lane_idx() == 0:
            tcgen05.commit(accd + 0)                       # last UMMA done -> readback

    cute.arch.mbarrier_wait(accd + 0, 0)                   # all 128 wait for acc ready
    rb = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    rbio = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)
    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
    rbio.store(rb.load().to(io_dtype))
    cute.autovec_copy(rbio, tDgC[None, None, 0])
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
    print(f"=== M6 hand-rolled ring: G1 V^T C  M={M} ({(M+BM-1)//BM} k-tiles) ===")
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
    print(f"  M={M:4d}: rel={rel:.2e}  -> {'PASS (RING WORKS!)' if rel < tol else 'FAIL'}")
    return rel


def _time(fn, it=30):
    import torch
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.empty((8, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


if __name__ == "__main__":
    import traceback
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=0)
    a = p.parse_args()
    for M in ([a.M] if a.M else [64, 128, 512]):
        try:
            run(M)
        except Exception:
            print(f"  M={M}: EXC\n" + traceback.format_exc()[-2600:])

    # ---- ring vs barrier-kloop, single-CTA G1 at M=512 (does removing per-tile barriers pay off?) ----
    try:
        import torch
        import cute_apply_gemm1_kloop as KL
        M = 512
        V = torch.randn(M, IB, device="cuda", dtype=torch.float32)
        C = torch.randn(M, BW, device="cuda", dtype=torch.float32)
        W = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
        Tn = lambda x, d: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                           .mark_compact_shape_dynamic(mode=1, divisibility=d))
        ring = lambda: host_function(Tn(V, IB), Tn(C, BW), Tn(W, 256), M)
        barr = lambda: KL.host_function(Tn(V, IB), Tn(C, BW), Tn(W, 256), M)
        t_ring = _time(ring); t_barr = _time(barr)
        print(f"\n  [G1 single-CTA M=512]  RING={t_ring:.2f}us  BARRIER-kloop={t_barr:.2f}us  "
              f"speedup={t_barr/t_ring:.2f}x  ({'ring FASTER' if t_ring < t_barr else 'no win'})")
    except Exception:
        print("timing EXC\n" + traceback.format_exc()[-1500:])
    print("DONE")
