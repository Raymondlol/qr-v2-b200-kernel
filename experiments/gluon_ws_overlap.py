"""PHASE 0 (real): warp-specialized overlap via gl.warp_specialize.
Worker partition issues the async tcgen05_mma sequence (trailing-like, tensor cores);
default partition runs the gl.sum reduction (panel-like, CUDA cores) CONCURRENTLY.
Compare SEQ (both in one partition, serial) vs WS (warp_specialize, overlapped) on B200.

GO if WS wall-time -> max(T_mma,T_red) and SEQ -> sum, i.e. speedup>=1.3x. No banned subs.
"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory, tcgen05_mma, tcgen05_commit, TensorMemoryLayout,
    mbarrier, fence_async_shared, get_tmem_reg_layout,
)
from triton.tools.triton_to_gluon_translater.translator_helpers import (
    get_shared_memory_mma_operand, default_blocked_layout,
)


@gluon.jit
def _mma_part(a_smem, b_smem, acc_tmem, bar, NITER: gl.constexpr):
    for k in range(NITER):
        tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=(k > 0), mbarriers=[bar])
    mbarrier.wait(bar, phase=0)


@gluon.jit
def _reduce_part(X, RED, pid, BM: gl.constexpr, BR: gl.constexpr, NRED: gl.constexpr):
    # reduction (CUDA cores) -> writes RED directly (partitions return None to avoid the
    # flatten-return path that needs _semantic). gl.num_warps() here = default-partition warps.
    r_blk: gl.constexpr = default_blocked_layout([BM, BR], gl.num_warps())
    xm = gl.arange(0, BM, layout=gl.SliceLayout(1, r_blk))[:, None]
    xr = gl.arange(0, BR, layout=gl.SliceLayout(0, r_blk))[None, :]
    xt = gl.load(X + xm * BR + xr)
    red = gl.zeros([BM], gl.float32, layout=gl.SliceLayout(1, r_blk))
    for j in range(NRED):
        acc_chain = xt + red[:, None] * 1e-30
        red += gl.sum(acc_chain * acc_chain, axis=1)
    gl.store(RED + pid * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, r_blk)), red)


@gluon.jit
def fused(A, B, X, OUT, RED, sam, sak, sbk, sbn,
          BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BR: gl.constexpr,
          NITER: gl.constexpr, NRED: gl.constexpr, MODE: gl.constexpr,
          WK_WARPS: gl.constexpr, WK_REGS: gl.constexpr):
    pid = gl.program_id(0)

    a_blk: gl.constexpr = default_blocked_layout([BM, BK], gl.num_warps())
    b_blk: gl.constexpr = default_blocked_layout([BK, BN], gl.num_warps())
    am = gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
    ak = gl.arange(0, BK, layout=gl.SliceLayout(0, a_blk))[None, :]
    bk = gl.arange(0, BK, layout=gl.SliceLayout(1, b_blk))[:, None]
    bn = gl.arange(0, BN, layout=gl.SliceLayout(0, b_blk))[None, :]
    a = gl.load(A + am * sam + ak * sak)
    b = gl.load(B + bk * sbk + bn * sbn)
    a_smem = get_shared_memory_mma_operand(a, 0, False)
    b_smem = get_shared_memory_mma_operand(b, 1, False)

    m: gl.constexpr = 128 if BM >= 128 else 64
    n: gl.constexpr = 256 if BN >= 256 else BN
    col_stride: gl.constexpr = 32 // gl.float32.primitive_bitwidth
    acc_layout: gl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BM, BN), acc_layout, gl.num_warps())
    acc0 = gl.zeros([BM, BN], gl.float32, layout=reg_layout)
    acc_tmem = allocate_tensor_memory(gl.float32, [BM, BN], acc_layout, acc0)
    fence_async_shared()
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=NITER)

    if MODE == 0:  # SEQ: MMA fully, then reduction (no overlap)
        for k in range(NITER):
            tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=(k > 0), mbarriers=[bar])
        mbarrier.wait(bar, phase=0)
        _reduce_part(X, RED, pid, BM, BR, NRED)
    else:          # WS: reduction (default) || MMA (worker) overlapped
        gl.warp_specialize(
            [(_reduce_part, (X, RED, pid, BM, BR, NRED)),
             (_mma_part, (a_smem, b_smem, acc_tmem, bar, NITER))],
            [WK_WARPS], [WK_REGS])

    out = acc_tmem.load(reg_layout)
    out = gl.convert_layout(out, a_blk)
    cm = gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
    cn = gl.arange(0, BN, layout=gl.SliceLayout(0, a_blk))[None, :]
    gl.store(OUT + pid * BM * BN + cm * BN + cn, out)


def run(NPROG, NITER, NRED, MODE, BM=128, BN=128, BK=128, BR=256, nw=4, wk_warps=4, wk_regs=64):
    A = torch.randn(BM, BK, device="cuda")
    B = torch.randn(BK, BN, device="cuda")
    X = torch.randn(BM, BR, device="cuda")
    OUT = torch.empty(NPROG, BM, BN, device="cuda")
    RED = torch.empty(NPROG, BM, device="cuda")
    fused[(NPROG,)](A, B, X, OUT, RED, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                    BM=BM, BN=BN, BK=BK, BR=BR, NITER=max(NITER, 1), NRED=NRED,
                    MODE=MODE, WK_WARPS=wk_warps, WK_REGS=wk_regs, num_warps=nw)
    return A, B, X, OUT, RED


def check(NITER, NRED, MODE):
    A, B, X, OUT, RED = run(1, NITER, NRED, MODE)
    torch.cuda.synchronize()
    ref_mma = (A.double() @ B.double()).float() * NITER
    rel = (OUT[0] - ref_mma).abs().max().item() / (ref_mma.abs().max().item() + 1e-9)
    ref_red = ((X * X).sum(dim=1) * NRED)
    rrel = (RED[0] - ref_red).abs().max().item() / (ref_red.abs().max().item() + 1e-9)
    print(f"  MODE={MODE}: mma relerr={rel:.2e}, red relerr={rrel:.2e}")
    return rel < 5e-3 and rrel < 1e-3


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=50):
    for _ in range(8):
        try:
            fn()
        except Exception as e:
            return "ERR:" + repr(e)[:300]
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1000


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    print("\n=== correctness ===")
    try:
        ok_seq = check(8, 8, 0)
    except Exception as e:
        print("  SEQ FAIL:", repr(e)[:600]); return
    try:
        ok_ws = check(8, 8, 1)
    except Exception as e:
        import traceback
        print("  WS COMPILE/RUN FAIL (tail):", repr(e)[-1500:])
        print("  --- full traceback ---")
        traceback.print_exc()
        return
    print(f"  SEQ {'OK' if ok_seq else 'WRONG'}, WS {'OK' if ok_ws else 'WRONG'}")

    # async-worker overlap: (default nw / worker warps). 8+8=16-warp tests whether the
    # design-A warp budget (panel 8w + trailing 8w) still overlaps with the PROVEN low-level
    # async tcgen05 worker (isolates 16-warp occupancy from the tl_dot-coupling seen elsewhere).
    for NPROG, dnw, wkw, configs in [
        (640, 4, 4, [(512, 146), (1024, 293), (768, 146)]),
        (640, 8, 8, [(512, 146), (1024, 293), (768, 146)]),
        (640, 8, 4, [(512, 146), (1024, 293), (768, 146)]),
    ]:
        print(f"\n=== overlap (NPROG={NPROG}, default nw={dnw}, worker {wkw} warps = {dnw+wkw} total) ===")
        print(f"{'NITER':>6}{'NRED':>6} | {'mma':>8}{'red':>8}{'sum':>8}{'max':>8} | {'SEQ':>8}{'WS':>8} | {'spdup':>6}{'eff':>6}")
        for NITER, NRED in configs:
            t_mma = time_fn(lambda: run(NPROG, NITER, 0, 0, nw=dnw))
            t_red = time_fn(lambda: run(NPROG, 1, NRED, 0, nw=dnw))
            t_seq = time_fn(lambda: run(NPROG, NITER, NRED, 0, nw=dnw, wk_warps=wkw))
            t_ws = time_fn(lambda: run(NPROG, NITER, NRED, 1, nw=dnw, wk_warps=wkw))
            if isinstance(t_ws, str):
                print(f"{NITER:>6}{NRED:>6} | mma={t_mma} red={t_red} seq={t_seq} WS={t_ws}"); continue
            s = t_mma + t_red; mx = max(t_mma, t_red)
            spd = t_seq / t_ws
            eff = (t_seq - t_ws) / (t_seq - mx + 1e-9)
            print(f"{NITER:>6}{NRED:>6} | {t_mma:8.1f}{t_red:8.1f}{s:8.1f}{mx:8.1f} | {t_seq:8.1f}{t_ws:8.1f} | {spd:6.2f}{eff:6.2f}")
    print("\n16-WARP DIAG DONE (if 8+8 overlaps like 4+4 -> 16-warp occupancy OK, blocker is tl_dot coupling)")


if __name__ == "__main__":
    main()
