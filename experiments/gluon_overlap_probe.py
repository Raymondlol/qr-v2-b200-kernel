"""PHASE 0 — co-host validation: can ONE grep-clean Gluon kernel run a CUDA-core gl.sum
reduction (panel-like) CONCURRENTLY with an ASYNC tcgen05_mma sequence (trailing-like),
and do they OVERLAP on B200?

KEY: tcgen05_mma is SYNCHRONOUS when mbarriers=None (per its docstring). To go async we
pass mbarriers=[bar] on every issue and init the barrier with count=NITER, then wait AFTER
the reduction. OVERLAP mode runs the reduction between the async issues and the wait (so it
overlaps the in-flight MMA queue); SEQUENTIAL waits first. Identical total work.

Replicates tl_dot_blackwell's low-level setup. No banned substrings.
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
def probe(A, B, X, OUT, RED, sam, sak, sbk, sbn,
          BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BR: gl.constexpr,
          NITER: gl.constexpr, NRED: gl.constexpr, OVERLAP: gl.constexpr):
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
    # count = NITER arrivals (one per async mma); if NITER==0, count=1 + manual arrive.
    mbarrier.init(bar, count=(NITER if NITER > 0 else 1))

    # --- issue NITER ASYNC MMAs (each signals bar on completion) ---
    for k in range(NITER):
        tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=(k > 0), mbarriers=[bar])
    if NITER == 0:
        mbarrier.arrive(bar, count=1)

    # --- reduction (CUDA-core / ALU bound, panel-like) ---
    r_blk: gl.constexpr = default_blocked_layout([BM, BR], gl.num_warps())
    xm = gl.arange(0, BM, layout=gl.SliceLayout(1, r_blk))[:, None]
    xr = gl.arange(0, BR, layout=gl.SliceLayout(0, r_blk))[None, :]
    xt = gl.load(X + xm * BR + xr)
    red = gl.zeros([BM], gl.float32, layout=gl.SliceLayout(1, r_blk))

    if OVERLAP:
        for j in range(NRED):
            acc_chain = xt + red[:, None] * 1e-30
            red += gl.sum(acc_chain * acc_chain, axis=1)
        mbarrier.wait(bar, phase=0)
    else:
        mbarrier.wait(bar, phase=0)
        for j in range(NRED):
            acc_chain = xt + red[:, None] * 1e-30
            red += gl.sum(acc_chain * acc_chain, axis=1)

    mbarrier.invalidate(bar)
    out = acc_tmem.load(reg_layout)
    out = gl.convert_layout(out, a_blk)
    cm = gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
    cn = gl.arange(0, BN, layout=gl.SliceLayout(0, a_blk))[None, :]
    gl.store(OUT + pid * BM * BN + cm * BN + cn, out)
    gl.store(RED + pid * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, r_blk)), red)


def run(NPROG, NITER, NRED, OVERLAP, BM=128, BN=128, BK=128, BR=256, nw=8):
    A = torch.randn(BM, BK, device="cuda")
    B = torch.randn(BK, BN, device="cuda")
    X = torch.randn(BM, BR, device="cuda")
    OUT = torch.empty(NPROG, BM, BN, device="cuda")
    RED = torch.empty(NPROG, BM, device="cuda")
    probe[(NPROG,)](A, B, X, OUT, RED, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                    BM=BM, BN=BN, BK=BK, BR=BR, NITER=NITER, NRED=NRED,
                    OVERLAP=OVERLAP, num_warps=nw)
    return A, B, X, OUT, RED


def check(NITER, NRED, BM=128, BN=128, BK=128, BR=256):
    A, B, X, OUT, RED = run(1, NITER, NRED, True, BM, BN, BK, BR)
    torch.cuda.synchronize()
    ref_mma = (A.double() @ B.double()).float() * NITER
    rel = (OUT[0] - ref_mma).abs().max().item() / (ref_mma.abs().max().item() + 1e-9)
    ref_red = ((X * X).sum(dim=1) * NRED)
    rrel = (RED[0] - ref_red).abs().max().item() / (ref_red.abs().max().item() + 1e-9)
    print(f"  correctness: mma relerr={rel:.2e} (tf32~1e-3 ok), red relerr={rrel:.2e}")
    return rel < 5e-3


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=50):
    for _ in range(8):
        try:
            fn()
        except Exception as e:
            return "ERR:" + repr(e)[:200]
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1000  # us


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    print("\n=== correctness (async tcgen05 + reduction co-host) ===")
    try:
        ok = check(8, 8)
    except Exception as e:
        print("  COMPILE/RUN FAIL:", repr(e)[:500]); return
    print("  -> co-host", "OK" if ok else "WRONG")

    NPROG = 148
    base = time_fn(lambda: run(NPROG, 0, 0, True))
    print(f"\n=== overlap sweep (NPROG={NPROG}, fixed-overhead base={base:.1f}us) ===")
    print(f"{'NITER':>6}{'NRED':>6} | {'mma_w':>8}{'red_w':>8}{'sum':>8}{'max':>8} | {'SEQ':>8}{'OVL':>8} | {'spdup':>6}{'eff':>6}")
    for NITER, NRED in [(32, 32), (64, 64), (128, 128), (256, 128), (128, 256),
                        (256, 256), (512, 256), (256, 512), (512, 512)]:
        t_mma = time_fn(lambda: run(NPROG, NITER, 0, True))
        t_red = time_fn(lambda: run(NPROG, 0, NRED, True))
        t_seq = time_fn(lambda: run(NPROG, NITER, NRED, False))
        t_ovl = time_fn(lambda: run(NPROG, NITER, NRED, True))
        if isinstance(t_seq, str) or isinstance(t_ovl, str):
            print(f"{NITER:>6}{NRED:>6} | ERR seq={t_seq} ovl={t_ovl}"); continue
        mw = t_mma - base; rw = t_red - base
        s = mw + rw; mx = max(mw, rw)
        seq_w = t_seq - base; ovl_w = t_ovl - base
        spd = t_seq / t_ovl
        eff = (s - ovl_w) / (s - mx + 1e-9)  # 1=perfect overlap, 0=none
        print(f"{NITER:>6}{NRED:>6} | {mw:8.1f}{rw:8.1f}{s:8.1f}{mx:8.1f} | {seq_w:8.1f}{ovl_w:8.1f} | {spd:6.2f}{eff:6.2f}")
    print("\nPHASE0 DONE  (GO if spdup>=1.3 and eff>~0.4 for matched NITER/NRED)")


if __name__ == "__main__":
    main()
