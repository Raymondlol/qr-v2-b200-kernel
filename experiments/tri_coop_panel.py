"""MAKE-OR-BREAK microbench: at the n=4096 panel shape, does a MULTI-CTA cooperative panel
beat the single-CTA-per-matrix panel, AFTER paying the ~2.6us/barrier tax?

Single-CTA  : grid=(B,). One CTA owns the whole [M,W] tile (resident, ~128 regs/thread like the
              real panel), loops NCOL reflector columns: reduce column + rank-1 update. No barriers.
              At b=2 -> 2 CTAs on 148 SMs (146 idle = the large-n floor).
Cooperative : grid=(B*K,). K CTAs split the M rows of one matrix; per column: partial-reduce own
              rows -> atomicAdd to a per-(col,matrix) global slot -> grid barrier -> read full norm
              -> rank-1 update own rows -> grid barrier (so col j+1's reduction sees col j's update).
              2 barriers/column. Uses B*K SMs (the idle ones).

Work is the representative COST PATTERN (reduce M rows + elementwise update [.,W] per column), not
exact Householder (this is a timing test). GO if coop < single (extrapolated to NCOL=4096). No banned subs.
"""
import torch, triton, triton.language as tl


@triton.jit
def single_panel(tile_ptr, out_ptr, M: tl.constexpr, W: tl.constexpr, NCOL: tl.constexpr):
    b = tl.program_id(0)
    rm = tl.arange(0, M)[:, None]
    rw = tl.arange(0, W)[None, :]
    t = tl.load(tile_ptr + b * M * W + rm * W + rw)            # [M,W] resident
    for j in range(NCOL):
        norm = tl.sum(t * t)                                   # reduce over M*W (within-CTA)
        scale = 1.0 / (1.0 + norm * 1e-9)
        t = t * scale - 1e-9                                   # rank-1-like update over [M,W]
    tl.store(out_ptr + b, tl.sum(t))


@triton.jit
def coop_panel(tile_ptr, accum_ptr, counter_ptr, flag_ptr, out_ptr,
               M: tl.constexpr, K: tl.constexpr, W: tl.constexpr, NCOL: tl.constexpr,
               B: tl.constexpr, NCTA: tl.constexpr):
    pid = tl.program_id(0)
    matrix = pid // K
    sub = pid % K
    BLK: tl.constexpr = M // K
    rows = sub * BLK + tl.arange(0, BLK)[:, None]
    rw = tl.arange(0, W)[None, :]
    t = tl.load(tile_ptr + matrix * M * W + rows * W + rw)     # [BLK,W] resident (tiny)
    sense = 0
    for j in range(NCOL):
        partial = tl.sum(t * t)
        tl.atomic_add(accum_ptr + j * B + matrix, partial, sem="acq_rel")
        # ---- barrier 1: all partials added ----
        sense = 1 - sense
        old = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
        if old == NCTA - 1:
            tl.atomic_xchg(counter_ptr, 0, sem="acq_rel")
            tl.atomic_xchg(flag_ptr, sense, sem="release")
        else:
            d = tl.atomic_add(flag_ptr, 0, sem="acquire")
            while d != sense:
                d = tl.atomic_add(flag_ptr, 0, sem="acquire")
        norm = tl.load(accum_ptr + j * B + matrix)
        scale = 1.0 / (1.0 + norm * 1e-9)
        t = t * scale - 1e-9
        # ---- barrier 2: all updates done before col j+1's reduction ----
        sense = 1 - sense
        old = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
        if old == NCTA - 1:
            tl.atomic_xchg(counter_ptr, 0, sem="acq_rel")
            tl.atomic_xchg(flag_ptr, sense, sem="release")
        else:
            d = tl.atomic_add(flag_ptr, 0, sem="acquire")
            while d != sense:
                d = tl.atomic_add(flag_ptr, 0, sem="acquire")
    if sub == 0:
        tl.store(out_ptr + matrix, tl.sum(t))


def run_single(B, M, W, NCOL, nw=8):
    tile = torch.randn(B, M, W, device="cuda")
    out = torch.empty(B, device="cuda")
    single_panel[(B,)](tile, out, M=M, W=W, NCOL=NCOL, num_warps=nw)
    return out


def run_coop(B, M, K, W, NCOL, nw=2):
    tile = torch.randn(B, M, W, device="cuda")
    accum = torch.zeros(NCOL * B, device="cuda")
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.empty(B, device="cuda")
    NCTA = B * K
    coop_panel[(NCTA,)](tile, accum, counter, flag, out, M=M, K=K, W=W, NCOL=NCOL,
                        B=B, NCTA=NCTA, num_warps=nw)
    return out


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=20):
    for _ in range(4):
        try: fn()
        except Exception as e: return "ERR:" + repr(e)[:160]
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


def main():
    print("dev", torch.cuda.get_device_name(0), "SMs",
          torch.cuda.get_device_properties(0).multi_processor_count)
    B, M, W = 2, 4096, 8
    print(f"\n=== n=4096 panel chunk: single-CTA (b={B}, 2 CTAs) vs coop (K-CTA/matrix) ===")
    print(f"{'NCOL':>6}{'K':>5} | {'single':>10}{'coop':>10} | {'speedup':>8} | extrap NCOL=4096")
    for NCOL in [128, 512]:
        t_single = time_fn(lambda: run_single(B, M, W, NCOL))
        for K in [32, 64]:
            t_coop = time_fn(lambda: run_coop(B, M, K, W, NCOL))
            if isinstance(t_single, str) or isinstance(t_coop, str):
                print(f"{NCOL:>6}{K:>5} | single={t_single} coop={t_coop}"); continue
            spd = t_single / t_coop
            ext_s = t_single * (4096 / NCOL); ext_c = t_coop * (4096 / NCOL)
            print(f"{NCOL:>6}{K:>5} | {t_single:10.1f}{t_coop:10.1f} | {spd:8.2f} | "
                  f"single~{ext_s/1000:.1f}ms coop~{ext_c/1000:.1f}ms (geqrf~52ms)")
    print("\nCOOP PANEL MICROBENCH DONE (GO if coop < single AND coop-extrap < geqrf 52ms)")


if __name__ == "__main__":
    main()
