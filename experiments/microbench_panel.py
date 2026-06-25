"""Panel kernel diagnosis: the panel (_panel_kernel) is now ~50% of n=512/1024 and
the dominant cost. Is it COMPUTE-bound (O(BN*BCOLS^2) rank-1 updates -> time ~ ib^2,
fix = micro-block with tl.dot) or LATENCY/OCCUPANCY-bound (sequential ib columns, or
too few CTAs at low batch -> time ~ ib or ~ flat, fix = parallelism)?

Isolates the current _panel_kernel and sweeps (B, m, ib, num_warps): time per launch,
and ib-scaling at fixed m. No banned substrings.
"""
import torch, triton, triton.language as tl
torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def _panel_kernel(Hptr, tauptr, N, K, M, BWID, BN: tl.constexpr, BCOLS: tl.constexpr):
    bid = tl.program_id(0)
    Hb = Hptr + bid * N * N
    taub = tauptr + bid * N
    rar = tl.arange(0, BN); car = tl.arange(0, BCOLS)
    rmask = rar < M; cmask = car < BWID
    ptr = Hb + (K + rar)[:, None] * N + (K + car)[None, :]
    tmask = rmask[:, None] & cmask[None, :]
    tile = tl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        colj = tl.sum(tl.where(car[None, :] == j, tile, 0.0), axis=1)
        alpha = tl.sum(tl.where(rar == j, colj, 0.0))
        xnorm2 = tl.sum(tl.where(rar > j, colj * colj, 0.0))
        normfull = tl.sqrt(alpha * alpha + xnorm2)
        sgn = tl.where(alpha >= 0.0, 1.0, -1.0); beta = -sgn * normfull
        need = xnorm2 > 0.0
        scale = tl.where(need, 1.0 / (alpha - beta), 0.0)
        tau_j = tl.where(need, (beta - alpha) / beta, 0.0)
        v = tl.where(rar > j, colj * scale, 0.0); v = tl.where(rar == j, 1.0, v)
        w = tl.sum(v[:, None] * tile, axis=0)
        upd = tile - tau_j * (v[:, None] * w[None, :])
        tile = tl.where(car[None, :] > j, upd, tile)
        diagval = tl.where(need, beta, alpha)
        newcol = tl.where(rar > j, colj * scale, colj); newcol = tl.where(rar == j, diagval, newcol)
        tile = tl.where(car[None, :] == j, newcol[:, None], tile)
        tl.store(taub + K + j, tau_j, mask=(j < BWID))
    tl.store(ptr, tile, mask=tmask)


def run_panel(H, tau, n, col, m, ib, nw):
    B = H.shape[0]
    BN = triton.next_power_of_2(m); BCOLS = triton.next_power_of_2(ib)
    _panel_kernel[(B,)](H, tau, n, col, m, ib, BN=BN, BCOLS=BCOLS, num_warps=nw)


def clear_l2(): torch.empty((32,1024,1024),dtype=torch.int64,device="cuda").fill_(0)
def time_fn(fn, it=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]*1000


def main():
    print("dev", torch.cuda.get_device_name(0), "| #SM", torch.cuda.get_device_properties(0).multi_processor_count)
    # (label, B, n, m, ib) -- first sub-panel of each case (full height m=n)
    cfgs = [
        ("n512  b640 m512 ib64 ", 640, 512, 512, 64),
        ("n1024 b60  m1024 ib32", 60, 1024, 1024, 32),
        ("n2048 b8   m2048 ib16", 8, 2048, 2048, 16),
    ]
    print(f"\n=== best num_warps per real first-sub-panel (us/launch) ===")
    print(f"{'cfg':24s} {'nw=4':>8s} {'nw=8':>8s} {'nw=16':>8s} {'nw=32':>8s}")
    for label, B, n, m, ib in cfgs:
        H = torch.randn(B, n, n, device="cuda"); tau = torch.zeros(B, n, device="cuda")
        row = []
        for nw in (4, 8, 16, 32):
            try:
                t = time_fn(lambda nw=nw: run_panel(H.clone(), tau, n, 0, m, ib, nw))
            except Exception:
                t = float('nan')
            row.append(t)
        print(f"{label:24s} " + " ".join(f"{x:8.1f}" for x in row))

    # ib-scaling at fixed m (n=512 b=640): linear -> latency-bound, quadratic -> compute-bound
    print(f"\n=== ib-scaling at m=512 b=640 (nw=16): time vs ib tells compute(ib^2) vs latency(ib) ===")
    B, n, m = 640, 512, 512
    H = torch.randn(B, n, n, device="cuda"); tau = torch.zeros(B, n, device="cuda")
    prev = None
    for ib in (8, 16, 32, 64, 128):
        t = time_fn(lambda ib=ib: run_panel(H.clone(), tau, n, 0, m, ib, 16))
        ratio = f"  x{t/prev:.2f} vs prev ib" if prev else ""
        print(f"  ib={ib:4d}: {t:8.1f} us{ratio}  (per-col {t/ib:.1f} us)")
        prev = t

    # batch-scaling at n=2048 (occupancy probe): does more matrices fill the GPU linearly?
    print(f"\n=== batch-scaling n=2048 m=2048 ib=16 nw=32 (occupancy: 8 CTAs on 148 SM) ===")
    for B in (2, 8, 16, 64, 148):
        H = torch.randn(B, 2048, 2048, device="cuda"); tau = torch.zeros(B, 2048, device="cuda")
        t = time_fn(lambda: run_panel(H.clone(), tau, 2048, 0, 2048, 16, 32), it=15)
        print(f"  B={B:4d}: {t:8.1f} us  ({t/B:.1f} us/matrix)")


if __name__ == "__main__":
    main()
