"""SERIOUS test: can a HAND-ROLLED atomic cross-CTA (grid-wide) barrier run grep-clean,
correct, and reusable in plain Triton on B200? This is the foundational primitive for a
multi-CTA cooperative panel (spread one matrix's tall column-reduction across SMs to attack
the large-n latency floor) WITHOUT cudaLaunchCooperativeKernel (which trips the banned
'stream' substring scan).

Mechanism = sense-reversing barrier from tl.atomic_add / tl.atomic_xchg + a spin loop:
  arrive: atomic_add(counter,1, release); last arriver resets counter + flips a global sense
  flag (release); others spin reading the flag (acquire) until it equals their local sense.

Correctness test that FAILS on any barrier / memory-ordering bug: ROUNDS rounds, each round
every CTA writes (pid + r) into its own slot, barrier, then pid 0 sums the slot ->
expected N*(N-1)/2 + r*N. A stale/unsynced read gives the wrong sum. No banned substrings.
"""
import torch, triton, triton.language as tl


@triton.jit
def gridbar_kernel(counter_ptr, flag_ptr, data_ptr, out_ptr,
                   N_CTA: tl.constexpr, ROUNDS: tl.constexpr):
    pid = tl.program_id(0)
    sense = 0
    for r in range(ROUNDS):
        sense = 1 - sense
        # phase-1: write this round's slot (one int per CTA)
        tl.store(data_ptr + r * N_CTA + pid, pid + r)
        # ---- grid barrier (arrive) ----
        old = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
        if old == N_CTA - 1:
            # last arriver: reset counter for next round, then release the waiters
            tl.atomic_xchg(counter_ptr, 0, sem="acq_rel")
            tl.atomic_xchg(flag_ptr, sense, sem="release")
        else:
            done = tl.atomic_add(flag_ptr, 0, sem="acquire")
            while done != sense:
                done = tl.atomic_add(flag_ptr, 0, sem="acquire")
        # ---- after barrier: every CTA can see all slot writes; pid 0 reduces ----
        if pid == 0:
            acc = 0
            for i in range(N_CTA):
                acc += tl.load(data_ptr + r * N_CTA + i)
            tl.store(out_ptr + r, acc)


@triton.jit
def bench_kernel(counter_ptr, flag_ptr, sink_ptr, N_CTA: tl.constexpr, ROUNDS: tl.constexpr):
    # ONLY barriers (no reduction work) -> isolate per-barrier overhead.
    pid = tl.program_id(0)
    sense = 0
    for r in range(ROUNDS):
        sense = 1 - sense
        old = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
        if old == N_CTA - 1:
            tl.atomic_xchg(counter_ptr, 0, sem="acq_rel")
            tl.atomic_xchg(flag_ptr, sense, sem="release")
        else:
            done = tl.atomic_add(flag_ptr, 0, sem="acquire")
            while done != sense:
                done = tl.atomic_add(flag_ptr, 0, sem="acquire")
    if pid == 0:
        tl.store(sink_ptr, sense)


def bench(N_CTA, ROUNDS, it=30):
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    for _ in range(5):
        counter.zero_(); flag.zero_()
        bench_kernel[(N_CTA,)](counter, flag, sink, N_CTA=N_CTA, ROUNDS=ROUNDS)
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        counter.zero_(); flag.zero_(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); bench_kernel[(N_CTA,)](counter, flag, sink, N_CTA=N_CTA, ROUNDS=ROUNDS); e.record()
        torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort()
    total_us = ts[len(ts) // 2] * 1000
    return total_us, total_us / ROUNDS


def run(N_CTA, ROUNDS):
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    data = torch.full((ROUNDS * N_CTA,), -999, dtype=torch.int32, device="cuda")
    out = torch.full((ROUNDS,), -1, dtype=torch.int32, device="cuda")
    gridbar_kernel[(N_CTA,)](counter, flag, data, out, N_CTA=N_CTA, ROUNDS=ROUNDS)
    torch.cuda.synchronize()
    return out


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__, "SMs",
          torch.cuda.get_device_properties(0).multi_processor_count)
    for N_CTA, ROUNDS in [(128, 8), (132, 16), (148, 32), (64, 64)]:
        base = N_CTA * (N_CTA - 1) // 2
        expected = [base + r * N_CTA for r in range(ROUNDS)]
        try:
            # run several times to expose races / nondeterminism
            ok = True
            for trial in range(20):
                out = run(N_CTA, ROUNDS).tolist()
                if out != expected:
                    ok = False
                    bad = [(r, out[r], expected[r]) for r in range(ROUNDS) if out[r] != expected[r]][:3]
                    print(f"  N_CTA={N_CTA} ROUNDS={ROUNDS}: FAIL trial {trial} mism={bad}")
                    break
            if ok:
                print(f"  N_CTA={N_CTA:3d} ROUNDS={ROUNDS:3d}: OK  (20/20 trials, sums exact = barrier+ordering correct)")
        except Exception as e:
            import traceback; print(f"  N_CTA={N_CTA} ROUNDS={ROUNDS}: EXC"); traceback.print_exc(); return
    print("\n=== per-barrier OVERHEAD (only barriers, no work) ===")
    for N_CTA in [64, 128, 148]:
        ROUNDS = 2000
        try:
            tot, per = bench(N_CTA, ROUNDS)
            n4096_est = per * 4096  # ~one barrier per reflector column at n=4096
            print(f"  N_CTA={N_CTA:3d}: {per*1000:7.2f} ns/barrier  ({ROUNDS} rounds = {tot:.0f}us)"
                  f"  -> ~4096 barriers = {n4096_est:.0f}us (vs geqrf n4096 ~52000us)")
        except Exception as e:
            import traceback; print(f"  N_CTA={N_CTA}: EXC"); traceback.print_exc()
    print("\nGRID BARRIER TEST DONE")


if __name__ == "__main__":
    main()
