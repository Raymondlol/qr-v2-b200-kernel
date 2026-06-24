"""Standalone B200 benchmark, faithful to eval.py leaderboard timing.

Runs the 12 official benchmark cases (or the test set with --test), validates
correctness with reference.check_implementation, and times with CUDA events +
L2-cache clear, reporting per-case mean (us per batched call) and the geomean
score (what the leaderboard ranks on).

    python gpu_bench.py <submission.py> [--test] [--cases n=512]
"""
import sys, math, time, importlib.util, argparse
import torch

import reference

BENCHMARKS = [
    {"batch": 20, "n": 32, "cond": 1, "seed": 43214},
    {"batch": 40, "n": 176, "cond": 1, "seed": 423011},
    {"batch": 40, "n": 352, "cond": 1, "seed": 123456},
    {"batch": 640, "n": 512, "cond": 2, "seed": 1029},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 75342},
    {"batch": 8, "n": 2048, "cond": 1, "seed": 224466},
    {"batch": 2, "n": 4096, "cond": 1, "seed": 32412},
    {"batch": 640, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770003, "case": "rankdef"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770004, "case": "clustered"},
    {"batch": 60, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"},
]

MAX_ITERS = 50
BYTES_TARGET = 256 * 1024 * 1024


def clear_l2_cache():
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    dummy.fill_(42)
    del dummy


def load_kernel(path):
    import task
    sys.modules.setdefault("task", task)
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def batch_count(batch, n):
    bpi = batch * n * n * 4
    if bpi <= 0:
        return 1
    return max(1, min(MAX_ITERS, BYTES_TARGET // bpi))


def bench_one(kernel, tc, max_repeats=1000, max_time_ns=30e9):
    count = batch_count(tc["batch"], tc["n"])
    args = dict(tc)
    data_list = []
    for _ in range(count):
        if "seed" in args:
            args["seed"] += 42
        data_list.append(reference.generate_input(**args))
    check_copy = [d.clone() for d in data_list]

    # correctness gate (pre-timing)
    outs = [kernel(d.clone()) for d in data_list]
    for ref_d, out in zip(check_copy, outs):
        good, msg = reference.check_implementation(ref_d, out)
        if not good:
            return None, msg

    durations = []
    t0 = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()
        clear_l2_cache()
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        outs = [kernel(d) for d in data_list]
        ee.record()
        torch.cuda.synchronize()
        durations.append(se.elapsed_time(ee) * 1e6 / len(data_list))  # ns per input
        total = time.perf_counter_ns() - t0
        if i > 1 and total > 1e8:
            mean = sum(durations) / len(durations)
            var = sum((x - mean) ** 2 for x in durations) / (len(durations) - 1)
            err = math.sqrt(var) / math.sqrt(len(durations))
            if err / mean < 0.001 or mean * len(durations) > max_time_ns or total > 120e9:
                break
    mean = sum(durations) / len(durations)
    return mean, f"runs={len(durations)} count={count}"


# Official test-set stress cases (correctness gate in real submission). If the
# custom large-n path is numerically wrong on these, the whole submission fails.
STRESS = [  # the full official 22-case test set (task.yml) — the submission gate
    {"batch": 20, "n": 32, "cond": 1, "seed": 53124},
    {"batch": 40, "n": 176, "cond": 1, "seed": 3321},
    {"batch": 40, "n": 352, "cond": 1, "seed": 1200},
    {"batch": 16, "n": 512, "cond": 2, "seed": 32523},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 4327},
    {"batch": 1, "n": 4096, "cond": 1, "seed": 75342},
    {"batch": 16, "n": 512, "cond": 4, "seed": 32524, "case": "dense"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32525, "case": "rankdef"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32526, "case": "clustered"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32527, "case": "band"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32528, "case": "rowscale"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32529, "case": "nearcollinear"},
    {"batch": 4, "n": 1024, "cond": 4, "seed": 4328, "case": "dense"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4329, "case": "rankdef"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4330, "case": "nearrank"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4331, "case": "clustered"},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 224466, "case": "dense"},
    {"batch": 2, "n": 2048, "cond": 0, "seed": 224467, "case": "rankdef"},
    {"batch": 1, "n": 4096, "cond": 0, "seed": 75343, "case": "upper"},
    {"batch": 16, "n": 512, "cond": 2, "seed": 32530, "case": "mixed"},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 4332, "case": "mixed"},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 224468, "case": "mixed"},
]


def run_stress(kernel):
    print("\n--- stress correctness (official test-set shapes) ---")
    nfail = 0
    for tc in STRESS:
        data = reference.generate_input(**tc)
        out = kernel(data.clone())
        good, msg = reference.check_implementation(data, out)
        tag = "PASS" if good else "FAIL"
        nfail += (not good)
        print(f"[{tag}] n={tc['n']:5d} {tc.get('case','dense'):10s} {msg[:80]}")
    print(f"stress: {len(STRESS)-nfail}/{len(STRESS)} pass")
    return nfail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--filter", default="")
    args = ap.parse_args()

    print("torch", torch.__version__, "| cuda", torch.version.cuda,
          "| dev", torch.cuda.get_device_name(0))
    print("tf32 matmul:", torch.backends.cuda.matmul.allow_tf32)
    kernel = load_kernel(args.submission)

    if args.stress:
        run_stress(kernel)

    cases = BENCHMARKS
    if args.filter:
        cases = [c for c in cases if args.filter in str(c)]

    # FULL warmup pass over ALL cases (mimics eval.py leaderboard mode: it warms
    # every test first, which heats the GPU to sustained/throttled clocks before
    # timing). Without this, light-warmup timing catches boost clocks and is
    # optimistic by up to ~1.7x on the compute-heavy custom-path cases (n=512/1024).
    print("warming up (all cases, ~0.5s each) to reach sustained clocks...")
    for tc in cases:
        bench_one(kernel, tc, max_repeats=1000, max_time_ns=5e8)

    geo = 0.0
    n_ok = 0
    print(f"\n{'spec':52s} {'us/call':>12s}  note")
    for tc in cases:
        mean_ns, note = bench_one(kernel, tc)
        spec = f"b={tc['batch']} n={tc['n']} cond={tc['cond']} case={tc.get('case','dense')}"
        if mean_ns is None:
            print(f"[FAIL] {spec:46s} {'-':>12s}  {note[:80]}")
            continue
        us = mean_ns / 1000.0
        geo += math.log(us)
        n_ok += 1
        print(f"       {spec:46s} {us:12.1f}  {note}")
    if n_ok:
        print(f"\nGEOMEAN over {n_ok} cases: {math.exp(geo / n_ok):.1f} us"
              f"   (leaderboard-style score)")


if __name__ == "__main__":
    main()
