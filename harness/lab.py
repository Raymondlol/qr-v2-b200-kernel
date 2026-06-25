"""lab.py — unified B200 experiment lab (runs INSIDE the Modal container).

Replaces the old "one candidate per modal run, compare to a stale baseline" loop.
Implements the methodology upgrades (see docs/METHODOLOGY.md):
  (1) SAME-CONTAINER VARIANCE-AWARE A/B: baseline + N candidates are timed INTERLEAVED
      within each rep (shared clock state), R reps -> per-case paired delta + stderr.
      Only deltas > 2*stderr are "real" -> kills the cross-run clock-drift noise that
      made single-shot geomean comparisons (e.g. the NB128 sweep) lie.
  (2) FAST CORRECTNESS SMOKE GATE: exhaustive small-shape STRESS set, ALL code paths,
      NO timing -> catches Triton-kernel bugs (e.g. autotune-in-place) in seconds.
  (3) PROFILE: torch.profiler op-level breakdown (panel / GEMM / solve / glue).
  (5) STRUCTURED JSON: emits a machine-readable result block (modal_lab.py appends it
      to a local results log).
  (4) LOOP: orchestrated by a workflow that calls modal_lab in `compare` mode on a
      batch of generated candidates and promotes the significant winner.

CLI (invoked by modal_lab.py inside the container):
    python lab.py compare  sub_a.py sub_b.py ...      # A[0] is the baseline
    python lab.py correctness sub_a.py ...
    python lab.py profile  sub_a.py
"""
import sys, os, math, time, json, importlib.util
import torch
import reference

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# The 12 leaderboard benchmark cases (geomean is computed over these).
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
# Full official 22-case test-set shapes (correctness gate; exercises every code path).
STRESS = [
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
BYTES_TARGET = 256 * 1024 * 1024


def load_kernel(path):
    import task
    sys.modules.setdefault("task", task)
    spec = importlib.util.spec_from_file_location("uut_" + os.path.basename(path).replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def _spec(tc):
    return f"b{tc['batch']} n{tc['n']} {tc.get('case', 'dense')}"


# ----------------------------- (2) correctness smoke gate -----------------------------
def correctness(paths):
    print("=== (2) CORRECTNESS GATE (22 official-shape cases, no timing) ===")
    out = {}
    for p in paths:
        k = load_kernel(p)
        npass, fails = 0, []
        for tc in STRESS:
            data = reference.generate_input(**tc)
            try:
                good, msg = reference.check_implementation(data, k(data.clone()))
            except Exception as e:
                good, msg = False, f"EXC {type(e).__name__}: {e}"
            if good:
                npass += 1
            else:
                fails.append(f"{_spec(tc)}: {msg[:70]}")
        tag = "PASS" if npass == len(STRESS) else "FAIL"
        print(f"  [{tag}] {p}: {npass}/{len(STRESS)}")
        for f in fails[:6]:
            print(f"      {f}")
        out[p] = {"pass": npass, "total": len(STRESS), "ok": npass == len(STRESS),
                  "fails": fails}
    return out


# --------------------- (1) same-container variance-aware A/B timing -------------------
def _batch_count(tc):
    bpi = tc["batch"] * tc["n"] * tc["n"] * 4
    return max(1, min(8, BYTES_TARGET // bpi)) if bpi else 1


def _time_call(kfn, data_list):
    clear_l2()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for d in data_list:
        kfn(d)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / len(data_list)  # us per batched call


def compare(paths, reps=12):
    print(f"=== (1) VARIANCE-AWARE A/B: {len(paths)} kernels, {reps} interleaved reps ===")
    print(f"    baseline = {paths[0]}")
    kernels = {p: load_kernel(p) for p in paths}
    # generate inputs once per case
    data = {}
    for tc in BENCHMARKS:
        cnt = _batch_count(tc)
        args = dict(tc)
        lst = []
        for _ in range(cnt):
            if "seed" in args:
                args["seed"] += 42
            lst.append(reference.generate_input(**args))
        data[_spec(tc)] = lst
    cases = [_spec(tc) for tc in BENCHMARKS]
    # warmup every kernel on every case (JIT/autotune + sustained clocks)
    for kfn in kernels.values():
        for c in cases:
            for _ in range(2):
                _time_call(kfn, data[c])
    # interleaved timing: within each rep, time all kernels on each case back-to-back
    t = {p: {c: [] for c in cases} for p in paths}
    t0 = time.perf_counter()
    for _ in range(reps):
        for c in cases:
            for p in paths:
                t[p][c].append(_time_call(kernels[p], data[c]))
        if time.perf_counter() - t0 > 110:  # safety budget
            break
    # stats
    def mean(x): return sum(x) / len(x)
    def stderr(x):
        m = mean(x); v = sum((z - m) ** 2 for z in x) / max(1, len(x) - 1)
        return math.sqrt(v / len(x))
    res = {}
    base = paths[0]
    for p in paths:
        geo = math.exp(sum(math.log(mean(t[p][c])) for c in cases) / len(cases))
        per = {}
        for c in cases:
            bm, cm = mean(t[base][c]), mean(t[p][c])
            # paired per-rep deltas (cancels clock drift)
            n = min(len(t[base][c]), len(t[p][c]))
            d = [t[p][c][i] - t[base][c][i] for i in range(n)]
            per[c] = {"us": round(cm, 1), "se": round(stderr(t[p][c]), 1),
                      "d_vs_base": round(mean(d), 1), "d_se": round(stderr(d), 1) if n > 1 else 0.0}
        res[p] = {"geomean": round(geo, 1), "per_case": per}
    # report, ranked by geomean
    bg = res[base]["geomean"]
    print(f"\n{'kernel':46s} {'geomean':>9s} {'vs base':>9s}  significant per-case wins/losses (>2se)")
    for p in sorted(paths, key=lambda q: res[q]["geomean"]):
        g = res[p]["geomean"]
        sig = []
        for c in cases:
            pc = res[p]["per_case"][c]
            if p != base and abs(pc["d_vs_base"]) > 2 * max(pc["d_se"], 1e-9) and abs(pc["d_vs_base"]) > 0.02 * pc["us"]:
                sig.append(f"{c}{pc['d_vs_base']:+.0f}")
        ratio = f"{bg / g:.3f}x" if p != base else "(base)"
        print(f"{os.path.basename(p):46s} {g:9.1f} {ratio:>9s}  {' '.join(sig[:6])}")
    return res


# ----------------------------- (3) op-level profile -----------------------------
def _categorize(name):
    n = name.lower()
    if "panel" in n: return "PANEL"
    if "bmm_x3" in n or "_x3" in n: return "tf32x3 GEMM"
    if any(s in n for s in ("trsm", "triangular", "potrf", "getrf", "ormqr", "geqrf", "larf")): return "cuSOLVER(solve/qr)"
    if any(s in n for s in ("gemm", "cutlass", "tensorop", "cublas", "sm100", "sm90", "ampere")): return "cuBLAS GEMM"
    if any(s in n for s in ("tril", "triu", "where", "index", "fill", "copy", "elementwise", "sub", "add", "mul", "cast", "memcpy", "reciprocal", "arange", "cat", "zero")): return "elementwise glue"
    return "other"


def profile(path, reps=10):
    from torch.profiler import profile as tprofile, ProfilerActivity
    k = load_kernel(path)
    print(f"=== (3) PROFILE op-breakdown: {path} ===")
    out = {}
    for tc in [{"batch": 640, "n": 512, "cond": 2, "seed": 999},
               {"batch": 60, "n": 1024, "cond": 2, "seed": 999},
               {"batch": 8, "n": 2048, "cond": 1, "seed": 999}]:
        A = reference.generate_input(**tc)
        for _ in range(5):
            k(A.clone())
        torch.cuda.synchronize()
        with tprofile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                k(A.clone())
            torch.cuda.synchronize()
        cat = {}
        for e in prof.key_averages():
            ct = (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)) / reps
            if ct > 0:
                cat[_categorize(e.key)] = cat.get(_categorize(e.key), 0) + ct
        tot = sum(cat.values()) or 1
        print(f"\n  n={tc['n']} b={tc['batch']}  total CUDA {tot:.0f} us")
        for c in sorted(cat, key=lambda x: -cat[x]):
            print(f"    {c:20s} {cat[c]:9.1f}  {100*cat[c]/tot:5.1f}%")
        out[f"n{tc['n']}"] = {c: round(v, 1) for c, v in cat.items()}
    return out


def main():
    mode = sys.argv[1]
    paths = sys.argv[2:]
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
    if mode == "correctness":
        r = correctness(paths)
    elif mode == "compare":
        r = compare(paths)
    elif mode == "profile":
        r = profile(paths[0])
    else:
        print("unknown mode", mode); return
    # (5) structured result block (modal_lab.py extracts + appends to local log)
    print("\n===LAB_JSON===")
    print(json.dumps({"mode": mode, "paths": paths, "t": time.time(), "result": r}))
    print("===END_LAB_JSON===")


if __name__ == "__main__":
    main()
