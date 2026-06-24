"""Local CPU correctness harness for qr_v2.

Runs the *actual* reference.py checker against a submission's custom_kernel on CPU.
Catches numeric/contract bugs without a GPU. (Triton paths won't run on CPU; this
validates the eager/torch path and any new algorithm's CPU behavior.)

Usage:
    python check_local.py ../submission.py [maxn]
Only runs test cases with n <= maxn (default 512) so CPU stays fast.
"""
import sys, importlib.util, time
import torch

sys.path.insert(0, ".")  # harness dir has reference.py, task.py, utils.py
import reference

# The 22 official test cases (from task.yml), plus the 12 benchmark shapes scaled down.
TEST_CASES = [
    {"batch": 20, "n": 32, "cond": 1, "seed": 53124},
    {"batch": 40, "n": 176, "cond": 1, "seed": 3321},
    {"batch": 40, "n": 352, "cond": 1, "seed": 1200},
    {"batch": 16, "n": 512, "cond": 2, "seed": 32523},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 4327},
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
    {"batch": 16, "n": 512, "cond": 2, "seed": 32530, "case": "mixed"},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 4332, "case": "mixed"},
]


def load_kernel(path):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    # make `from task import ...` work
    import task
    sys.modules.setdefault("task", task)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../submission.py"
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    torch.manual_seed(0)
    kernel = load_kernel(path)
    npass = nfail = 0
    for tc in TEST_CASES:
        if tc["n"] > maxn:
            continue
        data = reference.generate_input(**tc)
        t0 = time.time()
        out = kernel(data.clone())
        dt = time.time() - t0
        good, msg = reference.check_implementation(data, out)
        tag = "PASS" if good else "FAIL"
        if good:
            npass += 1
        else:
            nfail += 1
        spec = f"b={tc['batch']} n={tc['n']} cond={tc['cond']} case={tc.get('case','dense')}"
        print(f"[{tag}] {spec:48s} {dt*1e3:7.1f}ms  {msg[:120]}")
    print(f"\n{npass} passed, {nfail} failed")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
