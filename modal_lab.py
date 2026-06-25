"""modal_lab.py — entrypoint for the unified experiment lab (harness/lab.py).

Ships [baseline + N candidates] to ONE B200 container and runs them together, so
all timing is same-clock (apples-to-apples, no cross-run drift). Appends the
structured result to a local log (results/lab_log.jsonl).

    # variance-aware A/B (FIRST sub is the baseline):
    .modalenv/bin/modal run modal_lab.py --mode compare \
        --subs "submission.py,experiments/cand_a.py,experiments/cand_b.py"

    # fast correctness gate (22 official-shape cases, no timing):
    .modalenv/bin/modal run modal_lab.py --mode correctness --subs "experiments/cand_a.py"

    # op-level profile of one submission:
    .modalenv/bin/modal run modal_lab.py --mode profile --subs "submission.py"
"""
import pathlib, json, re, modal

HERE = pathlib.Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .add_local_dir(str(HERE / "harness"), remote_path="/work", copy=True)
)
app = modal.App("qr-lab")


@app.function(gpu="B200", image=image, timeout=900)
def run(mode: str, subs):
    import subprocess, sys
    paths = []
    for label, src in subs:
        fn = label.replace("/", "_")
        with open(f"/work/{fn}", "w") as f:
            f.write(src)
        paths.append(fn)
    p = subprocess.run([sys.executable, "lab.py", mode] + paths, cwd="/work",
                       capture_output=True, text=True)
    return p.stdout + "\n-----STDERR (tail)-----\n" + p.stderr[-4000:]


@app.local_entrypoint()
def main(mode: str = "compare", subs: str = "submission.py"):
    src = [(label.strip(), (HERE / label.strip()).read_text()) for label in subs.split(",")]
    out = run.remote(mode, src)
    print(out)
    m = re.search(r"===LAB_JSON===\n(.*)\n===END_LAB_JSON===", out, re.S)
    if m:
        logp = HERE / "results" / "lab_log.jsonl"
        logp.parent.mkdir(exist_ok=True)
        with open(logp, "a") as f:
            f.write(m.group(1).strip() + "\n")
        print(f"\n[lab] result appended to {logp}")
