"""Run the qr_v2 B200 benchmark on Modal.

Setup (one time):
    source .modalenv/bin/activate && modal setup     # browser auth

Usage:
    .modalenv/bin/modal run modal_app.py --submission submission_triton_1.py
    .modalenv/bin/modal run modal_app.py --submission cand_C_solveT_tf32.py --test

The submission's SOURCE is shipped as a string arg (so editing repo files between
runs never trips Modal's "modified during build" check). Only the stable harness/
dir is mounted into the image.
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    # cu128 torch wheels support B200 (sm_100) and bundle the matching triton.
    .pip_install("numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .add_local_dir(str(HERE / "harness"), remote_path="/work", copy=True)
)

app = modal.App("qr-v2-bench")


@app.function(gpu="B200", image=image, timeout=900)
def run(submission_src: str, test: bool = False, filt: str = "", stress: bool = False):
    import subprocess, sys
    with open("/work/submission.py", "w") as f:
        f.write(submission_src)
    cmd = [sys.executable, "gpu_bench.py", "submission.py"]
    if test:
        cmd.append("--test")
    if stress:
        cmd.append("--stress")
    if filt:
        cmd += ["--filter", filt]
    p = subprocess.run(cmd, cwd="/work", capture_output=True, text=True)
    return p.stdout + "\n----- STDERR (tail) -----\n" + p.stderr[-6000:]


@app.local_entrypoint()
def main(submission: str = "submission_triton_1.py", test: bool = False,
         filt: str = "", stress: bool = False):
    src = (HERE / submission).read_text()
    print(run.remote(src, test, filt, stress))
