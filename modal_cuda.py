"""Modal runner for raw-CUDA (load_inline) work on B200 sm_100. Uses a CUDA-devel
base image (the torch cu128 wheel ships the runtime but NOT nvcc, which load_inline
needs). Image is cached after first build, so subsequent compile-iterate is fast.

    .modalenv/bin/modal run modal_cuda.py --script cuda_gate1.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent

# Pinned 2026-09-11. The original runs did NOT pin torch, which means the lab numbers in
# results/ cannot be reproduced bit-for-bit -- a real gap in a project whose central claim
# is that codegen-sensitive results do not transfer. 2.12.0 is the version modal_cute_lab.py
# was pinned to during the same sessions. If it fails to resolve, set TORCH_SPEC = "torch"
# and record what actually got installed (every lab run now prints it).
TORCH_SPEC = "torch==2.12.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("numpy", "ninja")
    .pip_install(TORCH_SPEC, index_url="https://download.pytorch.org/whl/cu128")
    .add_local_dir(str(HERE / "experiments"), remote_path="/work", copy=True)
    .add_local_dir(str(HERE / "harness"), remote_path="/work/harness", copy=True)
)

app = modal.App("qr-cuda")


@app.function(gpu="B200", image=image, timeout=1800)
def run(script_name: str):
    import subprocess, sys, os
    env = dict(os.environ)
    env["TORCH_CUDA_ARCH_LIST"] = "10.0"
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    p = subprocess.run([sys.executable, script_name], cwd="/work",
                       capture_output=True, text=True, env=env)
    return p.stdout + "\n--- STDERR (tail) ---\n" + p.stderr[-8000:]


@app.local_entrypoint()
def main(script: str = "cuda_gate1.py"):
    print(run.remote(script))
