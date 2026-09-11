import pathlib, modal
HERE = pathlib.Path(__file__).parent
# Pinned 2026-09-11. The original runs did NOT pin torch, which means the lab numbers in
# results/ cannot be reproduced bit-for-bit -- a real gap in a project whose central claim
# is that codegen-sensitive results do not transfer. 2.12.0 is the version modal_cute_lab.py
# was pinned to during the same sessions. If it fails to resolve, set TORCH_SPEC = "torch"
# and record what actually got installed (every lab run now prints it).
TORCH_SPEC = "torch==2.12.0"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy")
         .pip_install(TORCH_SPEC, index_url="https://download.pytorch.org/whl/cu128")
         .add_local_dir(str(HERE / "experiments"), remote_path="/work", copy=True)
         .add_local_dir(str(HERE / "harness"), remote_path="/work/harness", copy=True))
app = modal.App("qr-microbench")
@app.function(gpu="B200", image=image, timeout=900)
def run(script_name: str):
    import subprocess, sys
    p = subprocess.run([sys.executable, script_name], cwd="/work", capture_output=True, text=True)
    return p.stdout + "\n--- STDERR ---\n" + p.stderr[-4000:]
@app.local_entrypoint()
def main(script: str = "microbench_trailing.py"):
    print(run.remote(script))
