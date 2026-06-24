import pathlib, modal
HERE = pathlib.Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
         .add_local_dir(str(HERE / "experiments"), remote_path="/work", copy=True))
app = modal.App("qr-microbench")
@app.function(gpu="B200", image=image, timeout=900)
def run(script_name: str):
    import subprocess, sys
    p = subprocess.run([sys.executable, script_name], cwd="/work", capture_output=True, text=True)
    return p.stdout + "\n--- STDERR ---\n" + p.stderr[-4000:]
@app.local_entrypoint()
def main(script: str = "microbench_trailing.py"):
    print(run.remote(script))
