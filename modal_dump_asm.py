"""Dump FULL PTX + SASS hot-loop of the SOTA fused kernel (_fused_qr_k) for line-by-line reading.
Orthogonal to the moonshot (this is the plain-Triton V10). Run:
  modal run modal_dump_asm.py > /tmp/asm.log 2>&1
"""
import pathlib, modal
HERE = pathlib.Path(__file__).parent
# cand_fused.py lives on the persistent-engine worktree; mount it as submission.py.
CAND = HERE.parent / "dreamy-johnson-cc42cd" / "experiments" / "cand_fused.py"
HARN = HERE.parent / "dreamy-johnson-cc42cd" / "harness"
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .pip_install("numpy")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
         .add_local_dir(str(HARN), remote_path="/work", copy=True)
         .add_local_file(str(CAND), remote_path="/work/submission.py", copy=True))
app = modal.App("qr-dump-asm")


@app.function(gpu="B200", image=image, timeout=900)
def dump():
    import sys, importlib.util, gc, glob, subprocess, os
    sys.path.insert(0, "/work")
    import torch, triton
    sys.modules.setdefault("task", __import__("task"))
    spec = importlib.util.spec_from_file_location("uut", "/work/submission.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    import reference
    A = reference.generate_input(batch=640, n=512, cond=2, seed=999)
    for _ in range(4): mod.custom_kernel(A.clone())
    torch.cuda.synchronize()

    ptx = sass = None; name = None
    for obj in gc.get_objects():
        try:
            if type(obj).__name__ != "CompiledKernel": continue
            if getattr(obj, "name", "") != "_fused_qr_k": continue
            asm = getattr(obj, "asm", {}) or {}
            ptx = asm.get("ptx"); name = obj.name
            if "cubin" in asm:
                with open("/tmp/k.cubin", "wb") as f: f.write(asm["cubin"])
            break
        except Exception: continue

    print("==== KERNEL", name, "====")
    if ptx:
        print(f"\n========== FULL PTX ({ptx.count(chr(10))} lines) ==========\n")
        print(ptx)
    # SASS via cuobjdump
    if os.path.exists("/tmp/k.cubin"):
        r = subprocess.run("/usr/local/cuda/bin/cuobjdump -sass /tmp/k.cubin", shell=True,
                           capture_output=True, text=True)
        sass = r.stdout
        print(f"\n========== FULL SASS ({sass.count(chr(10))} lines) ==========\n")
        print(sass)
    return "done"


@app.local_entrypoint()
def main():
    print(dump.remote())
