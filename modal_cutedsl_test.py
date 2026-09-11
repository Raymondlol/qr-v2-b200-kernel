"""Feasibility probe for the CuTe-DSL (nvidia-cutlass-dsl) idea on B200.
Decisive questions for OUR competition (eval = torch+triton, NO nvcc):
  (A) is `cutlass`/cute-dsl PRE-PRESENT in a bare torch+triton image? (= the eval proxy)
  (B) if pip-installed, does it import + JIT-compile + RUN a trivial kernel on B200 WITHOUT nvcc
      (i.e. via the driver PTX JIT, the same path Triton uses) -> deployable-IF-present?
Run: modal run modal_cutedsl_test.py
"""
import pathlib, modal

# Image 1 = the eval proxy: torch + triton ONLY (matches modal_microbench / the official eval).
# Pinned 2026-09-11. The original runs did NOT pin torch, which means the lab numbers in
# results/ cannot be reproduced bit-for-bit -- a real gap in a project whose central claim
# is that codegen-sensitive results do not transfer. 2.12.0 is the version modal_cute_lab.py
# was pinned to during the same sessions. If it fails to resolve, set TORCH_SPEC = "torch"
# and record what actually got installed (every lab run now prints it).
TORCH_SPEC = "torch==2.12.0"

bare = (modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy")
        .pip_install(TORCH_SPEC, index_url="https://download.pytorch.org/whl/cu128"))
# Image 2 = torch+triton + the cute-dsl wheel, but NO cuda-devel (so NO nvcc) -> tests driver-JIT path.
# Pin to the EVAL version (4.5.2) so a green probe means our Modal loop matches the board.
withdsl = bare.pip_install("nvidia-cutlass-dsl==4.5.2")

app = modal.App("cutedsl-probe")


@app.function(gpu="B200", image=bare, timeout=300)
def probe_bare():
    import shutil, subprocess
    out = []
    out.append(f"nvcc present: {shutil.which('nvcc')}")
    for mod in ["cutlass", "cutlass.cute", "cute"]:
        try:
            __import__(mod); out.append(f"import {mod}: OK (PRE-PRESENT)")
        except Exception as e:
            out.append(f"import {mod}: ABSENT ({type(e).__name__})")
    return "=== BARE torch+triton image (eval proxy) ===\n" + "\n".join(out)


@app.function(gpu="B200", image=withdsl, timeout=600)
def probe_withdsl():
    import shutil, subprocess, traceback
    out = []
    out.append(f"nvcc present: {shutil.which('nvcc')}")
    try:
        import cutlass
        out.append(f"cutlass version: {getattr(cutlass, '__version__', '?')}")
    except Exception as e:
        out.append(f"import cutlass FAIL: {e}"); return "\n".join(out)
    try:
        import cutlass.cute as cute
        out.append("import cutlass.cute: OK")
    except Exception as e:
        out.append(f"import cutlass.cute FAIL: {repr(e)[:200]}")
    # minimal JIT compile+run: does it work on B200 with NO nvcc (driver PTX JIT)?
    try:
        import torch
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        @cute.jit
        def add_one(src: cute.Tensor, dst: cute.Tensor):
            # trivial elementwise; just to force a real JIT+launch on B200
            idx = cute.arch.thread_idx()[0] + cute.arch.block_idx()[0] * cute.arch.block_dim()[0]
            if idx < cute.size(src):
                dst[idx] = src[idx] + 1.0

        x = torch.arange(256, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        cute.compile(add_one, from_dlpack(x), from_dlpack(y))(from_dlpack(x), from_dlpack(y))
        torch.cuda.synchronize()
        ok = torch.allclose(y, x + 1)
        out.append(f"JIT compile+run trivial cute kernel on B200 (no nvcc): {'OK' if ok else 'RAN-WRONG'}")
    except Exception as e:
        out.append("JIT compile+run FAIL:\n" + traceback.format_exc()[-1500:])
    return "=== torch+triton + nvidia-cutlass-dsl, NO nvcc ===\n" + "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe_bare.remote())
    print()
    print(probe_withdsl.remote())
