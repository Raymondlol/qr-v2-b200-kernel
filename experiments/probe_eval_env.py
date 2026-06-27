"""ZERO-RISK eval-environment probe submission. Prints what's actually available in the qr_v2 eval
(nvcc? cutlass? cuda-python?) to stdout+stderr (-> the gpumode 'Debug Info' column), then returns a
CORRECT result via torch.geqrf (22/22, low score, no risk). Submit this ONCE to confirm the eval image
has nvcc + cute-dsl (verified for the general kernelbot main; this confirms the qr_v2 board specifically).
No banned substrings."""
import sys
import os
import shutil
import torch


def _probe_env():
    lines = []
    lines.append(f"nvcc={shutil.which('nvcc')}")
    lines.append(f"CUTLASS_PATH={os.environ.get('CUTLASS_PATH')}")
    lines.append(f"opt_cutlass_exists={os.path.isdir('/opt/cutlass')}")
    for mod in ["cutlass", "cutlass.cute", "cuda", "cuda.core", "cuda.bindings", "triton", "helion"]:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            lines.append(f"import {mod}=OK(v{ver})")
        except Exception as e:
            lines.append(f"import {mod}=FAIL({type(e).__name__})")
    msg = "EVAL-ENV-PROBE :: " + " | ".join(lines)
    print(msg, file=sys.stdout, flush=True)
    print(msg, file=sys.stderr, flush=True)


_probe_env()


def custom_kernel(data):
    # correct + safe fallback; we only care about the printed env info above.
    return torch.geqrf(data)
