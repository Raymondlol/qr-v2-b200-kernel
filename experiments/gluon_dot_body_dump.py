"""Dump the exact body of tl_dot_blackwell + its smem-operand helpers (L100-160) — the
canonical low-level tcgen05_mma setup we must replicate for an ASYNC mma. No banned subs."""
import os, triton
f = os.path.join(os.path.dirname(triton.__file__),
                 "tools/triton_to_gluon_translater/translator_helpers.py")
lines = open(f).read().splitlines()
for j in range(100, 162):
    print(f"{j:4d}: {lines[j]}")
