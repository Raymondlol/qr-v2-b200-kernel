"""Dump the version-matched (3.6.0) Gluon tcgen05 idioms from the installed package's
own translator_helpers.py, and check the convenience APIs (get_default_for, acc.load
signature). Ground truth beats the main-branch tutorial. No banned substrings.
"""
import os, re, inspect, triton
from triton.experimental.gluon import language as gl
print("triton", triton.__version__)
print("NVMMASharedLayout.get_default_for:", hasattr(gl.NVMMASharedLayout, "get_default_for"))

f = os.path.join(os.path.dirname(triton.__file__), "tools/triton_to_gluon_translater/translator_helpers.py")
txt = open(f).read()
lines = txt.splitlines()
# print windows around the key idioms
keys = ['tcgen05_mma','allocate_tensor_memory','TensorMemoryLayout','NVMMASharedLayout',
        'allocate_shared_memory','async_load','async_copy','.load(','mbarrier','tcgen05_commit',
        'get_default_for','tma.','TensorDescriptor','arange']
seen=set()
for i,l in enumerate(lines):
    if any(k in l for k in keys):
        lo=max(0,i-2); hi=min(len(lines),i+3)
        if lo in seen: continue
        for j in range(lo,hi): seen.add(j)
        print(f"\n--- L{i} ---")
        for j in range(lo,hi):
            print(f"{j:4d}: {lines[j]}")
