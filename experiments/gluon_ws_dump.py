"""Dump gl.warp_specialize: signature, source, and the code-generator lowering so we learn
the exact calling convention (partition functions, num_warps per partition, args, barriers).
Also grep installed triton tests/examples that actually CALL it. No banned substrings."""
import os, glob, inspect, importlib, triton
from triton.experimental.gluon import language as gl
print("triton", triton.__version__)

ws = getattr(gl, 'warp_specialize', None)
print("\n=== gl.warp_specialize ===")
try:
    print("  sig:", inspect.signature(ws))
except Exception as e:
    print("  sig N/A:", e)
try:
    print("  --- source ---")
    print(inspect.getsource(ws))
except Exception as e:
    print("  src N/A:", e)

# the semantic implementation
print("\n=== _semantic.warp_specialize impl ===")
sem = importlib.import_module("triton.experimental.gluon.language._semantic")
for nm in dir(sem):
    if 'warp_special' in nm.lower():
        try:
            o = getattr(sem, nm)
            print(inspect.getsource(o)[:3000])
        except Exception as e:
            print("  ", nm, e)
# maybe it's a method on the semantic class
for cn in dir(sem):
    c = getattr(sem, cn)
    if isinstance(c, type):
        for mn in dir(c):
            if 'warp_special' in mn.lower():
                try:
                    print(f"\n  --- {cn}.{mn} ---")
                    print(inspect.getsource(getattr(c, mn))[:3000])
                except Exception as e:
                    print("   ", mn, e)

print("\n=== code_generator.py warp_specialize handling (context) ===")
cg = os.path.join(os.path.dirname(triton.__file__), "compiler/code_generator.py")
lines = open(cg).read().splitlines()
for i, l in enumerate(lines):
    if 'warp_specialize' in l:
        lo = max(0, i - 3); hi = min(len(lines), i + 8)
        print(f"  --- L{i} ---")
        for j in range(lo, hi):
            print(f"  {j:5d}: {lines[j]}")

print("\n=== grep tests/anywhere that CALL warp_specialize (usage examples) ===")
root = os.path.dirname(os.path.dirname(triton.__file__))  # site-packages
for f in glob.glob(os.path.join(root, '**', '*.py'), recursive=True):
    try: txt = open(f).read()
    except Exception: continue
    if 'warp_specialize(' in txt and 'def warp_specialize' not in txt:
        print("  USE:", f.replace(root, '...'))
        for i, l in enumerate(txt.splitlines()):
            if 'warp_specialize(' in l:
                print(f"      L{i}: {l.strip()[:120]}")
print("\nDONE")
