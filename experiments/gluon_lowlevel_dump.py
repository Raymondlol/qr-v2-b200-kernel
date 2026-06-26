"""Dump the EXACT low-level async tcgen05 calling sequence from installed triton 3.6.0:
the full source of tl_dot (the only known-correct usage of tcgen05_mma in-tree) plus the
source of the low-level blackwell ops (allocate_tensor_memory, tcgen05_mma/commit/copy) so
we can hand-roll an ASYNC (non-blocking) mma + a concurrent reduction. No banned substrings.
"""
import os, inspect, importlib, triton
print("triton", triton.__version__)

# 1) Full tl_dot source (high-level, but its internals = correct low-level sequence)
f = os.path.join(os.path.dirname(triton.__file__),
                 "tools/triton_to_gluon_translater/translator_helpers.py")
txt = open(f).read()
lines = txt.splitlines()
# find tl_dot def and print until next top-level def
start = None
for i, l in enumerate(lines):
    if l.startswith("def tl_dot") or l.startswith("@") and i+1 < len(lines) and lines[i+1].startswith("def tl_dot"):
        start = i
        break
if start is not None:
    print(f"\n===== tl_dot @ L{start} =====")
    j = start
    # walk back to include decorators
    while start > 0 and (lines[start-1].startswith("@") or lines[start-1].strip()==""):
        if lines[start-1].startswith("@"): start -= 1
        else: break
    j = start
    while j < len(lines):
        print(f"{j:4d}: {lines[j]}")
        j += 1
        if j > start+3 and (lines[j].startswith("def ") or lines[j].startswith("@") and not lines[j-1].strip()):
            break
        if j-start > 120: break

# also dump any helper named *_dot or _mma or alloc in that file
print("\n===== other helper defs in translator_helpers.py =====")
for i, l in enumerate(lines):
    if l.startswith("def ") and any(k in l for k in ("dot","mma","alloc","tmem","tensor_mem","barrier","load")):
        print(f"  L{i}: {l}")

# 2) Source of the low-level blackwell builtins
print("\n===== blackwell low-level builtin sources =====")
bw = importlib.import_module("triton.experimental.gluon.language.nvidia.blackwell")
for nm in ['allocate_tensor_memory', 'tcgen05_mma', 'tcgen05_commit', 'tcgen05_copy']:
    o = getattr(bw, nm, None)
    if o is None:
        print(f"\n--- {nm}: MISSING ---"); continue
    try:
        src = inspect.getsource(o)
        print(f"\n--- {nm} ---")
        print(src[:2500])
    except Exception as e:
        print(f"\n--- {nm}: no source ({e}) ---")

# 3) mbarrier helpers (hopper)
print("\n===== mbarrier (hopper) sources =====")
mb = getattr(bw, 'mbarrier', None)
if mb:
    for nm in ['init', 'arrive', 'wait', 'expect', 'invalidate']:
        o = getattr(mb, nm, None)
        if o is None: continue
        try:
            print(f"\n--- mbarrier.{nm} ---")
            print(inspect.getsource(o)[:1200])
        except Exception as e:
            print(f"  mbarrier.{nm}: {e}")

# 4) tmem descriptor .load / .store methods
print("\n===== tensor_memory_descriptor methods =====")
tmd = getattr(bw, 'tensor_memory_descriptor', None)
if tmd:
    for nm in dir(tmd):
        if not nm.startswith('_') and callable(getattr(tmd, nm, None)):
            try:
                print(f"  tmd.{nm}{inspect.signature(getattr(tmd,nm))}")
            except Exception:
                print(f"  tmd.{nm} (sig N/A)")

print("\nDUMP DONE")
