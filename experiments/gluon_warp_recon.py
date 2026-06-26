"""Recon the warp-level primitives for HAND-ROLLED warp specialization in stock Gluon 3.6.0:
gl.warp_id / thread_id, ways to confine a layout to a warp subset, and any bundled gluon
kernel that does manual per-warp roles (attention/persistent/ws). No banned substrings.
"""
import os, glob, inspect, importlib, triton
from triton.experimental.gluon import language as gl
print("triton", triton.__version__)

print("\n=== gl namespace: warp/thread/lane/specialize primitives ===")
cands = [a for a in dir(gl) if any(s in a.lower() for s in
         ('warp', 'thread', 'lane', 'special', 'task', 'partition', 'async', 'cluster'))]
print(" ", cands)
for nm in ['warp_id', 'thread_id', 'num_warps', 'num_threads', 'program_id']:
    o = getattr(gl, nm, None)
    if o is not None:
        try: print(f"  gl.{nm}{inspect.signature(o)}")
        except Exception: print(f"  gl.{nm} present (sig N/A)")
    else:
        print(f"  gl.{nm}: MISSING")

# nvidia / blackwell extra
for modn in ['triton.experimental.gluon.language.nvidia',
             'triton.experimental.gluon.language.nvidia.blackwell',
             'triton.experimental.gluon.language.nvidia.hopper',
             'triton.experimental.gluon.language.nvidia.ampere']:
    try:
        m = importlib.import_module(modn)
        ks = [a for a in dir(m) if any(s in a.lower() for s in
              ('warp','thread','special','task','partition','mbarrier','named_barrier','barrier'))]
        print(f"  {modn.split('.')[-1]}: {ks}")
    except Exception as e:
        print(f"  {modn}: {type(e).__name__}")

print("\n=== search installed triton for warp-spec / per-warp-role idioms ===")
root = os.path.dirname(triton.__file__)
pats = ['warp_id(', 'warp_specialize', 'async_task', 'named_barrier', 'thread_id(', 'mma_warp', 'producer', 'consumer']
hits = {}
for f in glob.glob(os.path.join(root, '**', '*.py'), recursive=True):
    try: txt = open(f).read()
    except Exception: continue
    for p in pats:
        if p in txt:
            hits.setdefault(p, []).append(f.replace(root, '...'))
for p, fs in hits.items():
    print(f"  '{p}' in {len(fs)} files: {fs[:5]}")

print("\n=== look for gluon attention/matmul/persistent examples (likely WS) ===")
for f in glob.glob(os.path.join(root, '**', '*.py'), recursive=True):
    bn = os.path.basename(f).lower()
    if any(k in bn for k in ('attention', 'persistent', 'warp', 'matmul', 'flash')) and 'gluon' in open(f, errors='ignore').read():
        print("   ", f.replace(root, '...'))

print("\nDONE")
