"""Pin down the EXACT Gluon tcgen05 API in our installed triton 3.6.0 (main-branch
tutorials may differ). Print signatures/docs of the key ops, and locate any bundled
gluon example/test that USES tcgen05_mma so we can adapt a known-correct 3.6.0 kernel.
No banned substrings.
"""
import inspect, os, glob, importlib
import triton
print("triton", triton.__version__, "at", os.path.dirname(triton.__file__))

bw = importlib.import_module("triton.experimental.gluon.language.nvidia.blackwell")
gl = importlib.import_module("triton.experimental.gluon.language")
from triton.experimental import gluon

def sig(obj, name):
    try:
        print(f"  {name}{inspect.signature(obj)}")
    except Exception:
        d = (getattr(obj, '__doc__', '') or '').strip().splitlines()
        print(f"  {name}  (sig N/A) doc: {d[0] if d else '?'}")

print("\n=== blackwell tcgen05 / TMEM signatures ===")
for nm in ['tcgen05_mma','tcgen05_commit','tcgen05_copy','tcgen05_mma_scaled',
           'allocate_tensor_memory','tensor_memory_descriptor','TensorMemoryLayout','async_copy']:
    o = getattr(bw, nm, None)
    if o is not None: sig(o, nm)
    else: print(f"  {nm}: MISSING")

print("\n=== mbarrier ===")
mb = getattr(bw, 'mbarrier', None)
if mb:
    for nm in dir(mb):
        if not nm.startswith('_'):
            o=getattr(mb,nm)
            if callable(o): sig(o, f"mbarrier.{nm}")

print("\n=== gl core: allocate_shared_memory, layouts, jit ===")
for nm in ['allocate_shared_memory','NVMMASharedLayout','BlockedLayout','SwizzledSharedLayout']:
    o = getattr(gl, nm, None)
    if o is not None: sig(o, nm)
sig(gluon.jit, 'gluon.jit')

print("\n=== find bundled gluon examples that USE tcgen05_mma ===")
roots = [os.path.dirname(triton.__file__),
         os.path.dirname(os.path.dirname(triton.__file__))]
hits = []
for r in roots:
    for f in glob.glob(os.path.join(r, '**', '*.py'), recursive=True):
        try:
            txt = open(f).read()
        except Exception:
            continue
        if 'tcgen05_mma' in txt and 'gluon' in txt:
            hits.append((f, txt))
print(f"  {len(hits)} files use tcgen05_mma + gluon:")
for f, _ in hits[:12]:
    print("   ", f)

# dump the most GEMM-like example (smallest / named matmul/gemm/dot) for adaptation
if hits:
    pick = sorted(hits, key=lambda ft: (('matmul' not in ft[0] and 'gemm' not in ft[0] and 'mma' not in ft[0]), len(ft[1])))[0]
    print(f"\n=== EXAMPLE: {pick[0]} (first 200 lines) ===")
    print("\n".join(pick[1].splitlines()[:200]))
