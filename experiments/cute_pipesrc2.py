"""Get the EXACT UMMA-completion->mbarrier-arrive primitive: PipelineUmmaAsync.producer_commit source
+ tcgen05 commit functions + cute.arch mma/tcgen05 names. This is the missing piece for the hand-rolled
manual-fill async ring (producer manual-arrives full; consumer needs UMMA-done->empty-arrive)."""
import inspect
import cutlass.pipeline as pl
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
P = lambda *a: print(*a, flush=True)

P("=== PipelineUmmaAsync FULL source (all methods) ===")
cls = pl.PipelineUmmaAsync
for m in ["create", "producer_acquire", "producer_commit", "consumer_wait", "consumer_release",
          "sync_object_array_full", "__init__"]:
    f = getattr(cls, m, None)
    if f is None:
        continue
    try:
        P(f"\n--- PipelineUmmaAsync.{m} ---")
        P(inspect.getsource(f)[:2200])
    except Exception as e:
        P(f"--- {m}: no src ({type(e).__name__}) ---")

P("\n\n=== tcgen05 module: commit / fence / arrive functions ===")
P("tcgen05 dir:", [n for n in dir(tcgen05) if not n.startswith("_")])
for n in dir(tcgen05):
    if any(k in n.lower() for k in ["commit", "fence", "arrive", "wait", "barrier"]):
        o = getattr(tcgen05, n)
        try:
            P(f"  tcgen05.{n}{inspect.signature(o)}")
        except Exception:
            P(f"  tcgen05.{n}: {type(o)}")

P("\n=== cute.arch: tcgen05/mma/mbarrier commit-arrive names ===")
import cutlass.cute.arch as arch
P([n for n in dir(arch) if any(k in n.lower() for k in
   ["commit", "mma", "tmem", "mbar", "arrive", "fence", "tcgen", "umma"])])

P("\n=== MbarrierArray / sync_object: the .arrive signature used by consumer_release ===")
for nm in ["MbarrierArray", "PipelineState", "make_pipeline_state"]:
    o = getattr(pl, nm, None)
    if o is None:
        continue
    P(f"\n-- {nm} --")
    try:
        P([m for m in dir(o) if not m.startswith("_")][:30])
        if hasattr(o, "arrive"):
            P("  .arrive sig:", inspect.signature(o.arrive))
    except Exception as e:
        P("  err", e)
print("DONE")
