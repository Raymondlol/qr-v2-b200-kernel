"""Print the cutlass.pipeline PipelineAsyncUmma / PipelineAsync producer/consumer source on the image,
to learn the EXACT arrive mechanism (why manual-st.shared producer_commit deadlocks vs cp.async)."""
import inspect
import cutlass.pipeline as pl
P = lambda *a: print(*a, flush=True)

P("=== pipeline classes present ===")
P([n for n in dir(pl) if "Pipeline" in n or "Cooperative" in n or "State" in n])

for cname in ["PipelineAsyncUmma", "PipelineAsync", "PipelineUmmaAsync", "PipelineTmaUmma"]:
    cls = getattr(pl, cname, None)
    if cls is None:
        P(f"\n### {cname}: ABSENT"); continue
    P(f"\n############### {cname} ###############")
    P("methods:", [m for m in dir(cls) if not m.startswith("_")])
    for m in ["create", "make_participants", "producer_acquire", "producer_commit",
              "producer_try_acquire", "consumer_wait", "consumer_release", "producer_tail"]:
        f = getattr(cls, m, None)
        if f is None:
            continue
        try:
            src = inspect.getsource(f)
            P(f"\n--- {cname}.{m} ---")
            P(src[:1800])
        except Exception as e:
            try:
                P(f"\n--- {cname}.{m} sig: {inspect.signature(f)} (no src: {type(e).__name__}) ---")
            except Exception:
                P(f"\n--- {cname}.{m}: <no src/sig> ---")
print("DONE")
