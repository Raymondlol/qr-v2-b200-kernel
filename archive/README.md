# archive/ — historical docs (and, later, experiments)

These files were moved out of the live working set on **2026-06-28** to reduce the
history burden, **not deleted** — full content is preserved here and in git history
(see tag `pre-trim-snapshot`). Every conclusion that still matters has already been
distilled into the canonical live docs.

> **Moved out 2026-09-11:** `archive/leader_top3/` is now `docs/leader_top3/`. It was the *newest*
> and most analytically useful directory in the repo — a post-deadline teardown of the top-3
> submissions — and filing it under "superseded" actively hid it. That was a filing error, not a
> judgement about the content.

## Where the live equivalents are
- **Latest cute-DSL state / entry point** → `docs/HANDOVER_CUTEDSL_M6.md`
- **What NOT to re-explore (all dead ends)** → `docs/DEAD_ENDS.md`
- **Score ledger / how we got here** → `docs/JOURNAL.md`
- **Profiler record** → `docs/PROFILING.md`
- **Engine design spine** → `docs/FA4_BLUEPRINT_FOR_QR.md`
- **cute-DSL 4.5.2 API reference** → `reference/CUTEDSL_CHEATSHEET.md`

## archive/docs/ — what's here and why it was archived
| File | Why archived |
|---|---|
| `HANDOVER_CUTEDSL.md` | Oldest cute-DSL handover; superseded by `_M3` → `_M6`. |
| `HANDOVER_CUTEDSL_M3.md` | M3c-0 handover; its next-steps are done and surpassed by M6. |
| `HANDOVER_NEXT_SESSION.md` | 2026-06-26 Gluon-era session log; conclusions are in `DEAD_ENDS.md`. |
| `NEXT_SESSION_PROMPT.md` | Autonomous plan for that same session; superseded. |
| `STAGE1_PROGRESS.md` | Gluon design-A/B overlap investigation; measured-dead, captured in `DEAD_ENDS.md`. |
| `SMEM_PANEL_PLAN.md` | Stage-A smem panel plan; own banner says NO-GO. |
| `FP8_SESSION_PROGRESS.md` | fp8/fp4 + fp16x3 session; results folded into `DEAD_ENDS.md`. |
| `NEXT_STEPS.md` | Old forward-plan; its overlap thesis was measured-dead (see `DEAD_ENDS.md`). |

Internal cross-links *inside* these archived files may point at the old `docs/`
locations — that's expected; they're frozen historical records. The live surfaces
(`CLAUDE.md`, `README.md`, the kept `docs/`) were updated to point here.
