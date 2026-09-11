# milestones/ — every scored version, in order

One file per rung of the ladder, named `NN_<name>_official<score>us.py`. The number in the filename
is the **official GPU MODE leaderboard geomean** that file scored — not a lab number, not an
estimate. Sorted `ls` order is the optimization history.

| file | official geomean | id | what it added | provenance |
|---|---|---|---|---|
| `01_eager_official123203us.py` | 123,203 µs | — | correct blocked compact-WY, all eager PyTorch | scored |
| `02_triton_panel_official43576us.py` | 43,576 µs | — | fused Triton sub-panel + `bmm` trailing | scored |
| `03_tf32x3_2level_official14760us.py` | 14,760 µs | — | tf32x3 emulated-FP32; two-level blocking; T via one triangular solve | scored |
| `04_panel_warps_official8580us.py` | 8,580 µs | — | `num_warps` by tile size (n=1024: 54 ms → 10.9 ms) | scored |
| `05_V4_route2048_official7788us.py` | 7,788 µs | — | route n=2048 to the custom one-CTA panel | scored — ⚠️ see note 1 |
| `06_V5_engine_official5915us.py` | 5,915 µs | 834868 | "engine attack": gram precision routing, adaptive ib, glue elimination, panel nw=8 | scored |
| `07_V7_implicitV_official6145us_REGRESSION.py` | 6,145 µs | — | implicit-V apply. **A regression** (+3.9% vs V5). Kept deliberately. | scored |
| `08_V8_gluefuse_official5898us.py` | 5,898 µs | — | M-builder fusion only | scored |
| `09_V9_gluefuse2_official5791us.py` | 5,791 µs | 838173 | full glue fusion (`_build_M`/`_build_Mt`/`_build_V`), bit-identical output | scored |
| `10_V10_fused_official5343us.py` | 5,343 µs | 838292 | one-shot fused Triton kernel for n≤512 | scored — ⚠️ see note 2 |
| `11_V10sub5_fused32_official4247us_BEST.py` | **4,247 µs** | 840028 | + route n≤64/B>16 to the fused kernel (b=20/n=32: 318 → 30 µs) | scored — **= `submission.py`** |

Every file except `11_..._BEST.py` carries a short `# === milestone:` banner giving its official
score and submission id. `11_..._BEST.py` deliberately has **no** banner, so that it stays
**byte-identical to `submission.py`** — a claim you can check with `diff`, which is worth more than
formatting consistency.

## Notes — read these before trusting a filename
1. **`05_V4_...7788us`** was committed as `m5_route2048_est8100us.py`, an *estimate* made before it
   was submitted. It later scored 7,788 µs officially. The rename reflects the official result; I
   did not re-verify byte-for-byte that the submitted file was this exact snapshot, so treat the
   7,788 attribution as high-confidence but not proven.
2. **`10_V10_...5343us`** was missing from this directory until 2026-09-11 — `README.md` and
   `docs/JOURNAL.md` both pointed at a `milestones/submissionV10_fused_modal4913.py` that had only
   ever been committed on branch `persistent-engine`. Recovered from there. (Its old filename
   carried its *Modal* number, 4913, not its official 5,343.)
3. **V6 (fp16x3, official 5,997 µs — a wash) has no milestone file.** It was never snapshotted here.
   The candidate that became it is `experiments/cand_fp16x3.py`.
4. **Header comments were normalized on 2026-09-11.** Several files carried stale headers (the
   worst claimed a geomean of "~10800us" on the 4,247 µs entry). The **executable code is
   unchanged**; only leading `#` comments were rewritten. `11_...BEST.py` is byte-identical to
   `submission.py`.
5. Three of these files were previously named with **Modal lab** numbers rather than official ones
   (`...modal5870`, `...modal5402`, `...modal5298`) — including V7, whose filename said 5,870 while
   the leaderboard said 6,145 and called it a regression. Given that this project's central lesson
   is *lab numbers are not scores*, that naming was indefensible. Fixed.

## Provenance of the official numbers
These are leaderboard readings, recorded by hand at the time, identified by submission id where I
still have it. They are **self-reported**: this repo contains no archived leaderboard response.
The lab numbers, by contrast, all have raw artifacts in `results/`. If you are evaluating this repo
and the official figures matter to you, treat them as claims backed by submission ids on a public
board, not as evidence in hand.
