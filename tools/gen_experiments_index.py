#!/usr/bin/env python3
"""Regenerate Part 2 of experiments/INDEX.md — the complete file listing.

The hand-written Part 1 is preserved verbatim; everything from the
'# Part 2' marker onward is replaced. Also asserts 100% coverage, so
INDEX.md can never silently drift back to documenting a quarter of the
directory (which is what it did for most of the competition).

    python3 tools/gen_experiments_index.py [--check]

--check exits non-zero if the file on disk is out of date.
"""
import collections, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
INDEX = EXP / "INDEX.md"
MARKER = "\n---\n\n# Part 2 — complete file listing (generated)\n"

GROUPS = collections.OrderedDict([
    ("cute_", "cute-DSL engine build (M1→M6) — research, never submitted"),
    ("gluon", "Gluon warp-specialization / tcgen05 investigation"),
    ("designb", "design-B inter-CTA overlap gates"),
    ("stage", "Gluon stage-0/1 overlap foundations"),
    ("fused_qr", "persistent fused one-shot kernel (became V10)"),
    ("v10_", "V10 / V10-sub5 routing variants"),
    ("deepdive", "profiling deep-dive tools (roofline, launch-gap)"),
    ("microbench_", "microbenchmarks (isolated component timing)"),
    ("profile_", "per-phase profilers"),
    ("probe_", "eval-environment probes"),
    ("cand_", "single-variable submission candidates"),
    ("cuda_", "raw-CUDA / PTX gates"),
    ("", "other / helpers"),
])


def describe(path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = re.match(r'\s*(?:#!.*\n)?\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', txt, re.S)
    if m:
        d = m.group(1)
    else:
        lines = []
        for line in txt.split("\n"):
            if line.startswith("#"):
                lines.append(line.lstrip("#").strip())
            elif lines or line.strip():
                break
        d = "\n".join(lines)
    d = " ".join(d.split())
    d = re.sub(r"^(experiments?/)?[\w./-]+\.py[\s:\u2014-]*", "", d)
    return (d[:150] + "\u2026") if len(d) > 150 else (d or "\u2014")


def build():
    files = sorted(EXP.rglob("*.py"))
    assigned, out = set(), []
    for prefix, title in GROUPS.items():
        sel = [p for p in files
               if p not in assigned and (p.name.startswith(prefix) or (prefix and prefix in p.name))]
        if not sel:
            continue
        assigned |= set(sel)
        out.append(f"\n### {title}  ({len(sel)} files)\n")
        out.append("| file | what it is |\n|---|---|")
        for p in sel:
            out.append(f"| `{p.relative_to(EXP)}` | {describe(p)} |")
    assert len(assigned) == len(files), f"coverage gap: {len(assigned)}/{len(files)}"
    body = "\n".join(out)
    return (MARKER + "\nEvery file in `experiments/`, with a one-line description taken from its own"
            " header. Generated, so\nit cannot drift out of coverage; the descriptions are only as"
            " good as each file's header comment.\nEntries already covered in Part 1 appear here"
            f" too.\n{body}\n\n<!-- total indexed: {len(assigned)} of {len(files)} -->\n")


def main():
    current = INDEX.read_text(encoding="utf-8")
    head = current.split(MARKER)[0]
    new = head + build()
    if "--check" in sys.argv:
        if new != current:
            print("experiments/INDEX.md is out of date — run tools/gen_experiments_index.py")
            return 1
        print("experiments/INDEX.md is up to date")
        return 0
    INDEX.write_text(new, encoding="utf-8")
    print(f"wrote {INDEX} ({len(new)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
