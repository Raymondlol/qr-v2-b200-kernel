#!/usr/bin/env bash
# Prune the branch list before making this repo public.
#
# Context: the working repo accumulated 35 branches, most with machine-generated
# names (claude/bold-shannon-6a7521, codex/..., etc). That is the first thing a
# visitor sees in the GitHub branch dropdown, and it reads as mess rather than
# history. This keeps the handful of branches that carry a story and deletes the
# rest.
#
# NOTHING IS LOST. Every branch was tagged `archive/<branch-name>` first, so each
# tip stays reachable forever:
#     git tag -l 'archive/*'            # list them
#     git checkout archive/<name>       # visit one
#     git branch <name> archive/<name>  # bring one back
#
# This script is NOT run automatically, because deleting branches is destructive
# and because branches currently checked out in a git worktree cannot be deleted
# (and deleting one would break whatever session is using it). Review, then run.
#
#     bash tools/prune_branches.sh          # dry run: show what would go
#     bash tools/prune_branches.sh --apply  # actually delete
set -euo pipefail
cd "$(dirname "$0")/.."

# Branches worth keeping: each is referenced from the docs as a research artifact.
KEEP=(
  main                 # the public branch
  persistent-engine    # V10's fused one-shot kernel (docs/JOURNAL.md)
  profiling-deepdive   # the roofline + launch-gap deep dive (docs/PROFILING.md)
  fp8-fp4-attack       # fp8/fp4/fp16x3 precision investigation (docs/DEAD_ENDS.md)
  gluon-tcgen05        # parked Gluon tcgen05 investigation (docs/DEAD_ENDS.md)
  cutedsl-engine       # the cute-DSL engine build (docs/HANDOVER_CUTEDSL_M6.md)
  qr-py-v1             # the sibling qr_py board, kept as a research artifact
)

APPLY=0; [[ "${1:-}" == "--apply" ]] && APPLY=1

# bash 3.2 (macOS default) has no mapfile -- use a newline-delimited string.
HELD="$(git worktree list --porcelain | sed -n 's|^branch refs/heads/||p')"
is_kept() { for x in "${KEEP[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }
is_held() { printf '%s\n' "$HELD" | grep -qxF "$1"; }

kept=0; held=0; gone=0
while read -r b; do
  if   is_kept "$b"; then printf '  keep             %s\n' "$b"; kept=$((kept+1))
  elif is_held "$b"; then printf '  SKIP (worktree)  %s\n' "$b"; held=$((held+1))
  else
    git rev-parse -q --verify "refs/tags/archive/$b" >/dev/null || {
      printf '  REFUSE (no archive/ tag) %s\n' "$b"; continue; }
    if (( APPLY )); then git branch -D "$b" >/dev/null; printf '  deleted   %s\n' "$b"
    else printf '  would delete %s  (preserved at archive/%s)\n' "$b" "$b"; fi
    gone=$((gone+1))
  fi
done < <(git for-each-ref --format='%(refname:short)' refs/heads/)

echo
if (( APPLY )); then echo "keep=$kept  worktree-held=$held  deleted=$gone"
else echo "keep=$kept  worktree-held=$held  would-delete=$gone"; fi
(( APPLY )) || echo "dry run — re-run with --apply to delete. Tips stay at archive/<name> either way."
