#!/bin/bash
: '
=============================================================================
decollide.sh — end the macOS case-collision in factorjepa (and repair the
                backslash names its first version produced)
=============================================================================

THE PROBLEM
    origin/main carries two different model arms whose paths differ only by
    one letter s case:

        vjepa_2_1_vitG   2B ViT-G  (champion / default_backbone)
        vjepa_2_1_vitg   1B ViT-g  (scale-axis ablation)

    348 tracked paths existed in both spellings with DIFFERENT content. macOS
    APFS is case-insensitive, so it can only hold one of each pair: during
    checkout the second write lands on the first file. The result was a
    working tree that no "git reset --hard" could make clean, and a "git add ."
    that would have pushed one arm s bytes over the other s on GitHub.

THE FIX
    Rename the 1B arm to the suffixed form the codebase ALREADY uses elsewhere
    (src/utils/output_paths.py builds vjepa_2_1_vitg_1B / vjepa_2_1_vitG_2B):

        iter/**/vjepa_2_1_vitg/**            ->  iter/**/vjepa_2_1_vitg_1B/**
        **/scale_poc_vs_full_vjepa_2_1_vitg.{pdf,png}
                                             ->  ..._vjepa_2_1_vitg_1B.{pdf,png}

THE REPAIR (2026-09-21)
    The first version of this script built the new names with a bash
    ${var//pat/rep} whose replacement kept its backslashes, so commit ea6849e7
    renamed 408 paths to ".../metrics_watch\/vjepa_2_1_vitg_1B\/file" — a
    directory literally named "metrics_watch\" followed by "vjepa_2_1_vitg_1B\".
    Linux and macOS accept such names, Windows cannot check them out, and every
    glob in the pipeline misses them. This version rewrites any "\/" back to
    "/" (it also no longer produces them), so re-running it on the damaged tree
    yields the intended names. Renames are done with sed on the path string.

WHY THIS RUNS ON THE MAC, NOT THE GPU
    Everything happens in git objects via a TEMPORARY INDEX. No file is ever
    written to disk, so the case-insensitive filesystem is never asked to hold
    both spellings, and paths with odd characters never touch the working tree.

USAGE
    bash scripts/decollide.sh              # build + verify the commit, push NOTHING
    bash scripts/decollide.sh --push       # same, then push it to origin/main

    Afterwards, on every clone:
        git fetch origin && git reset --hard origin/main && git restore .
=============================================================================
'

set -euo pipefail

REPO="/Users/kapilwanaskar/Downloads/research_projects/factorjepa"
BRANCH="main"
OLD_DIR="vjepa_2_1_vitg"
NEW_DIR="vjepa_2_1_vitg_1B"
DO_PUSH=0

while [ $# -gt 0 ]; do
    case "$1" in
        --push) DO_PUSH=1 ;;
        *) echo "Unknown arg: $1"; echo "Usage: bash scripts/decollide.sh [--push]"; exit 1 ;;
    esac
    shift
done

cd "$REPO"

# tree_paths <tree-ish> — raw (un-quoted) path names, one per line
tree_paths() { git ls-tree -r -z --name-only "$1" | tr '\0' '\n'; }
count_collisions() { tree_paths "$1" | awk '{print tolower($0)}' | sort | uniq -d | wc -l | tr -d ' '; }
count_backslash() { tree_paths "$1" | awk '/\\/{n++} END{print n+0}'; }

echo "[1/6] fetch origin/$BRANCH"
git fetch origin "$BRANCH" --quiet
BASE=$(git rev-parse "origin/$BRANCH")
echo "      base = $BASE"

echo "[2/6] inspect the base tree"
before=$(count_collisions "$BASE")
bs_before=$(count_backslash "$BASE")
echo "      case-colliding paths : $before"
echo "      paths with a backslash: $bs_before"
if [ "$before" -eq 0 ] && [ "$bs_before" -eq 0 ]; then
    echo "      nothing to do — origin/$BRANCH is collision-free and backslash-free"
    exit 0
fi

# A temporary index means the real index and the working tree are never
# touched, so nothing is ever written to the case-insensitive filesystem.
TMP_INDEX=$(mktemp -t decollide-index)
trap 'rm -f "$TMP_INDEX"' EXIT
export GIT_INDEX_FILE="$TMP_INDEX"

echo "[3/6] load the base tree into a temporary index"
git read-tree "$BASE"

echo "[4/6] stage the renames"
n_dir=0
n_file=0
n_repair=0
# -z: raw paths (git C-quotes names with special characters otherwise).
while IFS= read -r -d '' entry; do
    meta="${entry%%	*}"
    path="${entry#*	}"
    mode=$(printf '%s' "$meta" | awk '{print $1}')
    sha=$(printf '%s' "$meta" | awk '{print $2}')

    new="$path"
    case "$path" in
        *\\*)                                   # repair: "\/" -> "/"
            new=$(printf '%s' "$new" | sed 's#\\/#/#g'); n_repair=$((n_repair + 1)) ;;
    esac
    case "$new" in
        */$OLD_DIR/*)                           # directory rename
            new=$(printf '%s' "$new" | sed "s#/$OLD_DIR/#/$NEW_DIR/#g"); n_dir=$((n_dir + 1)) ;;
        *_$OLD_DIR.pdf|*_$OLD_DIR.png)          # the two colliding file names
            new="${new%_$OLD_DIR.*}_$NEW_DIR.${new##*.}"; n_file=$((n_file + 1)) ;;
    esac
    [ "$new" = "$path" ] && continue

    git update-index --add --cacheinfo "$mode,$sha,$new"
    git update-index --force-remove "$path"
done < <(git ls-files -s -z)

echo "      $n_repair path(s) repaired  \\/  ->  /"
echo "      $n_dir path(s) under $OLD_DIR/  ->  $NEW_DIR/"
echo "      $n_file file(s) *_$OLD_DIR.{pdf,png}  ->  *_$NEW_DIR.{pdf,png}"

echo "[5/6] write the tree and verify it"
TREE=$(git write-tree)
after=$(count_collisions "$TREE")
bs_after=$(count_backslash "$TREE")
echo "      case-colliding paths after : $after"
echo "      paths with a backslash after: $bs_after"
if [ "$after" -ne 0 ] || [ "$bs_after" -ne 0 ]; then
    echo "      FATAL: the tree is still wrong. Nothing pushed."
    tree_paths "$TREE" | awk '/\\/' | head -5
    exit 1
fi

# File count must be identical: this is a pure rename, no content may be lost.
b_before=$(tree_paths "$BASE" | wc -l | tr -d ' ')
b_after=$(tree_paths "$TREE" | wc -l | tr -d ' ')
echo "      tracked files: $b_before -> $b_after"
if [ "$b_before" -ne "$b_after" ]; then
    echo "      FATAL: file count changed — a rename collided. Nothing pushed."
    exit 1
fi

MSG="fix(paths): repair the backslash directory names produced by the 1B-arm rename

Commit ea6849e7 renamed the 1B arm to vjepa_2_1_vitg_1B to end the macOS
case collision with vjepa_2_1_vitG, but built the new names with a bash
substitution that kept its backslashes: 408 paths became
.../metrics_watch\\/vjepa_2_1_vitg_1B\\/file, i.e. directories literally named
metrics_watch\\ and vjepa_2_1_vitg_1B\\. Linux and macOS tolerate the names,
Windows cannot check them out, and the pipeline globs miss them.

Rewrite every \\/ back to /. Pure rename: no blob changed, file count
unchanged, collisions stay at 0, backslash paths $bs_before -> 0."

COMMIT=$(git commit-tree "$TREE" -p "$BASE" -m "$MSG")
echo "[6/6] commit built: $COMMIT"

if [ "$DO_PUSH" -eq 0 ]; then
    echo
    echo "Nothing pushed (no --push). To inspect it:"
    echo "    git diff --stat --name-status $BASE $COMMIT | head -20"
    echo "To publish:"
    echo "    bash scripts/decollide.sh --push"
    exit 0
fi

unset GIT_INDEX_FILE
echo "      pushing $COMMIT -> origin/$BRANCH"
git push origin "$COMMIT:refs/heads/$BRANCH"

echo
echo "Pushed. Now bring this clone in line:"
echo "    git fetch origin && git reset --hard origin/$BRANCH && git restore ."
