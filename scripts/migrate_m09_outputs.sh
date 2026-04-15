#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Phase 3 of #49: migrate outputs/.../m09_pretrain/{ablation,lambda*,explora,surgery}
# to new per-module dirs: m09a_pretrain/, m09b_explora/, m09c_surgery/.
#
# Idempotent: skip target if already exists. Reversible: uses `mv`, not `rm`.
# Only needed once per machine; safe to re-run.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

moved=0
skipped=0
missing=0

migrate_one() {
    local src="$1"
    local dst="$2"
    if [[ ! -e "$src" ]]; then
        missing=$((missing + 1))
        return 0
    fi
    if [[ -e "$dst" ]]; then
        echo "  SKIP: $dst already exists (source: $src)"
        skipped=$((skipped + 1))
        return 0
    fi
    mkdir -p "$(dirname "$dst")"
    echo "  MOVE: $src → $dst"
    mv "$src" "$dst"
    moved=$((moved + 1))
}

for mode in sanity poc full; do
    base="outputs/${mode}/m09_pretrain"
    if [[ ! -d "$base" ]]; then
        continue
    fi

    # m09a_pretrain/ absorbs lambda* and ablation
    for sub in "$base"/lambda* "$base"/ablation; do
        if [[ ! -e "$sub" ]]; then
            continue
        fi
        migrate_one "$sub" "outputs/${mode}/m09a_pretrain/$(basename "$sub")"
    done

    # m09b_explora/ absorbs explora
    migrate_one "$base/explora" "outputs/${mode}/m09b_explora"

    # m09c_surgery/ absorbs surgery
    migrate_one "$base/surgery" "outputs/${mode}/m09c_surgery"

    # Remove m09_pretrain/ if it's now empty
    if [[ -d "$base" ]] && [[ -z "$(ls -A "$base" 2>/dev/null)" ]]; then
        echo "  RMDIR: empty $base"
        rmdir "$base"
    fi
done

echo ""
echo "Migration summary: moved=${moved}, skipped=${skipped}, missing=${missing}"
