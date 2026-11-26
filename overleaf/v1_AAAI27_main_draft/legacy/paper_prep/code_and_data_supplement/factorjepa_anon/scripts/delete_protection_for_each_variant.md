# 🛡️ Delete Protection — Cache-Policy Contract for Shell Orchestrators

> **The MUST condition:** ⚠️ The `*.sh` script **MUST collect ALL cache-policy decisions UP FRONT** — *before any compute starts*. The Python `input()` prompt is a fallback for direct `.py` invocation, NOT a per-stage gate during overnight chains.

---

## 🎯 Why this exists

🐛 **Past failure (iter12 #74, 2026-04-22):** a blanket `rm -rf outputs/full/m05*` nuked hidden `.m05_checkpoint*.npz` files mid-overnight → 9,297 cached embeddings vaporised → +3.2 h re-compute + ~$2.56 GPU sunk.

🛡️ **The fix:** every destructive delete now lives behind `utils.cache_policy.guarded_delete()` in the `.py` layer, gated by a `--cache-policy {1,2}` CLI flag. The shell layer's job is to **decide once, propagate everywhere**.

---

## 🔒 Three-layer protocol

### 1️⃣ Variant detection (e.g. `run_train.sh:71-79`, `run_paired_eval_10k.sh`)

🔍 Build a `VARIANTS=()` array by checking each candidate `student_encoder.pt` (or equivalent artifact) — **only variants whose checkpoints exist get evaluated**. Skips dead branches before prompting.

### 2️⃣ `_check_and_prompt_any` helper (e.g. `run_train.sh:44-71`, ~lines 102-126 in legacy paired_eval)

🔄 For each `(call-site × variant)` combo, check if any cached artifact exists at known paths. If yes:

| Condition | Decision |
|---|---|
| 🌍 `CACHE_POLICY_ALL=1\|2` env-var set | ✅ Use that, skip prompt entirely |
| 🤖 Non-TTY (`! -t 0`) — overnight `tmux`, CI, ssh -T | 🔇 Silent default to `1` (keep) |
| 🧑 Interactive TTY | 💬 `read -p "$key cache at $found [1=keep / 2=recompute] (Enter=1): "` → store in `POLICY[$key]` |

🚫 Missing caches → policy defaults to `1` (keep — there's nothing to delete anyway).

### 3️⃣ Dependency propagation (e.g. `run_paired_eval_10k.sh:154-162`)

🔗 If user chose **recompute** upstream (e.g. `m05`), **auto-invalidate downstream** (`m06`, `m08b`) — stale embeddings → stale metrics. Then 📋 print the full plan + ask `"Proceed with this plan? [y/N]"` **once**, before the chain runs.

---

## 🌙 What the chain looks like at run-time

🚀 Once gathered, the chain runs **unattended overnight**, passing the policy explicitly:

```bash
python -u src/m05_vjepa_embed.py --FULL ... --cache-policy "${POLICY[m05_$v]}"
python -u src/m06_faiss_metrics.py --FULL ... --cache-policy "${POLICY[m06_$v]}"
python -u src/m08b_compare.py --FULL ... --cache-policy "${POLICY[m08b_$v]}"
```

🔇 The `.py`-level `input()` prompt **never fires** because `resolve_cache_policy_interactive` (in `src/utils/cache_policy.py`) sees `--cache-policy` already on `argv` and short-circuits.

---

## ✅ Bypasses (for unattended runs)

```bash
# 🟢 Keep all caches, no prompts
CACHE_POLICY_ALL=1 ./scripts/legacy2/run_train.sh <yaml1> <yaml2>

# 🔴 Recompute everything, no prompts
CACHE_POLICY_ALL=2 ./scripts/legacy2/run_train.sh <yaml1> <yaml2>

# 🤖 Non-TTY auto-default (e.g., piping or running detached)
nohup ./scripts/legacy2/run_train.sh <yaml1> <yaml2> > overnight.log 2>&1 &
```

---

## 🚨 Anti-pattern (what NOT to do)

❌ **DON'T** call `python -u src/m*.py` from a shell loop **without** `--cache-policy` — each invocation will hit the `.py`-level `input()` prompt:
- 🧑 In a TTY: blocks waiting for keypress per stage (interrupts overnight).
- 🤖 In non-TTY: silently defaults to `1`, but the operator can't see *which* stages were prompted (logs get swallowed by wandb chatter — see `logs/run_src_probe_v1.log:13,44`).

✅ **DO** gather upfront → `declare -A POLICY` → `_check_and_prompt` → pass `--cache-policy "$P"` to every `.py` call.

---

## 📚 Reference implementations

| Script | Pattern |
|---|---|
| `scripts/legacy2/run_train.sh:43-89, 124, 133, 152` | 🥇 Gold-standard reference — per-variant prompt + per-call propagation |
| `scripts/legacy2/run_paired_eval_10k.sh` | 🏆 3-tier idempotency gates (G1/G2/G3) + dependency propagation (m05→m06→m08b) |
| `scripts/lib/common.sh::prompt_cache` | 🧰 Shared helper if you don't want to copy `_check_and_prompt` |

---

> 🔑 **TL;DR:** decisions UPFRONT, compute UNATTENDED. The `.py`-level prompt is a **safety net**, not a UX feature.
