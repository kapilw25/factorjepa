# 🔀 `run_factor_prep_parallel.sh` — remaining parity gaps

> **Context** — `scripts/run_factor_prep_parallel.sh` is the parallel mirror of `scripts/run_factor_prep.sh` (N parallel m10 workers → merge → m11). It exists, runs, and produces the same outputs as the serial version at ~2× speedup with N=4. **This doc tracks the UX-parity items still missing vs the serial wrapper** so they can be picked up next session without re-deriving the diff.

---

## ✅ Already shipped (do not redo)

| # | Item | Where landed |
|---|---|---|
| 1 | Renamed `run_m10_parallel.sh` → `run_factor_prep_parallel.sh` | scripts/ |
| 2 | Mode flag passthrough (`--SANITY` / `--POC` / `--FULL`) — 3rd positional arg | `run_factor_prep_parallel.sh:39,49-54` |
| 3 | `manifest.json` existence check (FATAL if missing) | `run_factor_prep_parallel.sh:67-70` |
| 4 | m11 chaining (Step 5/5 — auto-runs after merge) | `run_factor_prep_parallel.sh:146-167` |
| 5 | Per-tar drop logging (`_read_one_tar` emits `dropped=N (missing_part=X, empty_mp4=Y)` per tar) | `src/utils/data_download.py` (FIX-27a) |
| 6 | `curate_verify` top-N restricted to renderable video_ids | `src/utils/curate_verify.py` (FIX-27b) |

---

## 🔴 Three remaining gaps (~40 LoC total, all in shell — no Python/merge changes)

### 1️⃣ `TRAIN_SUBSET` env-var support · ~10 LoC

**Why** — serial has it (lines 73, 82-86) so users can filter the clip universe (e.g. only chennai walking). Parallel has no equivalent → can't run "parallel m10 over a subset slice".

**Patch** — in `scripts/run_factor_prep_parallel.sh` after `LOCAL_DATA=...`:

```bash
TRAIN_SUBSET="${TRAIN_SUBSET:-}"
if [ -n "$TRAIN_SUBSET" ] && [ ! -e "$TRAIN_SUBSET" ]; then
    echo "FATAL: TRAIN_SUBSET=$TRAIN_SUBSET set but file not found." >&2
    exit 3
fi
```

…then thread it into the splitter call (Step 1/5):

```bash
SPLITTER_INPUT="${TRAIN_SUBSET:-${LOCAL_DATA}/manifest.json}"
python -u src/utils/m10_split_subset.py \
    --manifest "$SPLITTER_INPUT" \
    --existing-segments "${CANONICAL_DIR}/segments.json" \
    --n-workers "$N_WORKERS" \
    --out-dir "$SUBSET_DIR"
```

🛠️ **Companion edit** — `src/utils/m10_split_subset.py` accepts both schemas (`manifest.json` has `"saved_keys"`; `subset.json` has `"clip_keys"`):

```python
# In main() after json.load:
if "saved_keys" in manifest:
    all_keys = manifest["saved_keys"]                # manifest.json
elif "clip_keys" in manifest:
    all_keys = manifest["clip_keys"]                 # subset JSON (TRAIN_SUBSET)
else:
    print(f"FATAL: input has neither 'saved_keys' nor 'clip_keys'")
    sys.exit(2)
```

### 2️⃣ Header banner — show `TRAIN_SUBSET` status · ~4 LoC

**Why** — serial banner reports filter status; parallel banner doesn't, so user can't tell at a glance whether subset filtering is active.

**Patch** — extend the existing banner block (around line 80):

```bash
echo "  LOCAL_DATA:    $LOCAL_DATA"
if [ -n "$TRAIN_SUBSET" ]; then
    echo "  TRAIN_SUBSET:  $TRAIN_SUBSET  (filtering clip set)"
else
    echo "  TRAIN_SUBSET:  <unset>  (using \$LOCAL_DATA/manifest.json — all clips)"
fi
echo "  CANONICAL_DIR: $CANONICAL_DIR"
```

### 3️⃣ Interactive cache-policy prompt · ~20 LoC

**Why** — without `CACHE_POLICY_ALL` env var, parallel silently defaults to `2` (which means "wipe worker scratch dirs"). On a re-run after partial-kill, this **loses up to ~25-30 min of partial worker progress** with no warning. Serial has `_check_and_prompt` to guard against this; parallel doesn't.

**Patch** — port the function from `scripts/run_factor_prep.sh:99-131` and wire it before the worker spawn:

```bash
declare -A POLICY
_check_and_prompt() {
    local key="$1"; shift
    local found=""
    for path in "$@"; do
        local hit=$(compgen -G "$path" 2>/dev/null | head -n1)
        if [ -n "$hit" ]; then found="$hit"; break; fi
    done
    if [ -z "$found" ]; then POLICY[$key]=1; return; fi
    if [ -n "${CACHE_POLICY_ALL:-}" ]; then
        POLICY[$key]=$CACHE_POLICY_ALL
        echo "  $key: cache at $found -> policy=${POLICY[$key]} (CACHE_POLICY_ALL)"
        return
    fi
    if [ ! -t 0 ]; then
        POLICY[$key]=1
        echo "  $key: cache at $found -> policy=1 (non-TTY default)"
        return
    fi
    local ans
    read -p "  $key cache at $found [1=keep / 2=recompute] (Enter=1): " ans
    case "${ans:-1}" in
        2|recompute) POLICY[$key]=2 ;;
        *)           POLICY[$key]=1 ;;
    esac
}

# Probe each worker scratch dir + canonical (resume-anchor heuristic):
for i in $(seq 0 $((N_WORKERS - 1))); do
    _check_and_prompt "worker_$i" "${LOCAL_DATA}/m10_sam_segment_w${i}/*"
done
_check_and_prompt "canonical" "${CANONICAL_DIR}/segments.json"
# Replace the unconditional CACHE_POLICY="${CACHE_POLICY_ALL:-2}" with the per-key value
CACHE_POLICY="${POLICY[worker_0]:-1}"   # workers share policy in practice
```

---

## ⚠️ Resume-semantics caveat (document, don't auto-fix)

| Scenario | Behavior with `CACHE_POLICY_ALL=2` | Behavior with `CACHE_POLICY_ALL=1` |
|---|---|---|
| First run, empty disk | wipe worker scratch (no-op since empty) | same — empty scratch |
| **Re-run after killed partial** | **❌ wipes worker scratch → restarts each worker from clip 0** of its 2370-clip slice (loses ~25-30 min) | ✅ each worker resumes from its own checkpoint |
| Want fresh restart | `CACHE_POLICY_ALL=2` + manually `rm canonical_dir/segments.json` (canonical is NEVER wiped by the parallel script) | n/a |

📝 **Document this in the file's docstring** alongside gap #3's prompt — "if you killed a partial run, set `CACHE_POLICY_ALL=1` to resume".

---

## 🧪 SANITY validation (TODO #182, still pending)

After applying gaps #1-#3:

```bash
# ~3 min: 2 workers × ~10 SANITY clips each + merge + m11 streaming
CACHE_POLICY_ALL=2 ./scripts/run_factor_prep_parallel.sh \
    configs/train/surgery_3stage_DI.yaml 2 --SANITY \
    2>&1 | tee logs/factor_sanity_parallel_v1.log
```

✅ Pass criteria:
1. Banner shows `TRAIN_SUBSET: <unset>` line (gap #2)
2. Both worker logs end with `Done: N clips segmented`
3. Merge log: `merged segments.json: ~540 total clips` (520 prior + ~20 new SANITY)
4. `summary.json["quality_gate"] == "PASS"`
5. m11 finishes; `D_L/*.npy` files materialized

---

## 📋 TODO state

| # | Task | Status |
|---|---|---|
| (deleted) | SANITY parallel orchestrator smoke | was #182 — re-create when ready to land gaps #1-#3 |

➡️ **Pickup recipe** — when ready: re-create the `#182 SANITY parallel orchestrator` task, apply the 3 patches above (~40 LoC), then run the SANITY block to validate.
