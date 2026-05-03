---
name: Load-bearing CLAUDE.md conventions (condensed)
description: The handful of CLAUDE.md rules that bite hardest if violated; copy this into your working memory
type: project
---

# Conventions — the rules that bite

> Full text: `src/CLAUDE.md` + `iter/iter13_motion_probe_eval/plan_training.md`. The list below is the load-bearing subset that will cause the hardest regressions if violated.

## 🚨 Top 5 rules (violate any → real incident)

### 1. NO HARDCODED VALUES IN PYTHON

Every numeric default lives in YAML (`configs/`), never as a Python literal or `argparse default=<scalar>`. CLI flags use `default=None` so they fall through to YAML when unspecified.

❌ **Banned**:
```python
parser.add_argument("--early-stop-patience", type=int, default=5)         # 5 is hardcoded
patience = 5  # hardcoded
patience = args.early_stop_patience or 5  # 5 is hardcoded
```

✅ **Required**:
```python
# In YAML (configs/pipeline.yaml):
action_probe_train:
  early_stop_patience: 5

# In Python:
parser.add_argument("--early-stop-patience", type=int, default=None)      # None falls through
patience = (args.early_stop_patience
            if args.early_stop_patience is not None
            else get_pipeline_config()["action_probe_train"]["early_stop_patience"])
```

User explicitly bites on this. See [feedback_no_hardcoded_defaults.md](feedback_no_hardcoded_defaults.md).

### 2. NO `.get(key, <non-None>)` ON YAML SUBSCRIPTS

YAML keys must FAIL LOUD if missing. `.get(key, None)` is OK (matches argparse's None-default pattern); `.get(key, <scalar>)` is BANNED — it silently substitutes a default and you lose the YAML schema as the single source of truth.

❌ **Banned**: `cfg["optimization"].get("lr", 1e-5)`
✅ **Required**: `cfg["optimization"]["lr"]` — KeyError surfaces missing schema immediately

Preflight guard: B6 (AST-tracks vars from `yaml.safe_load` / `load_merged_config` and flags `.get` on them). Whitelist: `os`, `kwargs`, `request`, `headers`, `environ`, `response`.

### 3. FAIL HARD — no silent failures

- ❌ `try: ... except: pass`
- ❌ `try: ... except: continue` (without `raise` AND no progress check after)
- ❌ `try: ... except: return None` (without raising)
- ❌ Shell `|| true`, `|| continue`, `|| :`
- ❌ `WARNING:` without `exit` in shell

✅ Per CLAUDE.md "Silent failures = garbage metrics". When you have a defensive try/except, the except block MUST `raise` after the failure-mode handling, or the calling loop must check progress and raise hard if no progress was made (Bug B's post-train guard pattern).

Preflight: B43 (silent except in `.py`), B44 (shell silent-defaults).

### 4. CACHE POLICY GATING

Every destructive `.py` (m04/m04d/m05/m05b/m05c/m06/m08/m08b/m09a/m09b/m09c/m10/m11) registers `--cache-policy {1,2}` via `utils.cache_policy.add_cache_policy_arg()`. Shells stay THIN — pass `--cache-policy "$P_X"` to every call. NEVER use shell-level `rm -rf` (Bug-#74-class incidents).

Preflight: B64 (every `python -u m*.py` call must have `--cache-policy`).

### 5. SEMICOLONS NOT && BETWEEN INDEPENDENT LONG RUNS

Per CLAUDE.md "OVERNIGHT CHAINS — `;` NOT `&&`":

✅ `./train.sh ; ./eval.sh` — eval runs even if train fails (independent artifacts)
❌ `./train.sh && ./eval.sh` — single train failure kills 8 h of overnight queue

Reserve `&&` ONLY for cases where command 2 literally cannot run without command 1's output. Both `run_probe_train.sh pretrain` and `run_probe_train.sh surgery_*` are independent — chain with `;`.

## 🎯 Architectural conventions

### Module naming (`src/m*.py`)

- Numeric prefix `m{NN}` avoids import collision (m04 = VLM tag, m05 = embed, m06 = retrieval, m09 = continual SSL, m10 = SAM segment, m11 = factor datasets, m12 = action labels, m13 = probe train)
- **Suffixed variants** (`m04b`, `m09a/b/c`) signal related-but-isolated modules — same numeric stage, different technique
- **No cross-imports between m*.py files** (rule 32). Shared logic goes to `src/utils/`.

### `src/utils/training.py` (#49 contract)

Every function MUST be technique-agnostic. ZERO `if args.explora` / `if args.surgery` / `if cfg["technique"]` branches. Mode-specific behavior is configured via explicit parameters (`init_params=None`, `drift_cfg=None`, `explora_enabled=False`).

When adding a function that's specific to one technique (multi-task probe, ExPLoRA LoRA injection, etc.), put it in a separate utils file (`utils/multi_task_loss.py`, etc.) — keep `training.py` clean.

### Output organization

Per CLAUDE.md "Organize outputs into clean subdirectories. Each logical group owns its own dir + JSON. Never flatten structures":

```
outputs/<mode>/
├── probe_action/             # probe_action.py output (Stages 1-4)
│   ├── action_labels.json
│   ├── <encoder>/features_*.npy + probe.pt + test_metrics.json
│   └── probe_paired_delta.json
├── probe_taxonomy/           # probe_taxonomy.py output (Stage 1 sub-step)
├── probe_motion_cos/         # probe_motion_cos.py (Stages 5-7)
├── probe_future_mse/         # probe_future_mse.py (Stages 8-9)
├── probe_pretrain/           # m09a_pretrain.py (P2 trainer)
├── probe_surgery_3stage_DI/  # m09c_surgery.py (P3a)
├── probe_surgery_noDI/       # m09c_surgery.py (P3b)
└── probe_plot/               # probe_plot.py (Stage 10) — THE plots
```

## 🛡️ Workflow rules

### MANDATORY PROS/CONS GATE

Before any tool call that changes state (Edit / Write / Bash mutations / git ops), list ≥3 pros AND ≥3 cons for ≥2 options ("do nothing" counts as an option). Concrete past incident: 2026-04-20 `sed -i` on a running bash script looked obvious, silently failed, burned 4 h of overnight GPU.

### VERIFY-FIRST RECOMMENDATIONS

Don't trust the first recommendation. Every actionable claim must be backed by direct evidence gathered IN THE SAME TURN — log greps, file shape/mtime checks, jsonl tails, code reads, WEBSEARCH cites — before being delivered to the user. Banned phrases: "should be fine", "I think it's safe", "the cache should reuse", "no changes needed" without a tool-call result quoted in the same response.

### NO DEFER, NO TECH DEBT

Phrases banned: "defer to iter10", "harmless, minor cost", "not worth fixing now", "revisit later", "acceptable at this scale". Fix at the right layer, in this session. If genuinely out-of-scope, say so explicitly with LoC estimate — don't hide under "minor" / "harmless" adjectives.

### INTERRUPT FREELY

All GPU scripts have checkpoint resume. Don't say "let the run complete" when a fix is ready. Kill the run, apply the fix, restart.

## 🧪 Testing & validation

### 3-check gate (auto via `post-edit-lint.sh`)

After ANY edit to `src/**/*.py`:
1. `python3 -m py_compile <file>`
2. `python3 -c "import ast; ast.parse(open('<file>').read())"`
3. `ruff check --select F,E9 <file>`

Must all pass before proceeding.

### Preflight skill (`/preflight @<file>`)

CPU-runnable checks B1-B65 covering past regression classes. Source: `.claude/skills/preflight/SKILL.md`. Run on any non-trivial edit before kicking GPU.

### REPL smoke test

Always exercise NEW code via Python REPL on synthesized inputs BEFORE the GPU run. The 11-test REPL gate for `utils/multi_task_loss.py` saved a multi-hour failed FULL run.

## 🔧 Common one-liners

```bash
# Activate venv (every fresh shell)
source venv_walkindia/bin/activate

# 3-check gate
python3 -m py_compile <file> && python3 -c "import ast; ast.parse(open('<file>').read())" && ruff check --select F,E9 <file>

# What changed in this session
git status --short
git diff --stat HEAD

# Find a YAML key's consumers
grep -rE "cfg\[.<key>.\]|\.<key>\b" src/ | grep -v _archive

# Tail a JSONL log (crash-safe writes — partial last line is OK)
tail -f outputs/<mode>/<dir>/loss_log.jsonl | jq

# Disable wandb for one run
... --no-wandb 2>&1 | tee logs/<name>.log

# Check artifact mtime vs log mtime (catch stale outputs)
stat -c "%y %n" outputs/<mode>/<dir>/*.pt logs/<name>.log
```

## 🚫 Things you'll be tempted to do that are wrong

| Temptation | Why wrong | Right thing |
|---|---|---|
| Add a default value in `argparse` for a new knob | Hardcoded — violates Rule 1 | Default in YAML; argparse `default=None` |
| `cfg.get("key", scalar)` for a "safe" fallback | Silent failure — violates Rule 2 | Direct subscript; KeyError surfaces missing schema |
| Wrap a flaky CUDA call in `try/except: pass` | Silent failure — violates Rule 3 | Defensive handling + raise after recovery, OR fail-hard with a useful message |
| Quick-fix a regression with `|| true` in shell | Silent failure — violates Rule 3 | Diagnose root cause; if truly optional, use `if [ -f X ]; then ...; fi` |
| Chain trainers with `&&` for "fail-fast" | Independent runs — violates Rule 5 | Use `;`; let independent failures be independent |
| "Defer to next iter" or "minor cost, harmless" | NO DEFER NO TECH DEBT — fix at the right layer now | Identify minimum surgical fix; apply this session |
| Trust your first recommendation without evidence | VERIFY-FIRST — past incidents from this | Quote tool-call results inline before claiming |
| Reduce SANITY config knobs to "make it fit on 24 GB" for V-JEPA training | OOM is structural (params + teacher + opt > 24 GB) | Move training to 96 GB; SANITY is for code-path validation only |
