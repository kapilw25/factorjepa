---
name: audit-against-gold
description: Audit src/m*.py scripts for API mismatches, missing deps, deficiencies, AND hypothesis sufficiency (does the script produce the evidence the paper's decision gate requires?). Reads installed package source for bleeding-edge libs, WebSearches for stable ones, reads plan_training.md for the paper's win condition. Catches errors on CPU before expensive GPU runs.
allowed-tools: Read, Grep, Glob, WebSearch, WebFetch, Bash, Agent
argument-hint: <file-path or "all" to audit every src/m*.py>
---

# Gold Standard Audit + API Preflight + Hypothesis Sufficiency

For each specified script (or all `src/m*.py` if "all"), run THREE passes:

---

## Pass 1: API Preflight (catch argument errors before GPU)

### Step 1A: Extract all external library calls
Grep the script for imports and function calls to these GPU-expensive libraries:

| Library | Where installed | How to verify |
|---------|----------------|---------------|
| sam3 | `venv_walkindia/lib/python3.12/site-packages/sam3/` | Read source (bleeding-edge, no stable docs) |
| transformers | PyPI | WebSearch HuggingFace docs |
| peft | PyPI | WebSearch PEFT docs |
| faiss | `venv_walkindia/lib/python3.12/site-packages/faiss/` | Read source (custom build) |
| cuml | PyPI/RAPIDS | WebSearch RAPIDS docs |
| flash_attn | `venv_walkindia/lib/python3.12/site-packages/flash_attn/` | Read source (custom wheel) |
| torch / torchvision | PyPI nightly | WebSearch PyTorch docs |
| torchcodec | PyPI | WebSearch docs |
| timm | PyPI | WebSearch docs |

### Step 1B: For each external call, verify the signature

**Bleeding-edge packages (sam3, flash_attn, faiss custom wheels):**
- Read the actual installed source file (find via `Glob` in site-packages)
- Extract the function/method signature (def line + docstring)
- Compare against our call: argument names, types, required vs optional

**Stable packages (transformers, peft, torch, cuml, timm):**
- WebSearch for "{package} {function_name} API reference"
- Compare against our call

### Step 1C: Check import chains for hidden deps
For each `import X` or `from X import Y`:
- Try to trace the import chain (what does X.__init__.py pull in?)
- Flag any transitive dependency that might not be installed
- Especially: sam3 pulls in pycocotools, einops, iopath at import time

### Output for Pass 1:
```
=== API PREFLIGHT: src/m{XX}_{name}.py ===
External calls found: N

[ARGUMENT ERROR] line {N}: {our_call} — param '{name}' does not exist. Actual signature: {sig}
[MISSING DEP]    line {N}: import chain {X} → {Y} → {Z} requires '{pkg}' (not installed)
[SIGNATURE WARN] line {N}: {our_call} — param '{name}' is deprecated, use '{new}' instead
[OK]             line {N}: {our_call} — matches API ✓
```

---

## Pass 2: Gold Standard Comparison

### Step 2A: Identify what the script does
Read the file's docstring and first 50 lines. Classify it — crucially, **disambiguate pretraining loops vs continual-learning loops**, they have different gold-standard references:

- VLM inference (m04) → reference: vLLM serving, HF batch inference examples
- Video embedding (m05) → reference: Meta's vjepa2 inference code
- Baseline encoders (m05b) → reference: DINOv2/CLIP official eval scripts
- kNN evaluation (m06) → reference: FAISS official benchmarks, DINOv2 eval
- **Vanilla pretraining loop (m09a)** → reference: Meta's `vjepa2/app/vjepa/train.py` (JEPA L1 + val JEPA loss only)
- **Continual-learning / progressive-unfreezing loop (m09c surgery, m09b ExPLoRA, any script with multi-stage `set_trainable_prefix` / LoRA adapter injection / EWC-style drift control)** → reference: Lopez-Paz & Ranzato GEM (`arXiv:1706.08840`, BWT/FWT/intransigence formulas), Drive-JEPA (`arXiv:2601.22032`), Surgical V-JEPA (`arXiv:2509.06831`), "Beyond Cosine Decay" CoLLAs 2025 (`arXiv:2503.02844`). A pretrain-only reference will NOT surface the forgetting-metric / mid-training-probe gap.
- SAM segmentation (m10) → reference: sam3 demo scripts, sam3 eval code
- Factor datasets (m11) → reference: curriculum learning / data augmentation papers
- UMAP (m07) → reference: cuML UMAP examples
- Plotting (m08) → reference: academic paper figure scripts

**Branch rule**: if the script contains any of {`set_trainable_prefix`, `stage_mixture`, LoRA `peft.get_peft_model`, explicit per-stage optimizer rebuild, `drift_loss`, `init_params` capture, mode-gated yaml with stage definitions}, it IS continual learning regardless of filename — apply the CL reference set, NOT the pretrain one.

### Step 2B: WebSearch for gold standard
Search for the best open-source implementation of the same task:
- "{task} official implementation github"
- "{framework} best practices {task}"
- "state of the art {task} training loop pytorch"

### Step 2C: Compare and list deficiencies

| Category | What to check |
|----------|--------------|
| **Missing features** | Does gold standard have validation/eval that we lack? |
| **Error handling** | Does gold standard validate inputs we silently accept? |
| **Performance** | Does gold standard use optimizations we miss? (grad accumulation, mixed precision, compile) |
| **Logging** | Does gold standard log metrics we don't? |
| **Reproducibility** | Does gold standard set seeds, deterministic flags we miss? |
| **Data integrity** | Does gold standard validate data shapes/types we skip? |
| **Evaluation protocol** | Does the script compute the same metric (+ 95 % CI) that will appear in the paper's decision table — mid-training, not only post-hoc? If no, the training doesn't produce the evidence the hypothesis requires. Concrete checks for continual-learning scripts: BWT / forgetting monitor across task/stage boundaries; held-out probe eval at `total_steps//N` cadence; per-stage trajectory of the decision-gate metric; bootstrap CI matching `utils/bootstrap.N_BOOTSTRAP`. |

### Output for Pass 2:
```
=== GOLD STANDARD: src/m{XX}_{name}.py ===
Gold standard: {URL}
Deficiencies found: N

[CRITICAL] {description} — gold standard does X, we don't
[HIGH]     {description} — gold standard does X, we do Y (worse)
[LOW]      {description} — nice-to-have from gold standard
```

Be brutal. If our script is correct, say "0 deficiencies" and move on.
Do NOT fabricate deficiencies that don't exist.

---

## Pass 3: Hypothesis Sufficiency (the audit Pass 1 + Pass 2 cannot produce)

Pass 1 + Pass 2 only test *engineering quality* against a reference implementation. They don't test whether the script produces the **evidence the paper's main claim requires**. This is why a training script with correct APIs matching Meta's pretrain loop can still ship without the forgetting-metric / mid-training-probe infrastructure the experiment actually needs.

### Step 3A: Locate the hypothesis
Find and read the nearest planning doc that names the win condition. Search order:
1. `iter/iter*/plan_training.md`
2. `iter/iter*/plan_TODO.md`
3. `iter/iter*/runbook.md`
4. `<project_root>/CLAUDE.md` → "Immediate GOAL" / "Decision gate" / "Win condition"

Extract (a) the **decision-gate metric** (e.g., `Prec@K with non-overlapping 95 % CIs`), (b) the **evaluation data set** (e.g., `val_1k`), (c) the **comparison arms** (e.g., `Surgery vs ExPLoRA vs Frozen`), (d) the **statistical test** (e.g., bootstrap CI).

### Step 3B: Diff hypothesis against what the script emits
Grep the target script for the decision-gate metric being computed, saved, or logged:
```bash
grep -nE "(prec_at_k|map_at_k|cycle_at_k|bootstrap_ci|log_metrics\\(.*probe|training_summary\\.json)" <file>
```
For continual-learning scripts, ALSO grep for:
```bash
grep -nE "(bwt|backward_transfer|forgetting|compute_trajectory|stage.*eval|held.out.*eval)" <file>
```

If the script does NOT emit the decision-gate metric during training, it fails hypothesis sufficiency — no matter how correct its API calls and how closely it matches the pretrain gold standard. The paper cannot be written from this script's output alone; someone will have to re-run downstream eval and stitch numbers, which is brittle and defeats trajectory-aware figures.

### Step 3C: Report gaps as **blockers**, not deficiencies

| Severity | When to assign |
|---|---|
| **[BLOCKER]** | The script's output is *insufficient* to validate the paper's main claim. The training produces a checkpoint but no trajectory / no CI / no forgetting monitor. **Cannot ship** without re-running. |
| **[GAP]** | Decision-gate metric IS emitted but only post-hoc (after all training completes). No mid-training trajectory for early-stopping or figures. |
| **[OK]** | Decision-gate metric emitted at stage/epoch boundaries WITH 95 % CI matching `utils/bootstrap.N_BOOTSTRAP`, logged to wandb + JSONL + summary.json. |

### Output for Pass 3:
```
=== HYPOTHESIS SUFFICIENCY: src/m{XX}_{name}.py ===
Hypothesis: {paraphrased win condition from plan_training.md}
Decision-gate metric: {e.g., Prec@K with non-overlapping 95% CIs on val_1k}

[BLOCKER] Script does not emit {metric} during training → cannot produce trajectory figure, cannot early-stop, cannot validate {claim}.
[GAP]     Script emits {metric} post-hoc only → ok for final number, not for figures.
[OK]      Script emits {metric} with {CI method}, {cadence}, {persistence} → paper-ready.
```

**This is the check that catches methodological holes Pass 1 + Pass 2 miss.** Always run Pass 3 when the target script is named in a `plan_*.md` win condition or decision gate. If `plan_training.md` is absent, print a warning — do not silently skip.
