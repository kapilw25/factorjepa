---
name: audit-against-gold
description: Audit src/m*.py scripts for API mismatches, missing deps, and deficiencies. Reads installed package source for bleeding-edge libs, WebSearches for stable ones. Catches errors on CPU before expensive GPU runs.
allowed-tools: Read, Grep, Glob, WebSearch, WebFetch, Bash, Agent
argument-hint: <file-path or "all" to audit every src/m*.py>
---

# Gold Standard Audit + API Preflight

For each specified script (or all `src/m*.py` if "all"), run TWO passes:

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
Read the file's docstring and first 50 lines. Classify it:
- VLM inference (m04) → reference: vLLM serving, HF batch inference examples
- Video embedding (m05) → reference: Meta's vjepa2 inference code
- Baseline encoders (m05b) → reference: DINOv2/CLIP official eval scripts
- kNN evaluation (m06) → reference: FAISS official benchmarks, DINOv2 eval
- Training loop (m09) → reference: Meta's vjepa2/app/vjepa/train.py
- SAM segmentation (m10) → reference: sam3 demo scripts, sam3 eval code
- Factor datasets (m11) → reference: curriculum learning / data augmentation papers
- UMAP (m07) → reference: cuML UMAP examples
- Plotting (m08) → reference: academic paper figure scripts

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
