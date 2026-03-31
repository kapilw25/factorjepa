---
name: audit-against-gold
description: Compare any src/m*.py script against its gold-standard open-source equivalent. WebSearch for the best reference implementation, then brutally list every deficiency.
allowed-tools: Read, Grep, Glob, WebSearch, WebFetch
argument-hint: <file-path or "all" to audit every src/m*.py>
---

# Gold Standard Audit

For each specified script (or all `src/m*.py` if "all"):

## Step 1: Identify what the script does
Read the file's docstring and first 50 lines. Classify it:
- VLM inference (m04) → reference: vLLM serving, HF batch inference examples
- Video embedding (m05) → reference: Meta's vjepa2 inference code
- Baseline encoders (m05b) → reference: DINOv2/CLIP official eval scripts
- kNN evaluation (m06) → reference: FAISS official benchmarks, DINOv2 eval
- Training loop (m09) → reference: Meta's vjepa2/app/vjepa/train.py
- UMAP (m07) → reference: cuML UMAP examples
- Plotting (m08) → reference: academic paper figure scripts

## Step 2: WebSearch for gold standard
Search for the best open-source implementation of the same task:
- "{task} official implementation github"
- "{framework} best practices {task}"
- "state of the art {task} training loop pytorch"

## Step 3: Compare and list deficiencies
For each gold standard found, compare against our script and report:

| Category | What to check |
|----------|--------------|
| **Missing features** | Does gold standard have validation/eval that we lack? |
| **Error handling** | Does gold standard validate inputs we silently accept? |
| **Performance** | Does gold standard use optimizations we miss? (grad accumulation, mixed precision, compile) |
| **Logging** | Does gold standard log metrics we don't? |
| **Reproducibility** | Does gold standard set seeds, deterministic flags we miss? |
| **Data integrity** | Does gold standard validate data shapes/types we skip? |

## Output Format

For each script:
```
=== src/m{XX}_{name}.py ===
Gold standard: {URL}
Deficiencies found: N

[CRITICAL] {description} — gold standard does X, we don't
[HIGH]     {description} — gold standard does X, we do Y (worse)
[LOW]      {description} — nice-to-have from gold standard
```

Be brutal. If our script is correct, say "0 deficiencies" and move on.
Do NOT fabricate deficiencies that don't exist.
