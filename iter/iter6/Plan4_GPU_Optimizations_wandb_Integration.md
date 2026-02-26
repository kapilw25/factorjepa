╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
Plan: 4 GPU Optimizations + wandb Integration                                                                                                                                              

Context

Pipeline is ready for GPU. Before spinning up A100-40GB, apply 4 optimizations: torch.compile for V-JEPA, cuML GPU UMAP (50-100x speedup), FAISS GPU for all indices, and wandb experiment
tracking. Per src/CLAUDE.md: GPU only, no M1/CPU fallback.

Files Modified (8 files)

┌──────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│           File           │                                Changes                                 │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ src/m05_vjepa_embed.py   │ +torch.compile, +wandb                                                 │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ src/m07_umap_plot.py     │ cuML GPU UMAP (replace umap-learn), FAISS GPU confusion matrix, +wandb │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ src/m06_faiss_metrics.py │ Overlap@K FAISS→GPU, +wandb                                            │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ src/m04_vlm_tag.py       │ +wandb (worker subprocess only)                                        │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ src/utils/wandb_utils.py │ NEW — shared wandb helper                                              │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ setup_env_uv.sh          │ +cuml-cu12, +wandb install steps                                       │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ requirements_gpu.txt     │ +wandb>=0.15.0, +cuml-cu12 note                                        │
├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ iter/iter6/plan.md       │ Update with new optimizations                                          │
└──────────────────────────┴────────────────────────────────────────────────────────────────────────┘

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
Opt 1: torch.compile(model) — m05_vjepa_embed.py

Where: After model.eval() (line ~456), before consumer loop.

model.eval()
print("Applying torch.compile (first batch will be slow due to compilation)...")
model = torch.compile(model)

- V-JEPA ViT-G has static shapes (batch × 64 × 384 × 384) → ideal for torch.compile
- 15-30% forward pass speedup after warmup
- First batch takes 30-90s to compile (print warns user)
- Zero new dependencies (built into PyTorch 2.x)

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
Opt 2: cuML GPU UMAP — m07_umap_plot.py
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
Change 1 — Import (lines 27-37): Replace import umap with:
try:
    from cuml.manifold import UMAP as cuUMAP
except ImportError:
    print("FATAL: cuML not installed. GPU UMAP required (no CPU fallback).")
    print("Install via setup_env_uv.sh --gpu")
    sys.exit(1)
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
Change 2 — Usage (lines 106-108): Replace umap.UMAP(...) with:
reducer = cuUMAP(n_components=2, n_neighbors=n_neighbors,
                min_dist=min_dist, random_state=42, verbose=True)
emb_2d_cu = reducer.fit_transform(embeddings)
emb_2d = emb_2d_cu.get() if hasattr(emb_2d_cu, 'get') else np.asarray(emb_2d_cu)
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
Change 3 — Docstring: Update from "CPU-only" to "GPU-only (cuML UMAP, Nvidia CUDA)"
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
Change 4 — Confusion matrix FAISS (lines ~163-165): Move to GPU:
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatL2(embeddings.shape[1])
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.add(embeddings)

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

Opt 3: Overlap@K FAISS→GPU — m06_faiss_metrics.py

Where: Lines 186-192 in compute_overlap_at_k()

res = faiss.StandardGpuResources()
idx_a_cpu = faiss.IndexFlatL2(mid)
idx_b_cpu = faiss.IndexFlatL2(d - mid)
idx_a = faiss.index_cpu_to_gpu(res, 0, idx_a_cpu)
idx_b = faiss.index_cpu_to_gpu(res, 0, idx_b_cpu)
idx_a.add(emb_a)
idx_b.add(emb_b)

Same pattern as existing build_faiss_index() line 71.

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

Opt 4: wandb Integration
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
4A. New file: src/utils/wandb_utils.py

Shared helpers to avoid boilerplate in each module:
- add_wandb_args(parser) — adds --no-wandb flag
- init_wandb(module_name, mode, config, enabled) — init run, graceful error handling
- log_metrics(run, metrics) — log dict if run active
- log_image(run, key, path) — log plot as wandb.Image
- log_artifact(run, name, path) — log file as artifact
- finish_wandb(run) — close run

All functions are no-ops if run is None (when --no-wandb is passed).
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
4B. m04_vlm_tag.py — wandb in WORKER only

- Import wandb_utils functions
- Add --no-wandb to argparse
- Pass --no-wandb through orchestrator→worker subprocess cmd
- init_wandb("m04", ...) in worker_main() after check_gpu()
- log_metrics(run, {...}) in progress block (~line 749)
- log_artifact(run, "tags", tags_file) + finish_wandb(run) at end of worker

NOTE: Orchestrator does NOT init wandb (no GPU work). Only workers get wandb runs.
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
4C. m05_vjepa_embed.py

- Add --no-wandb to argparse
- init_wandb("m05", ...) after check_gpu()
- log_metrics(run, {...}) per batch in consumer loop (throughput, count)
- log_artifact for embeddings.npy + paths.npy at end
- finish_wandb(run) at end of main()
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
4D. m06_faiss_metrics.py

- Add --no-wandb to argparse
- init_wandb("m06", ...) after check_gpu()
- log_metrics(run, {...}) for all 9 metrics with easy/ and hard/ prefixes
- log_metrics for confidence sweep
- log_image for 4 plots, log_artifact for metrics.json
- finish_wandb(run) at end
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
4E. m07_umap_plot.py

- Add --no-wandb to argparse
- init_wandb("m07", ...) after output_dir setup
- log_metrics for macro/micro from metrics_data
- log_image for umap, confusion_matrix, knn_grid plots
- finish_wandb(run) at end

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

Dependencies

requirements_gpu.txt — add:

# Experiment Tracking
wandb>=0.15.0

# cuML GPU (UMAP) - Nvidia GPU ONLY
# cuml-cu12 installed via setup_env_uv.sh --gpu (RAPIDS PyPI)

setup_env_uv.sh — add after FAISS install (renumber steps):

# [6/7] Install cuML (GPU UMAP)
uv pip install cuml-cu12 --extra-index-url https://pypi.nvidia.com

# [7/7] Install wandb
uv pip install wandb

Add to final verification:
import cuml
import wandb
print(f'cuML:           {cuml.__version__}')
print(f'wandb:          {wandb.__version__}')

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

Implementation Order

1. Opt 1 (torch.compile) — 3 lines, zero risk
2. Opt 3 (FAISS GPU) — 6 lines in m06, 4 lines in m07, follows existing pattern
3. Opt 4 (wandb) — new wandb_utils.py + additions to 4 modules, additive only
4. Opt 2 (cuML UMAP) — requires new dependency, do last

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

Verification

source venv_walkindia/bin/activate
# py_compile all modified files
for f in src/m04_vlm_tag.py src/m05_vjepa_embed.py src/m06_faiss_metrics.py src/m07_umap_plot.py src/utils/wandb_utils.py; do
python -m py_compile "$f" && echo "PASS: $f"
done
# AST check
for f in src/m04_vlm_tag.py src/m05_vjepa_embed.py src/m06_faiss_metrics.py src/m07_umap_plot.py src/utils/wandb_utils.py; do
python -c "import ast; ast.parse(open('$f').read())" && echo "AST OK: $f"
done
# --help check (m04, m06, m07 should show --no-wandb flag)
python src/m06_faiss_metrics.py --help | grep no-wandb
python src/m07_umap_plot.py --help | grep no-wandb
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌


