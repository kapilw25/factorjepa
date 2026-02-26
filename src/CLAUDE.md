1) numbering "m"odules for sorting: [src/m01_<name>.py, src/m02_<name>.py, ] (start with letter "m" to not get into import error with number as prefix)
-- the number m*[01, 02, 3.., 0n] must NOT be REPEATED
2) whem moving form current/previous phase to next phase, move the current python files to @src/legacy/ directory and rename the existing files as per nature of next phase
3) create common / UTILS files here @src/utils
4) note:for both (inference and training) on Nvidia's GPU only. no M1/CPU fallback
keep M1 macbook for only CPU or API based operations
5) in each python script, keep DOCSTRING limited to terminal commands (covering only 2 models --SANITY, --FULL, etc arguments) to be executed 
5.1) each command must have  `python -u src/*.py --args arg_name 2>&1 | tee logs/<log_name>log` format
5.2) start DOCSTRING with  max 2 lines of explanation about the code
6) once all python files are built @src/ directory,
read current versions and then update 
@setup_env.sh , 
@requirements.txt [covering non GPU libraries e.g - numpy, matplotlib, etc], 
@requirements_gpu.txt [covering Nvidia GPU libraries only, e.g - torch, transformers, etc]
6.1) for REPRODUCIBILITY purposes, install everything on "venv_*" via @requirements.txt and/or @requirements_gpu.txt ONLY
no individual isntallations >> which will make REPRODUCIBILITY difficult

7) TEST [`py_compile` && `--help` && `ast` && `import`] using VIRTUAL ENVIRONMENT `source venv_*/bin/activate`
"venv_*" represent to local virtual environment available .e g- here "venv_walkindia"
```
source venv_*/bin/activate     # find   local <venv_*>                                                                       
python -m py_compile src/m01_*.py  # Syntax check                                                      
python src/m01_*.py --help          # Args check                                                       
python -c "import ast; ast.parse(open('src/m01_*.py').read())"  # AST check 
```


8) for each @outputs/plots - both [.png & .pdf] versions should be generated  >> build (src/utils/plotting.py) accordingly
9) be DEVIL's ADVOCATE to look for a)  OOM b) underutiliation of GPU [solution: batch processing, vLLM], C) idle period for GPU are, d) Data starvation (46% of GPU underutil per Microsoft study), e) VRAM memory leaks across long runs, f) Numerical instability with fp16 — soln: uses flash-attn-2 which handles fp16 correctly, g) Checkpoint corruption

10) GPU Optimizations (apply to any GPU pipeline):
10.1) torch.compile(model) after model.eval() — 15-30% speedup on static-shape models (ViT, ResNet). Print warning about first-batch compile latency.
10.2) FAISS GPU for ALL index operations — use faiss.StandardGpuResources() + faiss.index_cpu_to_gpu(res, 0, cpu_index). Never leave CPU FAISS in GPU scripts.
10.3) cuML GPU replacements — replace sklearn/umap-learn CPU ops with cuML equivalents (UMAP, DBSCAN, KMeans, PCA). 50-100x speedup. Install via `--extra-index-url https://pypi.nvidia.com`.
10.4) wandb experiment tracking — shared utils pattern:
  - src/utils/wandb_utils.py with: add_wandb_args(), init_wandb(), log_metrics(), log_image(), log_artifact(), finish_wandb()
  - all functions are no-ops when run is None (--no-wandb flag)
  - every GPU module gets --no-wandb argparse flag
  - log metrics per batch (throughput, loss), log plots as wandb.Image, log output files as artifacts
  - add wandb>=0.15.0 to requirements_gpu.txt (NOT individual pip install)