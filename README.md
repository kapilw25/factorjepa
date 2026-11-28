# FactorJEPA: World Models for Crowded, Chaotic Global South Urban Scenes

**Factorizing monolithic future prediction into layout, agent, and interaction channels, trained and evaluated on DenseWorld.**

[![arXiv](https://img.shields.io/badge/arXiv-2608.01049-b31b1b?logo=arxiv)](https://arxiv.org/abs/2608.01049)
[![Project Page](https://img.shields.io/badge/Project-Page-8B3A2A)](https://kapilw25.github.io/factorjepa/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-ffd21e?logo=huggingface)](https://huggingface.co/datasets/anonymousML123/denseworld-115k)

> **115,687 clips** | 714 videos | 22 cities | 276 hours | 121 GB

**[See 192 video clips from all cities and taxonomy categories on the project page](https://kapilw25.github.io/factorjepa/)**

---

## Key Finding

Frozen video encoders are **nearly motion-blind** on crowded Global South scenes; factor-view predictor surgery (**FactorJEPA**) restores predictive structure. With an *identical probe head* over either backbone, only the FactorJEPA-adapted encoder answers motion questions reliably:

| Motion probe (1,825 held-out clips) | Frozen V-JEPA 2.1 | FactorJEPA (ours) |
|-------------------------------------|-------------------|-------------------|
| Motion-speed quartile (chance 25.6%) | 60.9% | **69.8%** |

At the full 115k-clip scale, FactorJEPA separates from the strongest fine-tuning rival on **all four predictive diagnostics** (in 95%-CI units): mask-ratio slope **43.3×**, future-frame L1 **33.2×**, motion-cosine **20.0×**, causal L1 **13.9×**. Method rankings replicate across the 2B and 1B V-JEPA 2.1 backbones (Spearman ρ = 0.895–0.978).

📄 **[Read the paper on arXiv (2608.01049)](https://arxiv.org/abs/2608.01049)**

---

## Setup

A fresh NVIDIA box goes from `git clone` to a running evaluation in three steps. It was written for a rented vast.ai RTX PRO 6000 Blackwell (sm_120) on Ubuntu with a CUDA base image, and works on any Linux box with an NVIDIA GPU. There is no CPU fallback.

### 1. Clone and add your Hugging Face token

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
echo "HF_TOKEN=hf_your_token_here" > .env   # .env is git-ignored. Never commit it.
```

All data and saved models come from the gated HF dataset [anonymousML123/factorjepa-outputs](https://huggingface.co/datasets/anonymousML123/factorjepa-outputs). Accept its access terms once with the account that owns the token. The environment step also pulls the gated [facebook/sam3](https://huggingface.co/facebook/sam3) weights, so accept those terms too.

### 2. Run the setup script

Pick a profile, look at the plan with `--dry-run`, then run it inside `tmux`:

```bash
set -o pipefail; mkdir -p logs
bash scripts/setup_new_box.sh --profile eval-heldout --dry-run
bash scripts/setup_new_box.sh --profile eval-heldout 2>&1 | tee logs/setup_new_box_$(date +%Y%m%d_%H%M%S).log
```

| Profile | Downloads | Download size | Free disk needed | Enables |
|---|---|---|---|---|
| `eval-heldout` | `outputs/heldout_video`, the corpus manifest `full_local.json`, and the ~1,700 held-out clips cut from the HF shards | ~3 GB | ~78 GB | the video held-out test of the 4 saved 1B 10k models |
| `train-10k` | `data/eval_10k_local` (clips, SAM masks, factor sets, tags, motion features) | ~21 GB | ~271 GB | training 1B arms on the 10k corpus |
| `eval-10k` | `data/eval_10k_local` and `outputs/poc/vjepa_2_1_vitg_1B` (every saved 1B 10k arm and its eval) | ~244 GB | ~344 GB | evaluating, and training, 1B arms on the 10k corpus |
| `full` | `data/full_local` (115k clips) and `outputs/full` | ~319 GB | ~959 GB | the FULL 115k run at 1B |

Sizes come from the HF file listing (Nov 28 2026). "Free disk needed" is the download plus working space: frame caches, leased checkpoints and new results. For `full` the working space is the frame cache cap in `configs/pipeline.yaml` plus 40 GB, the same rule `scripts/ngpu_run.py` checks. The environment itself adds about 50 GB on top (venv, the 16.9 GB 1B base checkpoint, SAM 3 and Grounding DINO).

What the script does, in order:

| Step | What happens |
|---|---|
| 1. Preflight | Checks Linux, the GPU and its compute capability, that `.env` has an `HF_TOKEN` line (the value is never printed), and that `configs/pipeline.yaml` points at the profile's corpus |
| 2. Environment | Runs `./setup_env_uv.sh --gpu`. On compute capability 12.x it adds `--from-wheels` to use the prebuilt sm_120 flash-attention wheel. Then it activates `venv_walkindia` |
| 3. Disk | Compares free disk with the profile table and stops if it is short |
| 4. Download | Fetches whole HF subtrees with `src/utils/hf_outputs.py download-data`, never a filtered part of a tree |
| 5. Held-out clips | `eval-heldout` only: streams the labelled held-out clips out of the corpus shards with `src/utils/hf_shard_subset.py` |
| 6. Verify | `hf_outputs.py verify` for each outputs tree, and a file check for each data tree and the base checkpoint |
| 7. Summary | One table, then the next command to run |

| Flag | Meaning |
|---|---|
| `--profile <name>` | Required. One of the four profiles above |
| `--dry-run` | Prints every step and command and runs nothing. Safe on a laptop |
| `--skip-env` | Skips `setup_env_uv.sh` and reuses the existing `venv_walkindia`, for a re-run after a fix |

Every long step writes its own log under `logs/`. A re-run skips data trees that already passed verification.

The 10k profiles need the pipeline pointed at the 10k corpus first. The script stops and prints this command if it is not:

```bash
sed -i -e 's|local_data_dir:.*|local_data_dir: "data/eval_10k_local"|' \
    -e 's|master_manifest_name:.*|master_manifest_name: "eval_10k.json"|' configs/pipeline.yaml
```

### 3. Run

Start every run session with:

```bash
source venv_walkindia/bin/activate
set -o pipefail
```

**Video held-out test** (profile `eval-heldout`). It scores Future-frame L1, Forward L1, Mask-ratio slope and Motion cosine on 1,702 clips from 136 source videos that gave no clip to training or validation. Each arm's checkpoints are fetched from HF for the run and deleted after it. Each arm spec is `"<variant>|<arm dir on HF>|<checkpoint file>"`.

```bash
set -a; source .env; set +a   # the checkpoint fetch reads HF_TOKEN from the environment
# All 4 arms at once on one GPU under CUDA MPS (20 GB cap per arm):
bash scripts/heldout_video_parallel.sh 20 anonymousML123/factorjepa-outputs outputs/poc/vjepa_2_1_vitg_1B/train \
    data/heldout_video_local outputs/heldout_video/probe_action outputs/heldout_video_rerun ckpt_lease \
    configs/model/vjepa2_1_vitg.yaml \
    "vjepa_2_1_vitg_pretrain_encoder|m09a_pretrain_encoder|m09a_ckpt_best.pt" \
    "vjepa_2_1_vitg_surgery_raw_encoder|m09c_surgery_raw_encoder|m09c_ckpt_best.pt" \
    "vjepa_2_1_vitg_surgical_3stage_DI_diheavy_encoder|m09c_surgery_3stage_DI_diheavy_encoder|m09c_ckpt_best.pt" \
    "vjepa_2_1_vitg_peft_lora_encoder|m09b_peft_lora_encoder|m09c_ckpt_best.pt" \
    2>&1 | tee logs/heldout_video_parallel_$(date +%Y%m%d_%H%M%S).log
# One arm at a time (no MPS needed): same arguments with scripts/heldout_video_eval.sh
```

The shipped results sit in `outputs/heldout_video`. The command above writes a fresh copy to `outputs/heldout_video_rerun`, so the two can be compared. `src/utils/paired_ladder.py` turns the per-clip arrays into paired 95% intervals (see its docstring).

**Training** (profiles `train-10k`, `eval-10k`, `full`). Train one arm with `scripts/run_train.sh <arm> --SANITY|--POC|--FULL`. The arm names live in `configs/arm_registry.yaml`. List them with `python src/utils/arm_registry.py train-names`. Every other arm starts from `pretrain_encoder`, so train it first.

```bash
BACKBONE=vjepa_2_1_vitg ./scripts/run_train.sh pretrain_encoder --SANITY 2>&1 | tee logs/run_train_pretrain_sanity_$(date +%Y%m%d_%H%M%S).log
BACKBONE=vjepa_2_1_vitg ./scripts/run_train.sh pretrain_encoder --POC 2>&1 | tee logs/run_train_pretrain_poc_$(date +%Y%m%d_%H%M%S).log
BACKBONE=vjepa_2_1_vitg ./scripts/run_train.sh surgery_3stage_DI_diheavy_encoder --POC 2>&1 | tee logs/run_train_diheavy_poc_$(date +%Y%m%d_%H%M%S).log
# All arms plus their evals as one job graph across N GPUs:
ITER18_BACKBONE=vjepa_2_1_vitg python -u scripts/ngpu_run.py --mode POC --gpus 4 --cache 2 2>&1 | tee logs/ngpu_run_poc_$(date +%Y%m%d_%H%M%S).log
```

**Evaluation** (profile `eval-10k`). `scripts/run_eval.sh [--SANITY|--POC|--FULL] [--encoders "<list>"]` scores the listed encoders. An encoder name is `vjepa_2_1_vitg_frozen` or `vjepa_2_1_vitg_<token>`, with tokens from `python src/utils/arm_registry.py eval-tokens`. Encoders whose trained checkpoint is missing are dropped with a message.

```bash
# CACHE_POLICY_ALL=1 keeps the downloaded caches and scores only what is missing. 2 recomputes everything.
ITER18_BACKBONE=vjepa_2_1_vitg EVAL_CORPUS=eval_10k CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --POC \
    --encoders "vjepa_2_1_vitg_frozen vjepa_2_1_vitg_surgical_3stage_DI_diheavy_encoder" \
    2>&1 | tee logs/run_eval_poc_$(date +%Y%m%d_%H%M%S).log
```

**FULL 115k run** (profile `full`). Follow [the iter19 runbook](iter/iter19_train_115kclips/runbook_train_115kclips.md). It sets `ITER18_BACKBONE`, `EVAL_CORPUS=full`, `EMIT_PERFRAME=1` and the list of arms to skip, then runs `scripts/ngpu_run.py --mode FULL`. Watch it from a second pane with `python -u scripts/ngpu_run_status.py --mode FULL --log <the run log>`.

---

## Pipeline

The training and eval scripts resume from their own checkpoints, so they are safe to stop and restart.

| Script | What it runs |
|---|---|
| `scripts/setup_new_box.sh` | New box setup: environment, HF downloads, verification |
| `setup_env_uv.sh` | The environment alone. Its header lists the GPU, minimal eval and Mac modes |
| `scripts/run_factor_prep_parallel.sh` | Data prep for surgery: SAM 3 segmentation (m10) on parallel workers, then factor sets (m11) |
| `scripts/run_train.sh` | One training arm (m09a to m09f) at SANITY, POC or FULL |
| `scripts/run_eval.sh` | The eval stages (m04e labels, m12a to m12f metrics, m13 plots) for a list of encoders |
| `scripts/ngpu_run.py` | All arms and their evals as one job graph across N GPUs |
| `scripts/ngpu_run_status.py` | Live status of an `ngpu_run.py` run, with a 45 minute HF backup |
| `scripts/heldout_video_eval.sh` | Video held-out test, one arm at a time |
| `scripts/heldout_video_parallel.sh` | Video held-out test, several arms at once under CUDA MPS |
| `src/utils/hf_outputs.py` | Upload, download and verify trees in the HF outputs repo |

Older wrappers (`train_pretrain.sh`, `train_surgery.sh`, `run_embed.sh` and earlier orchestrators) live in `scripts/legacy/` and are no longer maintained.

The surgery arms train in three stages on layout, agent and interaction clips (`D_L`, `D_A`, `D_I`) built from SAM 3 masks. The rival arms (full fine-tuning, LP-FT, LoRA, DoRA, Auto-RGN, continual SSL) each have their own trainer, and all arms are listed in `configs/arm_registry.yaml`.

---

## Dataset

| Tier | Cities | Clips | Hours | GB |
|------|--------|-------|-------|----|
| Tier 1 | 6 metros | 68,614 | 161h | 74 |
| Goa | 1 | 5,835 | 14h | 6 |
| Tier 2 | 15 cities | 40,743 | 99h | 41 |
| Monuments | 3 | 495 | 1h | 1 |
| **Total** | **22** | **115,687** | **276h** | **121** |

---

## Code Structure

```
src/
├── m00-m03          # Data pipeline (YouTube → clips → WebDataset → HF)
├── m04              # VLM tagging (Qwen3-VL-8B, 16-field taxonomy)
├── m04d             # GPU-RAFT optical flow (motion features)
├── m04e/m04f        # Action and taxonomy labels for the probes
├── m05/m05b/m05c    # Embeddings (V-JEPA + baselines + True Overlap)
├── m07              # UMAP (cuML GPU)
├── m09a-m09f        # Training arms (pretrain, PEFT, surgery, continual SSL, Auto-RGN, naive FT)
├── m10/m11          # SAM 3 segmentation + layout/agent/interaction factor sets
├── m12a-m12f        # Eval metrics (action, motion cosine, taxonomy, future frame, predictor and encoder temporal)
├── m13              # Eval plots
├── m14-m18          # Visual demos and the VLM head
├── legacy/          # Retired modules (m06 FAISS metrics, m08 plots, ExPLoRA)
└── utils/           # Config, bootstrap CI, gpu_batch, HF transfer, wandb
```

---

## Authors

Kapil Wanaskar¹, Gaytri Jena², Aman Chadha³, Vinija Jain⁴, Vasu Sharma⁵, Amitava Das⁶

¹San Jose State University, USA · ²UC Berkeley, USA · ³Apple, USA · ⁴Meta, USA · ⁵PocketFM, USA · ⁶Pragya Lab, BITS Pilani Goa, India

Part of the **DenseWorld** research program: *World Models for Populous, Crowded, and Chaotic Global South*

## Citation

```bibtex
@article{wanaskar2026factorjepa,
  title={FactorJEPA: Factorizing Monolithic Futures into Layout-Agent-Interaction Channels for Crowded and Chaotic Global South Urban Worlds},
  author={Wanaskar, Kapil and Jena, Gaytri and Chadha, Aman and Jain, Vinija and Sharma, Vasu and Das, Amitava},
  journal={arXiv preprint arXiv:2608.01049},
  year={2026}
}
```
