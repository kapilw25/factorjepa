#!/usr/bin/env python3
"""N-GPU ARM-LEVEL scheduler for SANITY / POC / FULL (2×/4×/8× single-node) — runs the iter18
ablation DAG AND the iter19 full-scale training with the SAME code. One backbone, N train arms +
per-encoder eval jobs as a DAG, each job pinned to the next free GPU via CUDA_VISIBLE_DEVICES.

The DAG (mode-independent — POC/SANITY/FULL share it byte-for-byte, only n_clips/n_epochs differ):
  - ONE backbone (env ITER18_BACKBONE; default vjepa_2_1_vitG 2B) — the BACKBONES axis is dropped.
  - pretrain_encoder is the ROOT for *ALL* other train arms (its m09a_ckpt_best.pt is every arm's
    --init-from-ckpt via run_train.sh SURGERY_INIT).
  - train arms = the arm roster in configs/arm_registry.yaml (novelty + control + FT-technique
    baselines + pretrain); the iter18 ablations use the full roster, iter19 a subset.
  - per-encoder eval jobs run ONLY stages 2,3,11,5,6,8,8b,8c (SKIP_STAGES drops the shared stages);
    the §3 finale is ONE run_eval over ALL encoders with the per-encoder stages skipped → it runs
    the shared paired-Δ/plot stages (1,4,12,13,7,9,9b,9c,10) off the per-encoder caches.

WALL-TIME (per-step time ~constant; steps scale with clips × epochs):
  POC  (7,724 clips, 2 epochs, ~26.6 s/step): ~3.5-4.5 h per encoder arm, ~2-3 h per head arm →
    1 GPU ≈ 50 h · 2 GPU ≈ ~28 h · 4 GPU ≈ ~17 h  (pretrain is the serial ~3.6 h prefix)
  FULL (116k clips, 1 epoch, ~2,700 steps @ ~25 s/step): ~19 h per encoder arm (single-GPU per arm);
    with the pretrain→(diheavy ∥ peft_lora)→eval DAG the option-A wall is ~2-3 days (iter19 plan.md).

SAFETY: split/pool JSONs are atomic-written + deterministic (concurrent run_train.sh regenerations
produce identical bytes); shared labels are bootstrapped by the L: job (Stage-1-only) — every
label-gated job waits on the label files existing. Eval jobs write per-encoder dirs (disjoint).
RESUME: --cache 1 skips train arms whose student_encoder.pt exists.

USAGE (run inside tmux on the multi-GPU box; data + env must already be provisioned):
  python -u scripts/ngpu_run.py --mode POC    --gpus 4 --cache 2              # fresh POC ablation
  python -u scripts/ngpu_run.py --mode POC    --gpus 4 --cache 1              # resume
  python -u scripts/ngpu_run.py --mode FULL   --gpus 8 --cache 2              # iter19 full-scale run
  python -u scripts/ngpu_run.py --mode SANITY --gpus 2 --cache 2 --dry-run    # plan only
"""
import argparse
import contextlib
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))   # iter18 (2026-06-14): arm roster single-source (configs/arm_registry.yaml)
from utils.arm_registry import (  # noqa: E402
    arm2enc as _arm2enc, arm2dir as _arm2dir, merge_arms as _merge_arms, merge_recipe)
from utils.output_paths import eval_dir as _eval_dir, train_dir as _train_dir, head_stage_done  # noqa: E402
from utils.config import get_pipeline_config  # noqa: E402  (SINGLE source for the data-dir → corpus derivation)
from utils.clear_resume_anchors import finalize_missing, is_finalized  # noqa: E402  (trainer-owned "is this arm finalized?" policy)
# iter18 2026-06-08: BACKBONE is the run's encoder family, switchable via env so the SAME scheduler
# runs the 2B champion (vjepa_2_1_vitG, default) or the 1B scale-axis backbone (vjepa_2_1_vitg). The
# status tool reads it back from the run banner (or this same env) so its job-ids match. Set it once
# in BOTH the launch and the watch panes (or rely on the banner parse in the status tool).
BACKBONE = os.environ.get("ITER18_BACKBONE") or get_pipeline_config()["default_backbone"]
# iter18 (2026-06-20) backbone-first tree (plan_output_restructure.md): eval outputs land under
# outputs/<mode>/<backbone>_<size>/eval/<corpus>/ and encoder reads under .../train/, all via
# src/utils/output_paths.py. EVAL_CORPUS is inherited by each run_eval.sh subprocess; the scheduler reads
# it too so its resume markers check the matching eval/<corpus>/ dir. The cross-set INPUTS (LOCAL_DATA /
# EVAL_SUBSET / PROBE_SPLIT / CLASS_EDGES / EVAL_HEAD_REUSE_ROOT) are likewise inherited by run_eval.
#
# CORPUS is a dial ORTHOGONAL to mode. SANITY/POC/FULL are SCALE knobs (≈200 clips / ~7.7k / ~116k) on
# WHICHEVER dataset the pipeline points at — a SANITY run on full_local is 200 clips of full_local, not
# eval_10k. So TRAINED_CORPUS is DERIVED from configs/pipeline.yaml data.local_data_dir — the SAME single
# source run_train.sh reads (run_train.sh:74-77: `LOCAL_DATA=$(yaml_extract … data.local_data_dir)`) — by
# stripping the "_local" suffix off its basename (eval_10k_local→eval_10k, full_local→full,
# subset_10k_local→subset_10k). Flip data.local_data_dir to migrate the whole pipeline; the corpus follows
# automatically for every mode. An explicit TRAINED_CORPUS / EVAL_CORPUS env still wins (the cross-set
# retest, e.g. EVAL_CORPUS=subset_10k while trained on eval_10k). EVAL_CORPUS != TRAINED_CORPUS ⇒ cross-set.


def _corpus_from_data_dir():
    """The corpus name the trainers actually use, derived from pipeline.yaml data.local_data_dir (basename
    minus the '_local' suffix). Mode-INDEPENDENT — matches run_train.sh's LOCAL_DATA for SANITY/POC/FULL."""
    name = Path(get_pipeline_config()["data"]["local_data_dir"]).name
    return name[:-len("_local")] if name.endswith("_local") else name


TRAINED_CORPUS = os.environ.get("TRAINED_CORPUS") or _corpus_from_data_dir()
EVAL_CORPUS = os.environ.get("EVAL_CORPUS") or TRAINED_CORPUS


def enc_prefix():
    """iter17 multi-backbone naming: the CHAMPION 2B vitG drops its size tag → eval encoders are the
    bare 'vjepa_2_1_<arm>'; every OTHER backbone keeps its full name ('vjepa_2_1_vitg_<arm>',
    'vjepa_2_0_vitg_<arm>') so per-backbone eval dirs + the m13 stacked hero never collide/overwrite."""
    return "vjepa_2_1" if BACKBONE == "vjepa_2_1_vitG" else BACKBONE


def enc_name(arm_enc):
    """Full eval-encoder name for the current BACKBONE, e.g. 'vjepa_2_1_frozen' (2B) /
    'vjepa_2_1_vitg_frozen' (1B). Single source for both this scheduler and the status tool."""
    return f"{enc_prefix()}_{arm_enc}"

# (run_train arm → eval-encoder suffix). NOTE the surgery_→surgical_ rename for the factor arms +
# autorgn; the iter18 baselines keep their arm name verbatim (registry rows added 2026-06-04).
# iter18 (2026-06-14): the arm roster + arm→encoder→dir mapping moved to configs/arm_registry.yaml
# (SINGLE SOURCE — was re-typed here AND in run_train.sh/run_eval.sh/the plot scripts; that scatter
# caused 6 whack-a-mole FATALs in one session). arm2enc()/arm2dir() return the scheduler==true arms,
# byte-equal to the former literals (parity-tested). Add a new arm = ONE entry in the yaml.
ARM2ENC = _arm2enc()   # run_train arm → eval encoder-token  (surgery_* → surgical_* swap lives in the yaml)
ARM2DIR = _arm2dir()   # run_train arm → on-disk m09 output dir (resume done-marker: student_encoder.pt)
MERGE_ARMS = _merge_arms()   # kind=merge train_names → built by a post-hoc wiseft_merge job, never run_train
# Stage split (verified against scripts/run_eval.sh should_skip gates, 2026-06-04; 8c/9c iter18):
#   per-encoder stages: 2 features · 3 probe · 11 taxonomy-train · 5/6 motion_cos · 8 future_mse
#                       · 8b predictor_temporal · 8c encoder_temporal (m12f aot/tov/pace/tcc)
#   shared stages:      1 labels · 4 action-paired · 12 taxonomy-paired · 13 taxonomy-plot
#                       · 7 motion-paired · 9 future-paired · 9b predictor-paired
#                       · 9c encoder-temporal-paired · 10 m13 plots
EVAL_SKIP_SHARED = "1,4,12,13,7,9,9b,9c,10"  # per-encoder jobs skip the shared stages (9c = m12f paired)
S3_SKIP_PERENC = "2,3,11,5,6,8,8b,8c"        # the §3 finale skips the per-encoder stages
# iter18 2026-06-07 (metric-parallel Stage 8b): Stage 8b (m12e predictor_temporal, ~2.2 h, the eval
# bottleneck) is fanned into 6 independent single-metric jobs across the GPU pool → wall = slowest
# metric (~teacher_free), not the sum of six. The per-encoder E: job then runs stages 2-8 only.
# iter18 m12f revival: Stage 8c (encoder_temporal, 4 metrics) fans the same way into F: jobs.
EVAL_SKIP_PERENC_NO8B = EVAL_SKIP_SHARED + ",8b,8c"          # E: job — per-encoder stages EXCEPT the fans
EVAL_SKIP_ONLY_8B = "2,3,11,5,6,8,8c," + EVAL_SKIP_SHARED    # P: job — skip all but 8b
EVAL_SKIP_ONLY_8C = "2,3,11,5,6,8,8b," + EVAL_SKIP_SHARED    # F: job — skip all but 8c
PT_METRICS = ["rollout", "causal", "tdist", "teacher_free", "maskratio", "order"]  # mirrors m12e.METRICS keys
ET_METRICS = ["aot", "tov", "pace", "tcc"]                   # mirrors m12f.METRICS keys
# iter18 (2026-06-20) --taxheads-only: build ONLY the taxonomy-head REUSE SOURCE. Runs Stage 11
# (taxonomy-train) per encoder with KEEP_PROBE_HEADS=1 so the per-dim probe heads persist as the
# cross-set EVAL_HEAD_REUSE_ROOT source — fanned one-encoder-per-GPU, turning the ~10 h single-GPU
# `run_eval.sh --encoders <all-17>` taxheads pass into a ~gpus-wide one. No train, no §3 finale.
TAXHEADS_SKIP = "1,2,3,4,5,6,7,8,8b,8c,9,9b,9c,10,12,13"
# iter18 (2026-06-21) --etheads-only: build the ENCODER-TEMPORAL head reuse source (Stage 8c only).
# m12f runs ALL metrics (aot/tov/pace fit a tiny linear head; tcc is training-free) with KEEP_PROBE_HEADS=1
# → m12f --keep-heads saves head_{aot,tov,pace}.pt — the cross-set EVAL_HEAD_REUSE_ROOT source for Stage 8c.
# Fanned one-encoder-per-GPU (the single-GPU run_eval pass was ~12-18 h → ~5 h on 4 GPUs). No §3 finale.
ETHEADS_SKIP = "1,2,3,4,5,6,7,8,8b,9,9b,9c,10,11,12,13"
# cross-set label-bootstrap (L: job, see build_jobs): a FRESH eval corpus has NO action/taxonomy labels,
# and with all arms cache-skipped no run_train bootstraps them → the per-encoder eval jobs (needs_labels)
# stay blocked and the §3 finale FATALs at stage-4. L: runs Stage 1 ONLY, first. Stage-1-only = skip all
# but 1. 2026-06-23: the L: job is now added UNCONDITIONALLY (build_jobs L:labels below) — there is no
# live `!= TRAINED_CORPUS` conditional anymore; the labels used to arrive as a side-effect of the pretrain
# TRAIN job, but --cache 1 resume-skips a pre-existing seed → the labels were never generated. Decoupling
# label-bootstrap from training covers that path. On the normal same-corpus flow (EVAL_CORPUS==TRAINED_CORPUS)
# it is a cache no-op when the labels already exist; the corpus comparison is implicit via EVAL_CORPUS.
LABELS_ONLY_SKIP = "2,3,4,5,6,7,8,8b,8c,9,9b,9c,10,11,12,13"


def cpu_slots(n):
    """Private CPU-set per GPU slot (iter18 2026-06-07): each arm is launched under
    `taskset -c <slice>` owning 64/n physical cores PLUS their SMT siblings — no core
    migration, no cross-arm L3/CCD thrash, no SMT-sibling interference (the residual
    single-node concurrency tax left after worker autotune). This box is single-NUMA
    (lscpu: 1 node), so the cpuset is the applicable subset of the standard numactl
    prescription; the binding is inherited by every DataLoader/ProcessPool worker.
    Returns [None]*n (unpinned, status quo) when lscpu/taskset are unavailable."""
    import shutil as _sh
    try:
        if not _sh.which("taskset"):
            raise RuntimeError("taskset not on PATH")
        out = subprocess.run(["lscpu", "-p=CPU,CORE"], capture_output=True,
                             text=True, check=True, timeout=10).stdout
        core2cpus = {}
        for ln in out.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            cpu, core = map(int, ln.split(",")[:2])
            core2cpus.setdefault(core, []).append(cpu)
        cores = sorted(core2cpus)
        per = len(cores) // n
        if per < 1:
            raise RuntimeError(f"{len(cores)} physical cores < {n} slots")
        slots = []
        for g in range(n):
            chunk = cores[g * per:(g + 1) * per] if g < n - 1 else cores[(n - 1) * per:]
            cpus = sorted(c for k in chunk for c in core2cpus[k])
            slots.append(",".join(map(str, cpus)))
        return slots
    except Exception as e:
        print(f"  [cpuset] pinning disabled ({type(e).__name__}: {e}) — arms launch unpinned",
              flush=True)
        return [None] * n


def ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def build_jobs(mode, taxheads_only=False, etheads_only=False):
    """dict id→job. job = {id,kind,cmd,deps:set,needs_labels:bool,log}."""
    mflag, mtag = f"--{mode}", mode.lower()
    jobs = {}
    if taxheads_only:
        # Reuse-source build: one Stage-11-only job per encoder (frozen + every ARM2ENC arm),
        # KEEP_PROBE_HEADS=1 to persist the heads, no train deps (the encoders already exist on
        # disk — run_eval FATALs loud if one is missing, same trust model as --only). Each gates
        # only on labels_ready, so all N launch immediately and fan across the GPU pool.
        for arm, enc in [("frozen", "frozen")] + list(ARM2ENC.items()):
            encn = enc_name(enc)
            xjid = f"X:{encn}"
            jobs[xjid] = dict(
                id=xjid, kind="eval", deps=set(), needs_labels=True,
                cmd=(f"EVAL_CORPUS={EVAL_CORPUS} CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={TAXHEADS_SKIP} "
                     f"KEEP_PROBE_HEADS=1 CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {encn}"),
                log=f"logs/ngpu_run_{mtag}_taxheads_{encn}_{{ts}}.log")
        return jobs, mtag
    if etheads_only:
        # Encoder-temporal reuse-source build (mirror of taxheads_only): one Stage-8c-only job per encoder,
        # ET_METRIC unset (=all → decode shared across aot/tov/tcc; only pace re-decodes), KEEP_PROBE_HEADS=1
        # → m12f --keep-heads saves head_{aot,tov,pace}.pt (tcc training-free). No train deps; fans across GPUs.
        # Runs on the TRAINED corpus (eval_10k defaults) — DO NOT export the cross-set EVAL_CORPUS/LOCAL_DATA first.
        for arm, enc in [("frozen", "frozen")] + list(ARM2ENC.items()):
            encn = enc_name(enc)
            yjid = f"Y:{encn}"
            jobs[yjid] = dict(
                id=yjid, kind="eval", deps=set(), needs_labels=True,
                cmd=(f"EVAL_CORPUS={EVAL_CORPUS} CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={ETHEADS_SKIP} "
                     f"KEEP_PROBE_HEADS=1 CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {encn}"),
                log=f"logs/ngpu_run_{mtag}_etheads_{encn}_{{ts}}.log")
        return jobs, mtag
    seed_id = f"T:{BACKBONE}:pretrain_encoder"   # seeds shared labels + everyone's init ckpt
    for arm in ARM2ENC:
        jid = f"T:{BACKBONE}:{arm}"
        # iter18 lever #1: a kind=merge arm is NOT trained — its "train" job is a post-hoc weight merge
        # (OURS encoder [+ predictor] × FROZEN base V-JEPA). Recipe (base/alpha/predictor) ← registry
        # merge_recipe() — SINGLE source, no literals; the frozen base ckpt is resolved per backbone via
        # backbone_model_configs.<BACKBONE> → model.checkpoint_path. Depends on the base arm's train job.
        # The resume-guard skips it once the merge's student_encoder.pt exists → a re-run just evals.
        if arm in MERGE_ARMS:
            _rec = merge_recipe(arm)
            _base_dir = ARM2DIR[_rec["base"]]
            _wf_dir = ARM2DIR[arm]
            _yx = "scripts/lib/yaml_extract.py"
            _frozen = f"$({_yx} $({_yx} configs/pipeline.yaml backbone_model_configs.{BACKBONE}) model.checkpoint_path)"
            # 2026-06-23: route through output_paths.train_dir — the 2026-06-20 backbone-first restructure
            # moved encoders to <bb>_<size>/train/<arm>; the old outputs/<mtag>/<BACKBONE>/<arm> dropped BOTH
            # the size suffix (_1B) AND the train/ level, so the merge couldn't find intervene's surgery ckpt
            # on the 1B fresh node (matches the resume-check at the T:-train block, single source).
            _tr = _train_dir(mtag, BACKBONE)
            _pred = (f"--surgery-pred-ckpt {_tr}/{_base_dir}/m09c_ckpt_best.pt "
                     if _rec["predictor"] else "")
            jobs[jid] = dict(
                id=jid, kind="train", deps={f"T:{BACKBONE}:{_rec['base']}"}, needs_labels=False,
                cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} {{pin}}python -u src/utils/wiseft_merge.py "
                     f"--alpha {_rec['alpha']} --frozen-ckpt {_frozen} "
                     f"--surgery-ckpt {_tr}/{_base_dir}/student_encoder.pt "
                     f"{_pred}--out-dir {_tr}/{_wf_dir}"),
                log=f"logs/ngpu_run_{mtag}_merge_{arm}_{{ts}}.log")
            continue
        # iter18: EVERY non-pretrain arm inits from pretrain's m09a_ckpt_best.pt → all dep on it.
        deps = set() if arm == "pretrain_encoder" else {seed_id}
        jobs[jid] = dict(
            id=jid, kind="train", deps=deps, needs_labels=(jid != seed_id),
            cmd=(f"CUDA_VISIBLE_DEVICES={{gpu}} NGPU_CONCURRENCY={{conc}} BACKBONE={BACKBONE} CACHE_POLICY_ALL={{cache}} "
                 f"{{pin}}./scripts/run_train.sh {arm} {mflag}"),
            log=f"logs/ngpu_run_{mtag}_train_{arm}_{{ts}}.log")
    # iter18 (2026-06-21): standalone label-bootstrap via run_eval (see LABELS_ONLY_SKIP). No deps → runs
    # FIRST; every label-gated job's labels_ready gate unblocks once L: writes action+taxonomy labels.
    # 2026-06-23: ALWAYS add it (was gated on the cross-set retest only). The labels used to come as a
    # SIDE-EFFECT of the pretrain TRAIN job — but POC --cache 1 resume-skips that pre-existing seed, so the
    # labels were NEVER generated → every label-gated job stayed un-ready → the run "settled" with done=7
    # and §3 ran on 0 evaled encoders (FATAL). Decoupling labels from training covers the skipped-seed path.
    # Cache no-op when labels already exist. Frozen name is backbone-scoped via enc_name (1B keeps its vitg
    # tag; the 2B champion drops it) — the old hardcoded vjepa_2_1_frozen was 2B-only.
    jobs["L:labels"] = dict(
        id="L:labels", kind="eval", deps=set(), needs_labels=False,
        cmd=(f"EVAL_CORPUS={EVAL_CORPUS} CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={LABELS_ONLY_SKIP} "
             f"CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {enc_name('frozen')}"),
        log=f"logs/ngpu_run_{mtag}_labels_{{ts}}.log")
    # eval jobs (iter18 2026-06-07 metric-parallel): per encoder = ONE E: job (stages 2-8) + 6 P:
    # jobs (Stage 8b, one predictor_temporal metric each, fanned across GPUs). E: and all 6 P: gate
    # on the SAME dep (the encoder's TRAIN job, or none for frozen) — NOT on E: — so Stage 8b runs
    # concurrently with stages 2-8. Each job gets its OWN deps set (set(deps)); --only mutates deps
    # in place, so a shared set object would cross-contaminate the 7 sibling jobs.
    for arm, enc in [("frozen", "frozen")] + list(ARM2ENC.items()):
        encn = enc_name(enc)
        deps = set() if arm == "frozen" else {f"T:{BACKBONE}:{arm}"}
        ejid = f"E:{encn}"
        jobs[ejid] = dict(
            id=ejid, kind="eval", deps=set(deps), needs_labels=True,
            cmd=(f"EVAL_CORPUS={EVAL_CORPUS} CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={EVAL_SKIP_PERENC_NO8B} "
                 f"CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {encn}"),
            log=f"logs/ngpu_run_{mtag}_eval_{encn}_{{ts}}.log")
        # iter19 2026-07-07: ONE combined 8b job (PT_METRIC=all) instead of 6 per-metric jobs, so m12e
        # computes the mask-independent encode h=encoder(pixel) ONCE per batch and SHARES it across the 5
        # reusable metrics (order self-encodes: it permutes frames). MEASURED 1.80× on the pt-metric compute
        # (34.8→19.3s/8clips), PROVEN bit-identical (scratchpad/hfull_parity: 0.0 diff). No parallelism lost:
        # on 2 GPUs the pt-metrics ran serially on one GPU anyway (the other runs the E: probe-fit). m12e
        # --metric all cache-skips already-done metrics internally, so --cache 1 resume still works per-metric.
        pjid = f"P:{encn}:all"
        jobs[pjid] = dict(
            id=pjid, kind="eval", deps=set(deps), needs_labels=True,
            cmd=(f"EVAL_CORPUS={EVAL_CORPUS} CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={EVAL_SKIP_ONLY_8B} PT_METRIC=all "
                 f"CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {encn}"),
            log=f"logs/ngpu_run_{mtag}_pt_{encn}_all_{{ts}}.log")
        # iter18 m12f revival (#1): Stage 8c fans into 4 F: jobs (one encoder_temporal metric
        # each), gated on TRAIN like P: — runs concurrent with stages 2-8. If a F: job lands
        # before this encoder's Stage-2 features exist, m12f's share-features falls back to
        # fresh identity forwards with an explicit reason (correct either way; in practice the
        # queue drains E: stage 2 long before F: slots open, so the share usually hits).
        for m in ET_METRICS:
            fjid = f"F:{encn}:{m}"
            jobs[fjid] = dict(
                id=fjid, kind="eval", deps=set(deps), needs_labels=True,
                cmd=(f"EVAL_CORPUS={EVAL_CORPUS} CUDA_VISIBLE_DEVICES={{gpu}} SKIP_STAGES={EVAL_SKIP_ONLY_8C} ET_METRIC={m} "
                     f"CACHE_POLICY_ALL={{cache}} {{pin}}./scripts/run_eval.sh {mflag} --encoders {encn}"),
                log=f"logs/ngpu_run_{mtag}_et_{encn}_{m}_{{ts}}.log")
    return jobs, mtag


def labels_ready(mtag):
    return ((REPO / f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_action')}/action_labels.json").exists()
            and (REPO / f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_taxonomy')}/taxonomy_labels.json").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["POC", "SANITY", "FULL"], default="POC")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--cache", choices=["1", "2"], default="2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, metavar="ARM",
                    help="run ONLY these train arms (no eval jobs, no §3 finale). Deps outside the "
                         "subset are trusted to the operator — run_train FATALs on a missing init "
                         "ckpt anyway. Used by runbook §0.A5 (1× box trains just pretrain_encoder) "
                         "and for single-arm crash re-runs.")
    ap.add_argument("--skip-arms", nargs="+", default=None, metavar="ARM",
                    help="iter18 2026-06-07 (user order: park CaSSLe, 213.8 s/step measured): drop "
                         "these train arms AND their eval jobs from the DAG; the §3 finale covers "
                         "only the remaining encoders. Skipped arms keep every on-disk anchor — "
                         "rerun later WITHOUT this flag + --cache 1 to train them and rebuild the "
                         "finale over all 14 encoders.")
    ap.add_argument("--taxheads-only", action="store_true",
                    help="iter18 (2026-06-20): build ONLY the taxonomy-head reuse source — one "
                         "Stage-11 (KEEP_PROBE_HEADS=1) eval job per encoder, fanned across --gpus, "
                         "no train + no other stages + no §3 finale. Replaces the ~10 h single-GPU "
                         "`run_eval.sh --encoders <all-17>` taxheads pass. Use --cache 2 for a fresh "
                         "build; --cache 1 resumes (run_eval skips encoders already done in the tree).")
    ap.add_argument("--etheads-only", action="store_true",
                    help="iter18 (2026-06-21): build ONLY the encoder-temporal head reuse source — one "
                         "Stage-8c (KEEP_PROBE_HEADS=1) eval job per encoder, fanned across --gpus, no "
                         "train + no other stages + no §3 finale. Runs on the TRAINED corpus (eval_10k "
                         "defaults — do NOT export the cross-set env first). Replaces the ~12-18 h single-GPU "
                         "8c pass. Use --cache 2 for a fresh build; --cache 1 resumes.")
    ap.add_argument("--eval-first", nargs="+", default=None, metavar="ARM",
                    help="iter19 SSL-head gate (Prof Das): run these encoders' eval jobs BEFORE any "
                         "pending arm training — every pending train job gains a dep on them, so on a "
                         "2-GPU box the gate evals grab both cards first, finish, THEN training starts, "
                         "landing the go/no-go numbers (e.g. pretrain vs frozen) BEFORE the ~20 h spend. "
                         "Each ARM must be already-trained (a resume-skipped 'pretrain_encoder') or "
                         "'frozen' (no train) — gating a pending arm on its own eval would cycle. "
                         "Composes with --skip-arms; mutually exclusive with --only.")
    args = ap.parse_args()
    if args.taxheads_only and args.etheads_only:
        sys.exit("FATAL: --taxheads-only and --etheads-only are mutually exclusive.")
    if (args.taxheads_only or args.etheads_only) and args.only:
        sys.exit("FATAL: --taxheads-only/--etheads-only and --only are mutually exclusive.")
    if args.eval_first and args.only:
        sys.exit("FATAL: --eval-first and --only are mutually exclusive (--only drops all eval jobs).")
    if args.eval_first and (args.taxheads_only or args.etheads_only):
        sys.exit("FATAL: --eval-first and --taxheads-only/--etheads-only are mutually exclusive.")

    jobs, mtag = build_jobs(args.mode, taxheads_only=args.taxheads_only, etheads_only=args.etheads_only)

    if args.only:
        bad = [a for a in args.only if a not in ARM2ENC]
        if bad:
            sys.exit(f"FATAL: unknown arm(s) {bad} — choices: {sorted(ARM2ENC)}")
        keep = {f"T:{BACKBONE}:{a}" for a in args.only}
        jobs = {jid: j for jid, j in jobs.items() if jid in keep}
        for j in jobs.values():
            j["deps"] &= keep   # outside-subset deps: operator's responsibility (FAIL LOUD in run_train)
        print(f"  [--only] restricted to {sorted(args.only)} — eval jobs + §3 finale SKIPPED", flush=True)

    if args.skip_arms:
        if args.only:
            sys.exit("FATAL: --skip-arms and --only are mutually exclusive.")
        bad = [a for a in args.skip_arms if a not in ARM2ENC]
        if bad:
            sys.exit(f"FATAL: unknown arm(s) {bad} — choices: {sorted(ARM2ENC)}")
        if "pretrain_encoder" in args.skip_arms:
            # iter19 2026-07-08 (user order): the seed CAN be eval-skipped ONCE it is already trained — a
            # pure-eval resume where every arm that inits from it also already has its student_encoder.pt.
            # Skipping an UNtrained seed still FATALs (a dependent's train job would then have no init ckpt);
            # a dependent that is somehow still untrained FATALs loud in run_train on the missing init (same
            # trust model as --only), never silently here. Ckpt-check mirrors the resume detection below.
            _seed_ckpt = REPO / f"{_train_dir(args.mode, BACKBONE)}/{ARM2DIR['pretrain_encoder']}/student_encoder.pt"
            if not _seed_ckpt.exists():
                sys.exit("FATAL: cannot skip pretrain_encoder — it is every arm's init dependency and its "
                         f"trained ckpt is absent ({_seed_ckpt}). Skip it only on a resume where the seed "
                         "(and its dependents) are already trained.")
            print("  [--skip-arms] pretrain_encoder eval-skipped — trained seed ckpt present (pure-eval "
                  "resume); its already-trained dependents no longer need it to init.", flush=True)
        drop = ({f"T:{BACKBONE}:{a}" for a in args.skip_arms}
                | {f"X:{enc_name(ARM2ENC[a])}" for a in args.skip_arms}   # --taxheads-only jobs
                | {f"Y:{enc_name(ARM2ENC[a])}" for a in args.skip_arms}   # --etheads-only jobs
                | {f"E:{enc_name(ARM2ENC[a])}" for a in args.skip_arms}
                | {f"P:{enc_name(ARM2ENC[a])}:all" for a in args.skip_arms}
                | {f"F:{enc_name(ARM2ENC[a])}:{m}" for a in args.skip_arms for m in ET_METRICS})
        jobs = {jid: j for jid, j in jobs.items() if jid not in drop}
        print(f"  [--skip-arms] dropped {sorted(args.skip_arms)} (train+eval+8b/8c-metrics) — §3 finale "
              f"will cover the remaining encoders only", flush=True)

    # disk-preflight (per mode × cache): POC = 13 arms × ~14G ckpts ≈ 185G + eval artifacts ≈ ~210G;
    # SANITY is throwaway-tiny. FULL cache=1 (RESUME): the frame + tube caches are now HARD-CAPPED
    # (video_io + factor_streaming LRU; pipeline.yaml max_cache_gb 200/250), so they no longer scale
    # unbounded with the corpus. On resume an over-cap cache LRU-trims to its cap on the first training
    # stores (frees 150G+ here), so the old 350G floor was obsolete + deadlocked resume when the on-disk
    # caches were oversized — a small "can-start" floor is correct. cache=2 (fresh full-eval build from
    # scratch) keeps its larger floor. Tune from the measured footprint if it trips.
    import shutil
    free_gb = shutil.disk_usage(str(REPO)).free / 1e9
    REQ_GB = {"POC": {"1": 80, "2": 250}, "SANITY": {"1": 30, "2": 30}, "FULL": {"1": 90, "2": 500}}
    req_gb = REQ_GB[args.mode][args.cache]
    # iter19 2026-07-07: the eval frame cache LRU-grows to its HARD cap (probe.eval_frame_cache.max_cache_gb),
    # now sized to hold the full ~2.25TB eval working set so every metric re-read HITS instead of re-decoding
    # (the 4x eval blow-up vs POC was cache-miss thrash at the old 200G cap, NOT the eval set). The disk MUST
    # be able to hold that growth or the eval ENOSPCs mid-run → require free ≥ (cap − on-disk-now). FAIL LOUD
    # BEFORE the run so the operator provisions the ≥3TB disk. Only when eval jobs are actually scheduled.
    if any(jid.startswith(("E:", "P:", "F:")) for jid in jobs):
        _efc = get_pipeline_config()["probe"]["eval_frame_cache"]
        _cache_dir = REPO / get_pipeline_config()["data"]["local_data_dir"] / _efc["subdir"]
        _now_gb = 0.0
        if _cache_dir.is_dir():
            with os.scandir(_cache_dir) as _it:
                _now_gb = sum(f.stat().st_size for f in _it if f.is_file()) / 1e9
        req_gb = max(req_gb, (_efc["max_cache_gb"] - _now_gb) + 40)   # +40G margin: eval outputs / ckpt churn
        print(f"  [disk-preflight] eval-frame-cache cap={_efc['max_cache_gb']}G · on-disk={_now_gb:.0f}G "
              f"→ must hold {_efc['max_cache_gb'] - _now_gb:.0f}G more (full re-read set → per-clip parity)",
              flush=True)
    print(f"  [disk-preflight] free={free_gb:.0f}G · required≈{req_gb:.0f}G", flush=True)
    if free_gb < req_gb and not args.dry_run:
        _cap = get_pipeline_config()["probe"]["eval_frame_cache"]["max_cache_gb"]
        sys.exit(f"FATAL: insufficient disk — {free_gb:.0f}G free < {req_gb:.0f}G for {args.mode}. The eval "
                 f"frame cache must be able to grow to its {_cap}G cap (holds the full eval working set so "
                 f"metrics HIT, not re-decode → per-clip parity with POC). Provision a larger disk.")

    # cpu-preflight: each job spawns ~6-8 CPU threads (TAR decode + factor-streaming workers).
    cores = os.cpu_count() or 0
    need_cores = args.gpus * 6
    if cores < need_cores:
        print(f"  ⚠️  [cpu-preflight] {cores} cores < ~{need_cores} (gpus×6) — TAR-decode will "
              f"contend at full fan-out; wall-time will slip.", flush=True)
    else:
        print(f"  [cpu-preflight] {cores} cores ≥ ~{need_cores} (gpus×6) — OK", flush=True)

    # ram-preflight (iter19 2026-07-04): the RAM sibling of disk/cpu. m09a1 preloads the val split
    # into HOST RAM ("Collecting N val clips into memory") — a DATA-SCALED anon footprint that OOM'd
    # iter19's FULL seed (5,750 val clips ≈ 40 G) while SANITY's 232-clip val never neared the cap
    # (a resource-scaling bug no small-data smoke catches). Check the val-preload upper bound (bounded
    # by validation.max_val_clips) vs the cgroup ANON cap so an oversized cap / small box FATALs HERE,
    # not 3 min into a 19 h run. Projects the FULL footprint even from a SANITY invocation (#4).
    from utils.ram_preflight import estimate_for_backbone
    _ram = estimate_for_backbone(BACKBONE)
    _clipspec = (f"max_val_clips={_ram['max_val_clips']} × {_ram['per_clip_mb']:.1f}MB/clip "
                 f"@ {_ram['num_frames']}f×{_ram['crop']}²")
    if _ram["cap_gb"] is None:
        print(f"  [ram-preflight] no cgroup cap (unlimited host) · val-preload≈{_ram['val_preload_gb']:.0f}G "
              f"({_clipspec})", flush=True)
    else:
        _valpct = 100.0 * _ram["val_preload_gb"] / _ram["cap_gb"]
        print(f"  [ram-preflight] cgroup_cap={_ram['cap_gb']:.0f}G · val-preload≈{_ram['val_preload_gb']:.0f}G "
              f"({_valpct:.0f}% of cap; {_clipspec}) · anon-now≈{_ram['anon_now_gb'] or 0:.0f}G", flush=True)
        # Threshold calibrated to the iter19 incident: the uncapped 5,750-clip val was ~30% of cap and THAT
        # thrashed — because the non-val anon baseline (producer + model + optimizer) also measured ~30% of cap
        # (anon-now ≈41G/128G at max_val_clips=1000), so val+baseline blew the reclaim headroom. FATAL when the
        # val preload alone exceeds 25% of cap (⇒ val+baseline > ~55%, the thrash zone); WARN at 12%. A valid
        # run (max_val_clips=1000 ≈ 5%) clears both.
        if _valpct > 25.0 and not args.dry_run:
            sys.exit(f"FATAL: val preload ≈{_ram['val_preload_gb']:.0f}G = {_valpct:.0f}% of the {_ram['cap_gb']:.0f}G "
                     f"cgroup cap — with the ~30%-of-cap producer+model+optimizer baseline it WILL OOM-thrash "
                     f"(iter19 incident). Lower validation.max_val_clips (base_optimization.yaml) or use a bigger-RAM box.")
        elif _valpct > 12.0:
            print(f"  ⚠️  [ram-preflight] val preload is {_valpct:.0f}% of the cap — tight once the "
                  f"producer+model baseline stacks on; watch the oom-watchdog's anon% (not memory.current).", flush=True)

    done, failed, running = set(), {}, {}   # running: gpu→(jid, Popen, logfile)

    # iter18 (2026-06-20) backbone-first tree: a cross-set corpus (EVAL_CORPUS=subset_10k) reuses the
    # encoders at <bb>/train/ — with --cache 1 the train-resume below finds them and skips training, while
    # the P:/F: resume markers check eval/<corpus>/ so a fresh corpus is never falsely skipped.
    if args.cache == "1":
        for jid, j in jobs.items():
            if j["kind"] != "train":
                continue
            _, bb, arm = jid.split(":")
            # iter19 2026-07-08 (user order): the "is this arm trained?" POLICY lives WITH the trainers, in
            # utils.clear_resume_anchors.is_finalized (student_encoder.pt + *ckpt_best* present, no surviving
            # resume anchors) — this orchestrator only ASKS, it does NOT re-implement the check. A crashed-at-
            # finalize arm (peft_lora: student but no _best + a surviving ckpt_stage0) is NOT finalized → its
            # train job RE-RUNS (resume→finalize), never laundered into 'already-trained' off student alone
            # (which is written BEFORE the _best save). Warn loudly with the exact missing artifacts.
            _adir = REPO / f"{_train_dir(args.mode, bb)}/{ARM2DIR[arm]}"
            if is_finalized(_adir):
                done.add(jid)
            elif (_adir / "student_encoder.pt").exists():
                print(f"  ⚠️  [resume] {arm}: NOT a clean finalize — {finalize_missing(_adir)}. Its train job "
                      f"RE-RUNS to finalize (its predictor metrics had been SILENTLY skipped).", flush=True)
        n_train_done = len(done)
        if n_train_done:
            print(f"  [resume --cache 1] skipping {n_train_done} already-trained arms: "
                  f"{sorted(d.split(':')[2] for d in done)}", flush=True)
        # P: (Stage-8b single-metric) jobs are done when their aggregate_<metric>.json exists — skip
        # them on resume so a restart never relaunches the ~84 no-op m12e processes. (E: jobs re-run
        # but cache-skip their stages internally via CACHE_POLICY_ALL=1, so they stay schedulable.)
        pt_done = 0
        for jid in jobs:
            if not jid.startswith("P:"):
                continue
            enc_nm, metric = jid[2:].rsplit(":", 1)   # NOT 'enc_name' — that's the module helper; a main()-local shadows it → UnboundLocalError at the skip-arms block
            if (REPO / f"{_eval_dir(args.mode, BACKBONE, EVAL_CORPUS, 'predictor_temporal')}/{enc_nm}/aggregate_{metric}.json").exists():
                done.add(jid)
                pt_done += 1
        if pt_done:
            print(f"  [resume --cache 1] skipping {pt_done} already-done Stage-8b metric jobs",
                  flush=True)
        # F: (Stage-8c encoder_temporal) jobs — same done-marker pattern (m12f writes
        # aggregate_<metric>.json per metric; tcc's is aggregate_tcc.json).
        et_done = 0
        for jid in jobs:
            if not jid.startswith("F:"):
                continue
            enc_nm, metric = jid[2:].rsplit(":", 1)
            if (REPO / f"{_eval_dir(args.mode, BACKBONE, EVAL_CORPUS, 'encoder_temporal')}/{enc_nm}/aggregate_{metric}.json").exists():
                done.add(jid)
                et_done += 1
        if et_done:
            print(f"  [resume --cache 1] skipping {et_done} already-done Stage-8c metric jobs",
                  flush=True)
        # E: (Stage 2-8 head-metric bundle) jobs — iter19 2026-07-08 (user order): skip on resume when the
        # encoder's head-metric outputs already exist (utils.output_paths.head_stage_done — the SINGLE 'is E: done'
        # source, shared with the status pane's 2-8 cell so they can't drift). WITHOUT this the E: job was NEVER
        # resume-skipped (only P:/F: were) → it re-launched and stage-2 re-extracted the NON-DURABLE test features
        # (~2-3h/enc of waste): a completed frozen eval re-ran its 23k-clip feature pass every restart. Nothing
        # depends on E:, so marking a done bundle done is safe; any marker absent ⇒ E: still runs (a fresh encoder).
        e_done = 0
        for jid in jobs:
            if jid.startswith("E:") and head_stage_done(args.mode, BACKBONE, EVAL_CORPUS, jid[2:]):
                done.add(jid)
                e_done += 1
        if e_done:
            print(f"  [resume --cache 1] skipping {e_done} already-done E: (Stage 2-8 head-metric) job(s)",
                  flush=True)

    # iter19 (2026-07-05) --eval-first SSL-head gate (Prof Das): hold every PENDING arm-train job until
    # the named encoders' eval jobs finish, so the go/no-go numbers (e.g. pretrain vs frozen) land BEFORE
    # the ~20 h train spend, not after it. Pure dependency injection — the ready() gate does the ordering,
    # so no dict-reorder needed. Runs AFTER the --cache 1 resume so `done` is populated (resume-skipped seed).
    # Legal gate encoders: a resume-skipped seed or 'frozen' (train done/absent → no train⇢eval⇢train cycle).
    if args.eval_first:
        gate_encs, gate_ids = [], set()
        for a in args.eval_first:
            if a == "frozen":
                gate_encs.append(enc_name("frozen"))
            elif a in ARM2ENC:
                if f"T:{BACKBONE}:{a}" in jobs and f"T:{BACKBONE}:{a}" not in done:
                    sys.exit(f"FATAL: --eval-first {a} but its train job is still PENDING — gating a "
                             f"train on its own eval cycles. Only a resume-skipped seed or 'frozen' can gate.")
                gate_encs.append(enc_name(ARM2ENC[a]))
            else:
                sys.exit(f"FATAL: --eval-first unknown arm {a} — choices: frozen + {sorted(ARM2ENC)}")
        for encn in gate_encs:
            gate_ids |= {jid for jid in jobs
                         if jid == f"E:{encn}" or jid.startswith(f"P:{encn}:") or jid.startswith(f"F:{encn}:")}
        gate_ids -= done   # already-evaled (resume) gate jobs don't block anything
        if not gate_ids:
            print(f"  [--eval-first] all gate evals for {args.eval_first} already done (resume) — no hold added",
                  flush=True)
        else:
            n_gated = 0
            for jid, j in jobs.items():
                if j["kind"] == "train" and jid not in done:
                    j["deps"] |= gate_ids
                    n_gated += 1
            print(f"  [--eval-first] SSL-head GATE: {len(gate_ids)} eval job(s) for {args.eval_first} run "
                  f"FIRST → {n_gated} pending train job(s) now wait on them (go/no-go before the train spend)",
                  flush=True)

    # CPU-set pinning (iter18 2026-06-07): one private core slice per GPU slot —
    # NGPU_CPUSET tells stream_autotune the cores are private (divide by 1, not conc).
    slots = cpu_slots(args.gpus)
    if slots[0]:
        print("  [cpuset] GPU slots pinned: " +
              " · ".join(f"GPU{g}→{s.split(',')[0]}..{s.split(',')[-1]}({len(s.split(','))}t)"
                         for g, s in enumerate(slots)), flush=True)

    def pin_for(g):
        return (f"NGPU_CPUSET={len(slots[g].split(','))} taskset -c {slots[g]} "
                if slots[g] else "")

    free = list(range(args.gpus))
    print(f"═══ N-GPU scheduler · mode={args.mode} · backbone={BACKBONE} · corpus={EVAL_CORPUS} · "
          f"gpus={args.gpus} · cache={args.cache} · {len(jobs)} jobs "
          f"({sum(j['kind'] == 'train' for j in jobs.values())} train + "
          f"{sum(j['kind'] == 'eval' for j in jobs.values())} eval) ═══", flush=True)

    def ready(jid):
        j = jobs[jid]
        if jid in done or jid in failed or any(jid == r[0] for r in running.values()):
            return False
        if not j["deps"] <= done:
            return False
        if j["needs_labels"] and not labels_ready(mtag):
            return False
        return True

    if args.dry_run:
        print("[dry-run] dependency-ordered launch plan (no GPU work):")
        sim_done, wave = set(), 0
        while len(sim_done) < len(jobs):
            launchable = [j for j in jobs if j not in sim_done and jobs[j]["deps"] <= sim_done]
            if not launchable:
                print(f"  !! stuck — unmet deps: {[j for j in jobs if j not in sim_done]}")
                sys.exit(1)
            print(f"  wave {wave} ({len(launchable)} ready): {sorted(launchable)}")
            sim_done |= set(launchable)
            wave += 1
        print(f"[dry-run] {len(jobs)} jobs across {wave} dependency waves. OK.")
        return

    t0 = time.time()
    # FAIL-FAST (iter18 2026-06-08, user order): the FIRST job failure aborts the WHOLE run — stop
    # launching new jobs, SIGTERM every still-running job, and exit(1) — mirroring run_train.sh /
    # run_eval.sh's own `set -euo pipefail`. The OLD behaviour (keep launching all 95, report the
    # failures only at the very end, then skip §3) buried the first failure under ~80 later launches,
    # which read as "still making progress" — misleading. Now the run stops AT the first ✗.
    abort = None
    retried = set()   # one-shot retry bookkeeping: a jid here has already burned its single retry
    while len(done) + len(failed) < len(jobs):
        for jid in list(jobs):
            if abort or not free or not ready(jid):
                continue
            g = free.pop(0)
            log = jobs[jid]["log"].format(ts=ts())
            cmd = jobs[jid]["cmd"].format(gpu=g, cache=args.cache, conc=args.gpus, pin=pin_for(g))
            print(f"[{now()}] GPU{g} ◀ {jid}  → {log}", flush=True)
            lf = open(REPO / log, "w")
            # start_new_session: each job is its OWN process group, so a fail-fast abort can SIGTERM
            # the whole tree (shell + python + DataLoader/decode workers), not just the wrapper shell.
            p = subprocess.Popen(cmd, shell=True, cwd=str(REPO), stdout=lf,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            running[g] = (jid, p, lf)
        for g in list(running):
            jid, p, lf = running[g]
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[g]
            free.append(g)
            if rc == 0:
                done.add(jid)
                print(f"[{now()}] GPU{g} ✓ {jid}  ({len(done)}/{len(jobs)} done)", flush=True)
            else:
                # ONE-SHOT RETRY (iter18 2026-06-24, user order): a transient stall/hang — e.g. a CUDA or
                # decode hiccup near the end of m12f feature extraction (the peft_dora:pace 91%-freeze that
                # aborted a 17.5h run) — must NOT waste the whole multi-hour run. On a job's FIRST non-zero
                # exit, re-queue it ONCE: it is not added to done/failed and is already removed from
                # running above, so ready(jid) is True again and the dispatch loop re-launches it next
                # iteration — a FRESH timestamped log, and --cache reuses any partial m12f/checkpoint so the
                # retry RESUMES, not restarts. Only a SECOND failure of the SAME job trips the fail-fast
                # abort below, so a deterministic bug still stops the run after one confirming retry while a
                # one-off blip self-heals. Keeps the user's fail-fast intent (broken run stops fast) without
                # punishing transient flakiness on a costly box.
                if jid not in retried:
                    retried.add(jid)
                    print(f"[{now()}] GPU{g} ✗ {jid} rc={rc} — RETRY 1/1 (transient?), re-queueing "
                          f"(deps intact; --cache {args.cache} resumes any partial work)", flush=True)
                else:
                    failed[jid] = rc
                    print(f"[{now()}] GPU{g} ✗ {jid} rc={rc} (retry also failed) — see its log",
                          flush=True)
                    if abort is None:
                        abort = jid      # second failure of the same job → trip the fail-fast abort
        if abort is not None:
            if running:
                print(f"  ⛔ FAIL-FAST: {abort} failed (rc={failed[abort]}) — SIGTERM "
                      f"{len(running)} still-running job(s) + aborting the run", flush=True)
                for _g, (_jid, _p, _lf) in list(running.items()):
                    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                        os.killpg(os.getpgid(_p.pid), signal.SIGTERM)   # whole process group
                    _lf.close()
                running.clear()
            break
        if running or any(ready(j) for j in jobs):
            time.sleep(10)
        elif free and not running:   # nothing ready, nothing running → a failed dep blocks the rest
            break

    el = (time.time() - t0) / 3600
    if abort is not None:
        print(f"═══ ⛔ FAIL-FAST ABORT after {el:.1f}h · {abort} failed (rc={failed[abort]}) · "
              f"{len(done)} job(s) done before the abort · §3 finale NOT run ═══", flush=True)
        print(f"  ✗ first failure: {abort} — read its per-job log (named on its GPU ◀ line above).")
        print(f"  Fix the cause, then re-run with --cache 1 to keep the {len(done)} completed job(s).")
        sys.exit(1)
    print(f"═══ jobs settled in {el:.1f}h · done={len(done)} failed={len(failed)} ═══", flush=True)

    if args.only:
        print(f"═══ --only run complete ({sorted(args.only)}) — §3 finale skipped by design ═══",
              flush=True)
        return

    if args.taxheads_only:
        print(f"═══ --taxheads-only complete — {len(done)}/{len(jobs)} taxonomy-head job(s) done · "
              f"§3 finale skipped (reuse-source build only) ═══", flush=True)
        return

    if args.etheads_only:
        print(f"═══ --etheads-only complete — {len(done)}/{len(jobs)} encoder-temporal head job(s) done · "
              f"§3 finale skipped (reuse-source build only) ═══", flush=True)
        return

    # ── §3 finale: ONE run_eval over ALL encoders, per-encoder stages skipped (cached) →
    #    runs the shared paired-Δ + m13 stages with every encoder present. cache=1 keeps caches.
    #    --skip-arms: the finale covers only the encoders whose evals ran (paired-Δ/m13
    #    handle encoder subsets — the live 15-min preview exercises this same path daily).
    _skip = set(args.skip_arms or [])
    all_encs = " ".join([enc_name("frozen")]
                        + [enc_name(e) for a, e in ARM2ENC.items() if a not in _skip])
    s3_log = f"logs/ngpu_run_{mtag}_s3_{ts()}.log"
    s3 = (f"EVAL_CORPUS={EVAL_CORPUS} SKIP_STAGES={S3_SKIP_PERENC} CACHE_POLICY_ALL=1 "
          f"./scripts/run_eval.sh --{args.mode} --encoders \"{all_encs}\"")
    print(f"═══ all jobs PASSED — §3 finale (shared paired-Δ + m13) → {s3_log} ═══", flush=True)
    rc = subprocess.call(f"set -o pipefail ; ( {s3} ) 2>&1 | tee {s3_log}",
                         shell=True, executable="/bin/bash", cwd=str(REPO))
    print(f"═══ §3 rc={rc} · total wall {el:.1f}h + §3 · plots → "
          f"{_eval_dir(args.mode, BACKBONE, EVAL_CORPUS, 'probe_plot')}/eval/ ═══", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
