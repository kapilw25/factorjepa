#!/usr/bin/env python3
"""Live status for the N-GPU scheduler (SANITY / POC / FULL) — emoji TABLE + live-calibrated ETA per job
and for the whole run. Companion to scripts/ngpu_run.py; works for the iter18 ablations AND the iter19
full-scale run.

Imports build_jobs() + the naming/corpus helpers from ngpu_run.py so the job list is the SINGLE SOURCE OF
TRUTH (it can never drift from the scheduler). The mode comes from --mode; the corpus + backbone are
derived from the same single sources the scheduler uses (pipeline.yaml + ITER18_BACKBONE), cross-checked
against the run banner. Classifies every job ✅/🔄/⬚/❌ from the main log's GPU ◀/✓/✗ markers.

ETA: a forward DAG simulation over the GPU pool. Per-arm durations are OBSERVED, not hardcoded —
  · a completed job's measured wall sets that arm-class's estimate (mean of completed peers),
  · a RUNNING train's total = static step ledger ÷ its own live rate (see REAL-ETA below),
  · a RUNNING/PENDING eval's total = the eval STAGE ledger: current-stage remainder
    (live clip bar) + Σ measured walls of the queued stages (stamp-banner timestamps),
  · only an arm-class with NO completion yet AND no parseable progress falls back to a prior (below),
    seeded from the 06-05/06-06 measured runs and replaced the moment real data arrives.

AUTO-BACKUP (POC + FULL — the runs worth keeping; SANITY is throwaway): while you watch, this backs up
  outputs/<mode> to HF every UPLOAD_EVERY_MIN minutes (`upload`, reuse mode — mirrors EVERY file incl. the
  resume checkpoints *ckpt_latest/stage*.pt, so a box migration needs no extra upload; HF dedups unchanged
  files. The full-fidelity `_full-*.tar` shards + `_full-manifest.json` on the remote are PROTECTED from its
  mirror-cleanup) and once more when the run finishes, so the paid node can be killed right after completion.
  A missing HF_TOKEN FAILS the backup loudly (rc=1 → ❌ line).

USAGE:
  python -u scripts/ngpu_run_status.py                 # latest POC main log
  python -u scripts/ngpu_run_status.py --mode FULL     # the iter19 full-scale run
  python -u scripts/ngpu_run_status.py --mode SANITY
  python -u scripts/ngpu_run_status.py --log logs/ngpu_run_poc_20260606_101530.log
  watch -n60 'python -u scripts/ngpu_run_status.py'    # live, refresh every 60s
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import json  # noqa: E402
from ngpu_run import (  # noqa: E402  (canonical DAG — single source for naming + jobs + the corpus derivation)
    ARM2DIR, ARM2ENC, BACKBONE, ET_METRICS, EVAL_CORPUS, S3_SKIP_PERENC, build_jobs, enc_name,
    enc_prefix)
from utils.config import get_pipeline_config, load_merged_config  # noqa: E402  (trainers' own loader)
from utils.arm_registry import display_arms              # noqa: E402  (single-source arm roster)
from utils.output_paths import (  # noqa: E402  (single source for the backbone-first tree)
    eval_dir as _eval_dir, eval_root as _eval_root, train_dir as _train_dir, head_stage_done)

EMOJI = {"done": "✅", "running": "🔄", "pending": "⬚", "failed": "❌"}
_MIN_EVAL_POINTS = 5     # eval rate needs a few clips before extrapolating
_MIN_RATE_STEPS = 3      # a live TRAIN step-rate needs ≥ this many spanned steps to be trusted
                         # (1-2 steps still carry the one-time stage startup → wild rate; 06-14)
_SIM_GUARD = 10000       # DAG-sim infinite-loop backstop
# iter18 (2026-06-14): roster from the SINGLE source (configs/arm_registry.yaml).
# TRAIN_ORDER  = arms with a real step plan (kind != merge) — drives the workload ledger, the priors and
#   the calibration. A kind=merge arm has NO arm_train_configs entry, so it MUST stay out of the ledger.
# DISPLAY_ORDER = EVERY roster arm incl. the post-hoc merge (wiseft) — drives the two rendered tables, so
#   wiseft's merge + its E:/P:/F: eval jobs show up (they were always in the DAG, just hidden here).
# MERGE_ARMS   = the kind=merge rows (wiseft) — priced by a small prior, not the step ledger.
TRAIN_ORDER = [a for a, _e, _g, _k in display_arms(include_merge=False)]
DISPLAY_ORDER = [a for a, _e, _g, _k in display_arms(include_merge=True)]
MERGE_ARMS = {a for a, _e, _g, _k in display_arms(include_merge=True) if _k == "merge"}
_HEAD_ARMS = {"surgery_3stage_DI_head", "surgery_noDI_head"}
# ── REAL-ETA WORKLOAD LEDGER (iter18 2026-06-07) ──────────────────────────
# The total work of every arm is DETERMINED by yaml + split artifacts — guessing it
# from log archaeology (priors, plan banners, hand-kept stage maps) is what caused the
# all-night ETA drift (hidden noDI stage2, hidden lpft FT stage, hopeful 5.2h priors).
# This ledger computes each arm's EXACT optimizer-step plan from the SAME single
# sources the trainers read, mirroring their formulas line-by-line:
#   · merged cfg via utils.config.load_merged_config (the trainers' own loader)
#   · m09a1:514-515   spe = max(1, n_pool // batch);            total = spe × max_epochs
#   · m09c2:525       spe = max(1, ceil(n_pool / batch))  (head arms)
#   · m09c1:811-812   spe = n_factor // batch  (n_factor = pool ∩ factor_manifest)
#   · m09c1:1270      stage_steps = max(int(total × max_epochs_pct), 1) per surgery stage
#   · m09c1:700-710   LP-FT stage0 PREPENDED at lp_ft_stage0.max_epochs_pct of total
#   · probe schedule  probe_every = spe // saves_per_epoch; per stage
#                     n = max(1, round(stage_steps / probe_every))  (the exact schedule)
# ETA then = (ledger_total − progress) ÷ measured rate + calibrated overheads — the
# field-standard "remaining work / measured throughput", never extrapolated totals.
_LEDGER_CACHE = {}


def _build_ledger(mtag):
    """{arm: {stages:[int], total:int, n_probes:int}} — static plan per arm."""
    if mtag in _LEDGER_CACHE:
        return _LEDGER_CACHE[mtag]
    pcfg = get_pipeline_config()
    model_cfg = pcfg["backbone_model_configs"][BACKBONE]
    local = Path(pcfg["data"]["local_data_dir"])
    pool = set(json.loads((local / "train_pool.json").read_text())["clip_keys"])
    fm_raw = json.loads((local / pcfg["data"]["factor_subdir"] / "factor_manifest.json").read_text())
    fm = set(fm_raw.keys()) if isinstance(fm_raw, dict) else {
        x["clip_key"] if isinstance(x, dict) else x for x in fm_raw}
    n_factor = len(pool & fm)
    led = {}
    for arm in TRAIN_ORDER:
        cfg = load_merged_config(model_cfg, pcfg["arm_train_configs"][arm])
        opt = cfg["optimization"]
        batch = opt["batch_size"][mtag] if isinstance(opt["batch_size"], dict) else opt["batch_size"]
        me = opt["max_epochs"]
        max_epochs = me[mtag] if isinstance(me, dict) else me
        sp = cfg["checkpoint"]["saves_per_epoch"]
        saves = sp[mtag] if isinstance(sp, dict) else sp
        if arm == "pretrain_encoder":
            spe = max(1, len(pool) // batch)                       # m09a1:514
            stages = [spe * max_epochs]
        elif arm in _HEAD_ARMS:
            spe = max(1, (len(pool) + batch - 1) // batch)         # m09c2:525 (ceil)
            stages = [spe * max_epochs]
        else:
            spe = n_factor // batch                                # m09c1:811
            total = spe * max_epochs
            lp = cfg["surgery"]["lp_ft_stage0"]
            stages = ([max(int(total * lp["max_epochs_pct"]), 1)] if lp["enabled"] else [])
            stages += [max(int(total * s["max_epochs_pct"]), 1)    # m09c1:1270
                       for s in cfg["surgery"]["stages"]]
        probe_every = max(1, spe // saves)
        n_probes = sum(max(1, round(st / probe_every)) for st in stages)
        led[arm] = {"stages": stages, "total": sum(stages), "n_probes": n_probes}
    _LEDGER_CACHE[mtag] = led
    return led


# Per-stage FINAL tqdm bars ("surgery:<name>: 100%|…| S/S [H:MM:SS<00:00" — tqdm omits
# the hours field under 1h, so HH is optional) — their elapsed sums to the arm's pure
# in-loop wall; (job wall − Σ) = startup + stage-end probes + finalize = the overhead.
# iter19 (2026-07-04): match ANY tqdm bar prefix (m09a1_pretrain_encoder:, surgery:, peft:, …), not
# just 'surgery:'. The pretrain seed's bar was 'm09a1_pretrain_encoder:' → NEVER matched → its live rate
# AND live progress fell through to the ~25s/step prior, pinning the ETA at ~20h vs the true ~14h. \S+ is
# the desc token (no internal spaces); the %|bar| cur/tot [elapsed< structure keeps it tqdm-specific.
_RE_FINAL_BAR = re.compile(r"\S+\s+100%\|[^|]*\|\s*(\d+)/\1\s*\[(?:(\d+):)?(\d+):(\d+)<")
_RE_LIVE_BAR = re.compile(r"\S+\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:(\d+):)?(\d+):(\d+)<")
# The trainer's OWN windowed rate (tqdm postfix). Prefix-agnostic + already smoothed → the authoritative
# live step-rate ("the real ETA the operator reads off the bar"). Beats span-estimate + prior.
_RE_TRAIN_RECENT = re.compile(r"recent=([\d.]+)s/step")


def _bar_secs(h, m, s):
    return (int(h) if h else 0) * 3600 + int(m) * 60 + int(s)


def _calibrate(jobs, done, launched, mtag, ledger):
    """Measured (pure step rate, per-pad overhead) from THIS run's completed train arms.
    pure_rate = Σ(final-bar elapsed)/executed_steps; overhead = wall − Σ(bar elapsed),
    split over (n executed stages + 2) pads (each stage-end probe + startup + finalize)."""
    rates, pads, fins = [], [], []
    if not ledger:
        return {"ledger": ledger, "rate": None, "pad": _FINALIZE_PAD_S,
                "finalize": None, "mtag": mtag}
    for jid, t in done.items():
        if t == "resume" or not jid.startswith("T:") or jid not in launched:
            continue
        arm = _arm_of(jid)
        if arm not in ledger:        # kind=merge (wiseft) completes with a log but has NO step ledger —
            continue                 # indexing ledger[arm]["total"] below would KeyError the whole watch
        cands = sorted((p for p in REPO.glob(jobs[jid]["log"].format(ts="*")) if p.exists()), key=lambda p: p.stat().st_mtime)  # if p.exists(): skip dangling symlinks
        if not cands:
            continue
        try:
            full = cands[-1].read_text(errors="replace")
        except OSError:
            continue
        wall = (_sod(t) - _sod(launched[jid])) % 86400
        # set(): tqdm prints the 100% bar line TWICE (last update + close) — without
        # dedupe every stage double-counts (06-07: rate read 128 s/step = 2× true, pad 0).
        bars = [_bar_secs(h, m, s) for _, h, m, s in set(_RE_FINAL_BAR.findall(full))]
        rm = re.findall(r"Resumed from step (\d+)", full)
        executed = ledger[arm]["total"] - (int(rm[-1]) if rm else 0)
        if not bars or executed <= 0:
            continue
        rates.append((sum(bars), executed))
        pads.append(max(wall - sum(bars), 0) / (len(bars) + 2))
        # MEASURED finalize: ✓stamp − newest stage-ckpt mtime (saved right before
        # finalize). 06-07 measured 23m/32m/37m/67m — replaces the fake 15m constant.
        mt = _newest_stage_ckpt_mtime(arm, mtag)
        if mt is not None:
            fins.append((_sod(t) - _epoch_sod(mt)) % 86400)
    pads.sort()
    fins.sort()
    pooled = (sum(b for b, _ in rates) / sum(e for _, e in rates)) if rates else None
    return {"ledger": ledger, "mtag": mtag,
            "rate": pooled,    # pooled Σbar-seconds / Σexecuted-steps across completions
            "pad": pads[len(pads) // 2] if pads else _FINALIZE_PAD_S,
            "finalize": fins[len(fins) // 2] if fins else None}


_STAGE_NOTE = {}   # jid → "sN/M" live stage marker, filled by _running_total, shown in the cell
# COLD-START fallback for the end-of-training finalize (stage-end probe-trio on 451 clips
# + best-ckpt reload + export). 06-07 MEASURED on this run's completed arms (✓stamp −
# last-stage-ckpt mtime): 23m / 32m / 37m / 67m — the old 15m constant was 2-4× fake.
# Used ONLY before any in-log completion; afterwards calib["finalize"] (live median) wins,
# and the remaining-finalize display counts DOWN from it anchored on the ckpt mtime.
_FINALIZE_PAD_S = 35 * 60


def _newest_stage_ckpt_mtime(arm, mtag):
    """mtime (epoch) of the arm's newest *ckpt_stage*.pt — written immediately before
    finalize starts, so (✓stamp − it) = measured finalize, (now − it) = time IN finalize."""
    d = REPO / f"{_train_dir(mtag, BACKBONE)}/{ARM2DIR[arm]}"
    cks = sorted((p for p in d.glob("*ckpt_stage*.pt") if p.exists()), key=lambda p: p.stat().st_mtime)  # if p.exists(): skip dangling/cleared ckpt symlinks
    return cks[-1].stat().st_mtime if cks else None


def _epoch_sod(epoch):
    import time as _t
    g = _t.gmtime(epoch)
    return g.tm_hour * 3600 + g.tm_min * 60 + g.tm_sec


# ── EVAL REAL-ETA STAGE LEDGER (iter18 2026-06-07) ────────────────────────
# A per-encoder eval is a FIXED stage chain (run_eval.sh per-encoder loop):
#   2 features → 3 probe-train → 3.5 taxonomy (= stage id 11) → 5 motion-feat
#   → 6 cosine → 8 future_mse → 8b predictor_temporal
# The old estimator extrapolated ONLY the current stage's clip bar and capped the
# job at a hardcoded 3 h — once elapsed passed 3 h the cell froze at the 60s floor
# ("🔄 3h05m·~1m00s" while s7/7 had ~8m left AND nothing counted queued stages).
# run_eval.sh stamps "═══ HH:MM:SS · STAGE <id>" before every stage (stamp(),
# run_eval.sh:311), so completed stage walls are MEASURED from consecutive banner
# timestamps (the "DONE · total wall" banner closes the last). A stage NO eval has
# completed yet borrows the live full-stage projection (clip-bar total × recent
# rate) from any sibling currently inside it; a stage no eval has even reached
# falls back to the whole-job prior's per-stage share — replaced the moment any
# sibling reaches it (the same self-correcting contract as the train priors).
EVAL_PLAN = [("3.5" if s == "11" else s) for s in S3_SKIP_PERENC.split(",")]
_RE_STAMP = re.compile(r"═══ (\d\d:\d\d:\d\d) ·\s*(?:STAGE ([\w.]+)|(DONE))")
_RE_CLIP_BAR = re.compile(r"(\d+)/(\d+) \[[^\]]*recent=([\d.]+)s/clip")


def _runtime_extra_skip():
    """EVAL_PLAN display-ids dropped at runtime by run_eval.sh's EXTRA_SKIP_STAGES env /
    logs/.eval_extra_skip sentinel (iter18 2026-06-08) — so a mid-run `echo '3,11' >
    logs/.eval_extra_skip` (skip STAGE 3 action-probe-train + STAGE 3.5 taxonomy) makes the ETA
    DROP by those stages without a scheduler restart. Skip-id 11 maps to plan id '3.5'."""
    raw = os.environ.get("EXTRA_SKIP_STAGES", "")
    f = REPO / "logs" / ".eval_extra_skip"
    if not raw and f.exists():
        try:
            raw = f.read_text().strip()
        except OSError:
            raw = ""
    return {("3.5" if s.strip() == "11" else s.strip()) for s in raw.split(",") if s.strip()}


def _eval_plan_for(jid, mtag):
    """Stage plan for one per-encoder E: eval — stages 2-8 ONLY (iter18 2026-06-07: Stage 8b
    is now 6 separate P: metric jobs, estimated via _pt_total — so 8b is dropped here to avoid
    double-counting). Stage 8 is ALSO dropped when the encoder's predictor-bearing best ckpt is
    PROVABLY absent (student_encoder.pt exists, m09{a,c}_ckpt_best.pt doesn't) — mirrors
    run_eval.sh's Stage-8 preflight. A still-training arm keeps the full 2-8 plan."""
    plan = [s for s in EVAL_PLAN if s != "8b"]    # E: runs 2,3,3.5,5,6,8 — never 8b
    plan = [s for s in plan if s not in _runtime_extra_skip()]   # runtime taxonomy/probe-skip → ETA drops
    enc = jid.split(f"E:{enc_prefix()}_", 1)[-1]
    if enc == "frozen":
        return plan               # Meta ckpt always carries the predictor
    arm = next(a for a, e in ARM2ENC.items() if e == enc)
    d = REPO / f"{_train_dir(mtag, BACKBONE)}/{ARM2DIR[arm]}"
    best = "m09a_ckpt_best.pt" if ARM2DIR[arm].startswith("m09a_") else "m09c_ckpt_best.pt"
    if (d / "student_encoder.pt").exists() and not (d / best).exists():
        return [s for s in plan if s != "8"]
    return plan


def _eval_calibrate(jobs, mtag, now_s):
    """(medians, state, plans) for the eval stage ledger.
    medians: stage → median measured wall, pooled over EVERY eval log segment of this
      run (banner-timestamp diffs; interrupted segments contribute their closed stages).
      Stages with no closed wall get the median live projection (bar total × recent).
    state[jid] (running evals only): {cur, in, rem_cur} — current stage, seconds inside
      it, and its live-bar remainder (None when the stage has no recent= clip bar; the
      bar is read ONLY from bytes after the last banner, so a finished stage's stale
      bar can never masquerade as the current one's)."""
    walls, projs, state, plans = {}, {}, {}, {}
    for jid, j in jobs.items():
        if _arm_of(jid) != "eval" or jid.startswith(("P:", "F:", "Y:", "X:", "L:")):
            continue                  # P:/F: metric + Y:/X: regen + L: label-bootstrap use their own ETA, not the E: stage ledger
        plans[jid] = _eval_plan_for(jid, mtag)
        cands = sorted((p for p in REPO.glob(j["log"].format(ts="*")) if p.exists()), key=lambda p: p.stat().st_mtime)  # if p.exists(): skip dangling symlinks
        for p in cands:
            try:
                txt = p.read_text(errors="replace")
            except OSError:
                continue
            seq = [( _sod(m.group(1)), m.group(2) or "DONE", m.end())
                   for m in _RE_STAMP.finditer(txt)
                   if (m.group(2) in plans[jid]) or m.group(3)]
            for (t1, s1, _), (t2, _s2, __) in zip(seq, seq[1:]):
                walls.setdefault(s1, []).append((t2 - t1) % 86400)
            if p is not cands[-1] or not seq or seq[-1][1] == "DONE":
                continue
            cur = seq[-1][1]
            es = {"cur": cur, "in": (now_s - seq[-1][0]) % 86400, "rem_cur": None}
            pb = _RE_CLIP_BAR.findall(txt[seq[-1][2]:])
            if pb:
                c, tot, r = int(pb[-1][0]), int(pb[-1][1]), float(pb[-1][2])
                if tot and c >= _MIN_EVAL_POINTS:
                    es["rem_cur"] = (tot - c) * r
                    projs.setdefault(cur, []).append(tot * r)
            state[jid] = es
    med = {s: sorted(v)[len(v) // 2] for s, v in walls.items()}
    for s, v in projs.items():
        med.setdefault(s, sorted(v)[len(v) // 2])
    return med, state, plans
# COLD-START priors (seconds), used ONLY until a live measurement (a completion, or a running job's
# step progress) replaces them. Empirical: 06-06 1× pretrain 4h32m; 06-05 enc arms ~5h10m;
# head arms ~58m; eval = per-encoder stages only (2,3,11,5,6,8,8b) — self-corrects on 1st completion.
PRIOR = {arm: (int(4.6 * 3600) if arm == "pretrain_encoder"
               else int(1.0 * 3600) if arm in _HEAD_ARMS
               else int(5.2 * 3600))
         for arm in TRAIN_ORDER}
PRIOR["eval"] = int(0.75 * 3600)
# SANITY arms are ~5-8 min each (06-06 measured: pretrain 6m20s, frozen eval 5m) — the POC
# priors above are ~50× too big there and made the first SANITY ETA read 10h46m. Mode-scaled
# priors fix the cold start; live measurements still override both the moment they exist.
PRIOR_SANITY = {k: 7 * 60 for k in PRIOR}
# FULL-scale cold priors (iter19 plan.md, grounded in the v5_1B POC logs extrapolated to 116k×1 epoch):
# per surgery/pretrain arm ≈ ~19 h (~2,700 steps @ ~25 s/step); head arms scale ~5.6× off POC's ~1 h ≈ ~5 h;
# a per-encoder eval ≈ ~2-3× the 10k eval ≈ ~2 h. Self-corrects on the first live measurement like the others.
PRIOR_FULL = {arm: (int(19 * 3600) if arm == "pretrain_encoder"
                    else int(5 * 3600) if arm in _HEAD_ARMS
                    else int(19 * 3600))
              for arm in TRAIN_ORDER}
PRIOR_FULL["eval"] = int(2 * 3600)
# A kind=merge arm (WiSE-FT) is a post-hoc weight interpolation, NOT a training run: it has no step
# ledger, so the REAL-ETA path can't price it. It's a ~1-3 min ckpt load+lerp+save — give it a small
# dedicated prior so its "train" cell reads ~3m (not the 45m whole-eval prior). Self-corrects once its
# own log is measured; mode-scaled like the others.
_MERGE_PRIOR = {"poc": 3 * 60, "sanity": 60, "full": 5 * 60}   # merge cost = ckpt load+lerp+save (model-size, not data) → ~POC
# COLD prior for ONE Stage-8b metric job (P:) — used ONLY until that metric's first completion
# populates pt_med (per-metric). The whole-eval prior (est["eval"], ~3.4h incl. 8b) is a terrible
# per-metric prior, so use a dedicated one: at bs=16 the 6 metrics run ~10-25 min each (teacher_free
# longest). 20 min is conservative (safe-high) and self-corrects within ~1 metric completion.
_PT_COLD_PRIOR = {"poc": 20 * 60, "sanity": 60, "full": 45 * 60}   # 8b metric on the ~15× FULL eval corpus
# COLD prior for ONE Stage-8c metric job (F:, m12f) — F: jobs are MUCH heavier than P: ones
# (they decode pixels + extract features for test AND train; aot/tov/pace train a probe head).
# Measured 06-12: 24GB box 1h30-2h15/job; 4×96GB box tcc 14m (test-only) but aot 47m+ still
# running at first render — the 20m P: prior under-read the 35 pending F: jobs ~3× and the run
# ETA showed 3h for ~8h of work. Self-corrects per metric on the first completion, and pending
# jobs borrow a RUNNING sibling's live projection (see the _proj seeding below) even sooner.
_ET_COLD_PRIOR = {"poc": 75 * 60, "sanity": 3 * 60, "full": 180 * 60}   # 8c m12f (decode+features test&train) on FULL
# COLD prior for ONE regen job (--etheads-only Y: = 4-metric Stage-8c m12f, multi-phase ~2.5h; --taxheads-only
# X: = Stage-11 only, ~50m/enc = ~15h/17). A single clip bar under-reads the multi-phase m12f, so price from the
# measured median of completed peers; this is just the cold seed before the first wave lands (iter18 2026-06-21).
_REGEN_COLD_PRIOR = {"etheads": {"poc": 150 * 60, "sanity": 4 * 60, "full": 375 * 60},
                     "taxheads": {"poc": 53 * 60, "sanity": 3 * 60, "full": 130 * 60}}

# Back up outputs/<mtag> to HF this often (minutes) WHILE the run goes, so the final backup at the end
# is tiny and the paid node can be killed right away. Driven by the 60s `watch` refresh + a stamp file.
UPLOAD_EVERY_MIN = 45   # iter18 2026-06-06: full-artifact backups are heavier — user chose 45m
# True ONLY while a manual HF upload runs (commit race); False = auto-backup outputs every 45m.
AUTO_BACKUP_DISABLED = False
# Rebuild the §3-style preview plots from whatever evals are DONE this often (minutes). CPU-only.
PLOT_EVERY_MIN = 15


def _latest_log(mtag):
    """Latest MAIN scheduler log — excludes the per-job logs the scheduler itself writes
    (ngpu_run_<mtag>_train_*/_eval_*/_pt_*/_s3_*). Matches both the B5 main tee
    (ngpu_run_poc_<ts>.log) and variants like _regate_/_only_pretrain_.
    iter18 2026-06-07: _pt_ (the metric-parallel P: job logs) MUST be excluded too — they are
    NEWER than the main tee and carry no GPU ◀/✓ markers, so picking one made every job read as
    pending (all train cells showed ⬚ despite the resume-skip of 10 trained arms).
    iter18 2026-06-12: same bug, third strike — _et_ (the Stage-8c F: job logs) joined the
    family with m12f and made the whole table read ⬚/0🔄 on the 4×96GB POC restart.
    iter18 2026-06-14: _merge_ (the post-hoc WiSE-FT merge job log) is the SAME family — no GPU markers, and
    for the ~3 min the merge runs it's the newest ngpu_run_poc*.log, so leaving it in would flip the table.
    iter19 2026-07-04: the exclusion MUST cover EVERY per-job segment ngpu_run.py emits, else that job's log
    (newer than the main tee, no GPU markers) is picked as 'main' and the whole table reads pending. Complete
    set (verified against ngpu_run.py's 9 log= templates): _train_ _eval_ _pt_ _et_ _s3_ _merge_ _labels_
    _taxheads_ _etheads_. (_wiseft_ was a STALE marker — the real segment is _merge_; and _labels_/_taxheads_/
    _etheads_ were missing, so a cross-set / reuse-source run mis-read the table.)"""
    cands = [p for p in (REPO / "logs").glob(f"ngpu_run_{mtag}*.log")
             if p.exists() and not any(seg in p.name
                                       for seg in ("_train_", "_eval_", "_pt_", "_et_", "_s3_",
                                                   "_merge_", "_labels_", "_taxheads_", "_etheads_"))]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _arm_consumed(jobs, mtag):
    """{jid: Σ seconds across ALL of that job's per-arm log files} (iter18 2026-06-07:
    '✅ (cached)' hid that a resume-skipped arm may have burned 12h+ in earlier runs).
    One log per LAUNCH — mid-run interruptions create several logs per arm (full_ft has
    4: _185942, _234435, _001232, _002339). Each segment = (log mtime − the UTC start
    timestamp in its filename); mtime = the last byte tee wrote = completion/kill moment.
    Symlinked logs (moved to result_outputs/) stat-follow to the target — mtimes intact.
    A LIVE arm's open segment is included but unused (the 🔄 cell path never reads it)."""
    totals = {}
    del mtag  # jid→log template already carries the mode tag
    for jid, j in jobs.items():
        secs = 0
        for p in REPO.glob(j["log"].format(ts="*")):
            m = re.search(r"_(\d{8}_\d{6})\.log$", p.name)
            if not m:
                continue
            start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            try:
                secs += max(p.stat().st_mtime - start.timestamp(), 0)
            except OSError:
                continue
        if secs:
            totals[jid] = secs
    return totals


def _sod(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def _dur(secs):
    secs = int(round(secs))
    if secs < 0:
        secs += 86400
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _prior_run_dur(arm):
    """Duration of a resume-skipped train arm (trained in a PRIOR run — e.g. the pretrain seed on
    another box — whose ◀/✓ aren't in THIS run's log, so its consumed=0 → the misleading
    '(prior run)'). Sum the ◀→✓ span of its T: job across archived scheduler logs (this run's
    logs/ + iter*/logs seed dirs), using the logs' OWN timestamps — mtime is unreliable (later
    sync/upload touches inflate it)."""
    total, seen = 0.0, set()
    for f in list(REPO.glob("logs/*seed*.log")) + list(REPO.glob("iter/*/logs/**/*seed*.log")):
        if f in seen:
            continue
        seen.add(f)
        try:
            txt = f.read_text(errors="ignore")
        except OSError:
            continue
        lau = re.findall(rf"\[(\d\d:\d\d:\d\d)\] GPU\d+ ◀ T:\S*:{re.escape(arm)}\b", txt)
        don = re.findall(rf"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✓ T:\S*:{re.escape(arm)}\b", txt)
        if lau and don:
            total += (_sod(don[-1]) - _sod(lau[0])) % 86400
    return total


def _arm_of(jid):
    return jid.split(":")[2] if jid.startswith("T:") else "eval"


def _tail(path, nbytes=8000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - nbytes))
            return f.read().decode(errors="replace")
    except Exception:
        return ""


def _eval_total(jid, elapsed, prior, ecalib):
    """TOTAL wall estimate for ONE eval job from the stage ledger (med, state, plans).
      · RUNNING (jid in state): elapsed + current-stage remainder (live clip bar, else
        median-minus-time-in-stage) + Σ medians of the QUEUED stages — the queued 8b
        (m12e ~2.2 h on 1825 clips) is now counted instead of capped away.
      · PENDING: Σ medians over the full per-encoder stage plan.
    A stage with no measured wall AND no live projection borrows the per-stage share of
    the whole-job prior (self-corrects the moment any sibling walks that stage)."""
    med, state, plans = ecalib
    plan = plans.get(jid, EVAL_PLAN)
    share = prior / max(len(EVAL_PLAN), 1)

    def w(s):
        return med.get(s, share)

    es = state.get(jid)
    if es is None:                                  # pending — whole plan
        return sum(w(s) for s in plan)
    cur = es["cur"]
    i = plan.index(cur) if cur in plan else 0
    rem_cur = es["rem_cur"] if es["rem_cur"] is not None else max(w(cur) - es["in"], 60)
    _STAGE_NOTE[jid] = f"s{i + 1}/{len(plan)}"
    return elapsed + rem_cur + sum(w(s) for s in plan[i + 1:])


def _pt_total(jobs, jid, elapsed, prior, pt_med):
    """TOTAL wall for a Stage-8b single-metric P: job (iter18 2026-06-07). Each P: log is ONE
    metric's 0→1825 m12e run, so the single clip bar (cur/tot/`recent=`) is an exact estimate —
    no stage ledger needed. Pending (no bar yet) → that metric's measured median (pt_med), else
    the prior. Metrics differ a lot in cost (teacher_free ≫ causal) so pt_med is keyed per-metric."""
    metric = jid.rsplit(":", 1)[-1]
    cands = sorted((p for p in REPO.glob(jobs[jid]["log"].format(ts="*")) if p.exists()), key=lambda p: p.stat().st_mtime)  # if p.exists(): skip dangling symlinks
    txt = _tail(cands[-1]) if cands else ""
    cp = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", txt)
    rr = re.findall(r"recent=([\d.]+)s/clip", txt)
    if cp and rr and int(cp[-1][1]) and int(cp[-1][0]) >= _MIN_EVAL_POINTS:
        cur, tot, rate = int(cp[-1][0]), int(cp[-1][1]), float(rr[-1])
        return elapsed + (tot - cur) * rate
    return max(pt_med.get(metric, prior), elapsed + 300)


_RE_M12F_BAR = re.compile(r"m12f\[([\w/+-]+)\]:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")


def _regen_running_total(jobs, jid, elapsed, cold):
    """Live TOTAL-wall projection for a RUNNING --etheads-only Y: job from its own m12f bars. An et-job is
    a fixed 4-phase chain (aot-tov-tcc/test · aot-tov/train · pace/test · pace/train), so
    progress = (phases-at-100% + current-phase fraction) / 4 and total ≈ elapsed / progress — the phases
    are near-equal (test ~40m, train ~44m). A --taxheads-only X: job has a NON-linear dim bar (dim-1
    carries the one-time feature extraction), so it keeps the cold prior. Prior until the first bar prints
    (iter18 2026-06-21)."""
    if not jid.startswith("Y:"):
        return max(cold, elapsed + 300)
    cands = sorted((p for p in REPO.glob(jobs[jid]["log"].format(ts="*")) if p.exists()),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        return max(cold, elapsed + 300)
    try:
        txt = cands[-1].read_text(errors="replace")
    except OSError:
        txt = _tail(cands[-1], 20000)
    bars = _RE_M12F_BAR.findall(txt)
    if not bars:
        return max(cold, elapsed + 300)
    done = {lbl for lbl, c, t in bars if int(c) >= int(t)}      # phase labels that reached 100%
    _lbl, cur, tot = bars[-1]
    progress = (len(done) / 4 if int(cur) >= int(tot)
                else (len(done) + int(cur) / max(int(tot), 1)) / 4)
    if progress < 0.02:
        return max(cold, elapsed + 300)
    return max(elapsed / progress, elapsed + 300)


def _running_total(jobs, jid, elapsed, prior, calib, ecalib):
    """Total-duration estimate for a RUNNING job.
      · TRAIN — REAL ETA (iter18 2026-06-07): remaining = (ledger_total − progress) ×
        live step rate + calibrated pads for the remaining stage-end probes + finalize.
        Total work comes from the STATIC ledger (yaml+artifacts) — never extrapolated,
        no hidden stages possible. Rate comes from the CURRENT stage's own bar.
      · EVAL — current-stage remainder + Σ queued-stage medians (the eval stage ledger,
        _eval_total); no more current-stage-only blindness or 3 h cap.
      · fallback — prior, capped (only before any in-log data exists)."""
    cap = max(prior * 2.5, elapsed + 600)
    if _arm_of(jid) == "eval":
        return _eval_total(jid, elapsed, prior, ecalib)
    tmpl = jobs[jid]["log"]
    cands = sorted((p for p in REPO.glob(tmpl.format(ts="*")) if p.exists()),
                   key=lambda p: p.stat().st_mtime)   # if p.exists(): skip a log deleted mid-glob (dangling) — .stat() on it crashes the whole watch
    txt = _tail(cands[-1]) if cands else ""
    led = (calib["ledger"] or {}).get(_arm_of(jid))
    if jid.startswith("T:") and led and cands:
        try:
            full = cands[-1].read_text(errors="replace")
        except OSError:
            full = txt
        stages, total = led["stages"], led["total"]
        rm = re.findall(r"Resumed from step (\d+)", full)
        base = int(rm[-1]) if rm else 0
        cont = re.findall(r"stage (\d+) continues at local step (\d+)", full)
        bar_init = int(cont[-1][1]) if cont else 0   # mid-stage anchor: bar starts at L
        if cont:
            base -= bar_init                         # L is already inside the bar's cur
        done_now = sum(int(n) for n in re.findall(r"Stage \w+ complete: (\d+) steps", full))
        # current stage index in the LEDGER (resumed prefix + stages completed this run)
        pre, k = 0, 0
        while k < len(stages) and pre + stages[k] <= base:
            pre += stages[k]
            k += 1
        idx = min(k + len(re.findall(r"Stage \w+ complete:", full)), len(stages) - 1)
        _STAGE_NOTE[jid] = f"s{idx + 1}/{len(stages)}"
        bar = _RE_LIVE_BAR.findall(txt)
        contrib, live_rate = 0, None
        if bar:
            cur, stage_tot = int(bar[-1][0]), int(bar[-1][1])
            n_full_bars = len(set(_RE_FINAL_BAR.findall(full)))
            n_complete = len(re.findall(r"Stage \w+ complete:", full))
            if cur < stage_tot:
                contrib = cur                          # live mid-stage bar
                # RECENT rate = Δelapsed / Δcur between the OLDEST and NEWEST bar samples in
                # the tail. Anchoring on the SPAN (subtract the oldest sample's elapsed) drops
                # the one-time stage STARTUP baked into the bar clock — a factor-stream worker
                # respin makes the bar read "0/175 [54:02<?]", so a cumulative bsec/cur over a
                # single step read ~3000 s/step and flashed a 394h ETA (06-14). Trust it only
                # once the span covers ≥ _MIN_RATE_STEPS real steps; else leave live_rate None
                # → fall back to the measured (calib) rate / prior below, never a 1-step guess.
                samples = [(int(c), _bar_secs(*hms)) for c, _t, *hms in bar]
                (c0, e0), (cN, eN) = samples[0], samples[-1]
                if cN - c0 >= _MIN_RATE_STEPS and eN > e0:
                    live_rate = (eN - e0) / (cN - c0)
            elif n_full_bars > n_complete:
                # current stage's bar hit 100% but its "complete" line hasn't printed
                # yet (stage-end probe running) → its steps aren't in done_now yet.
                contrib = stage_tot
        # iter19 (2026-07-04): the trainer's OWN windowed recent=…s/step is the authoritative live rate —
        # prefix-agnostic + pre-smoothed, so it beats the _MIN_RATE_STEPS-gated span estimate AND the
        # ~25s/step prior (what pinned the first arm's ETA at ~20h vs the true ~14h). Overrides live_rate.
        _recent = _RE_TRAIN_RECENT.findall(txt)
        if _recent:
            live_rate = float(_recent[-1])
        progress = min(base + done_now + contrib, total)
        steps_left = total - progress
        # remaining pads: one per not-yet-passed stage end + 1 finalize-equivalent
        ends_left = sum(1 for i, s in enumerate(stages)
                        if sum(stages[:i + 1]) > progress) + 1
        rate = live_rate or calib["rate"] or (prior / max(total, 1))
        if steps_left <= 0:
            # FINALIZE running (stage-end probe-trio + export). Remaining = measured
            # median finalize MINUS time already spent in it (anchored on the last
            # stage-ckpt mtime) — counts down honestly instead of a frozen "~15m".
            med = calib["finalize"] or _FINALIZE_PAD_S
            mt = _newest_stage_ckpt_mtime(_arm_of(jid), calib["mtag"])
            in_fin = ((_sod(datetime.now(timezone.utc).strftime("%H:%M:%S"))
                       - _epoch_sod(mt)) % 86400) if mt else 0
            _STAGE_NOTE[jid] = (_STAGE_NOTE.get(jid) or "") + "·final"
            return elapsed + max(med - in_fin, 300)
        return elapsed + steps_left * rate + ends_left * calib["pad"]
    return min(max(prior, elapsed + 300), cap)


def _hf_backup(mtag):
    """Light `upload outputs/<mtag>` in reuse mode, non-interactive (HF_UPLOAD_MODE=reuse so the
    delete/reuse gate never blocks an unattended watch). The remote _full-*.tar shards +
    _full-manifest.json are protected from the light mirror-cleanup. Returns (rc, log_name)."""
    ulog = REPO / "logs" / f"upload_outputs_{mtag}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
    env = dict(os.environ, HF_UPLOAD_MODE="reuse", PYTHONPATH="src")
    with open(ulog, "wb") as fh:
        rc = subprocess.run(
            ["python", "-u", "src/utils/hf_outputs.py", "upload", f"outputs/{mtag}"],
            cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT,
        ).returncode
    return rc, ulog.name


def maybe_backup(fully_done, mtag):
    """Keep outputs/<mtag> backed up on HF every UPLOAD_EVERY_MIN minutes + once more at the finish.
    Returns one plain-English status line to print. POC + FULL are backed up (the runs worth keeping);
    SANITY outputs are throwaway and never backed up."""
    if AUTO_BACKUP_DISABLED:
        return ("  ⏸️  HF auto-backup PAUSED (manual upload in flight — flip "
                "AUTO_BACKUP_DISABLED=False in ngpu_run_status.py to re-enable)")
    if mtag not in ("poc", "full"):
        return "  ⏫ HF backup: skipped (SANITY outputs are throwaway)"
    last = REPO / "logs" / f".upload_outputs_{mtag}.LAST"
    lock = REPO / "logs" / f".upload_outputs_{mtag}.LOCK"
    done_flag = REPO / "logs" / f".upload_outputs_{mtag}.DONE"
    now_ts = datetime.now(timezone.utc).timestamp()
    if done_flag.exists():
        return "  ✅ run finished and fully backed up to HF — you can kill the vast.ai node any time."
    # stale-lock clear: 4× the cadence (not 2×) — the first every-file mirror moves the
    # ~346G resume anchors and can legitimately run ~2-3 h; clearing its lock early would
    # start a CONCURRENT upload of the same tree (HF commit races).
    if lock.exists() and (now_ts - lock.stat().st_mtime) > 4 * UPLOAD_EVERY_MIN * 60:
        lock.unlink(missing_ok=True)
    age_min = (now_ts - last.stat().st_mtime) / 60 if last.exists() else 1e9
    if age_min < UPLOAD_EVERY_MIN and not fully_done:
        return f"  ⏫ HF backup: last {int(age_min)}m ago · next in ~{max(int(UPLOAD_EVERY_MIN - age_min), 0)}m"
    try:
        _fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(_fd)
    except FileExistsError:
        return "  ⏫ HF backup already running — leaving it."
    try:
        last.write_text("")
        rc, name = _hf_backup(mtag)
        if rc != 0:
            return f"  ❌ HF backup FAILED rc={rc} (see {name}) — it will try again next backup."
        if fully_done:
            done_flag.write_text(name)
            return f"  ✅ FINAL HF backup done ({name}) — you can kill the vast.ai node now."
        return f"  ✅ HF backup done ({name}) · next in ~{UPLOAD_EVERY_MIN}m"
    finally:
        lock.unlink(missing_ok=True)


# venv python for every m13 subprocess — the status tool may run under SYSTEM python (no matplotlib/pymupdf),
# so resolve the project venv explicitly, NOT sys.executable. 2026-06-24: sys.executable was the bug that left
# metrics_watch 4.5h stale under `python …status.py` (ModuleNotFoundError: matplotlib). Single source below.
_VENV_PY = REPO / "venv_denseworld" / "bin" / "python"
_VENV_PY = str(_VENV_PY) if _VENV_PY.exists() else sys.executable


def _metrics_watch_cmd(mtag, mode):
    """m13 --metrics-watch-only argv → regenerates BOTH (a) the per-backbone metrics_watch/<bb>/
    (eval_scorecard.pdf/png + kept/paper/tcc/validity + eval_metrics.csv/json) AND (b) the TOP-LEVEL
    cross-backbone figures at outputs/<mtag>/probe_plot/metrics_watch/ (forest_plot_{best,frozen} +
    scale_replication + scale_poc_vs_full_<backbone> + eval_scorecard_combined.{pdf,png}) — the
    scale_poc_vs_full_* forests compare POC 10k vs FULL 116k per backbone (2B FULL panel = N/A until
    trained) — m13 folds cross_backbone_report into the
    --metrics-watch-out path (iter19 2026-07-09), so this ONE call is the single source for both the
    --plots path and the auto-preview. From the eval JSONs finished so far (CPU re-render, no GPU)."""
    cmd = [_VENV_PY, "-u", "src/m13_eval_plot.py", f"--{mode}",
           "--output-dir", f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_plot')}",
           "--outputs-root", _eval_root(mtag, BACKBONE, EVAL_CORPUS),
           "--metrics-watch-out", f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_plot')}/metrics_watch",
           "--metrics-watch-only"]
    # iter19 2026-07-09 (user order): forward the run's dropped arms EXPLICITLY (not just via the inherited
    # ITER18_SKIP_ARMS env) so EVERY watch figure — per-backbone scorecard, the combined scorecard, AND the
    # cross-backbone forest/scale — hides the same SKIP arms. Source = the env the pane set from the run's
    # "[--skip-arms] dropped […]" log line (see the skip-forward block in main).
    _skip = os.environ.get("ITER18_SKIP_ARMS", "").strip()
    if _skip:
        cmd += ["--skip-arms", _skip]
    return cmd


def maybe_plot(mtag, mode):
    """Every PLOT_EVERY_MIN min, rebuild the shared paired-Δ + m13 plots from the evals done SO FAR —
    a live partial hero table to screen-share while the run finishes. Mirrors the scheduler's §3
    finale EXACTLY (same run_eval invocation, same SKIP_STAGES) so there is one recipe, not two.
    SAFE: paired stages always re-aggregate; the final §3 rebuilds the COMPLETE plots at the end.
    With few evals done, run_eval may exit non-zero (missing caches) — reported, harmless."""
    last = REPO / "logs" / ".plot_preview.LAST"
    lock = REPO / "logs" / ".plot_preview.LOCK"
    now_ts = datetime.now(timezone.utc).timestamp()
    hero = REPO / f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_plot')}/eval/m13_hero_table.png"
    if lock.exists() and (now_ts - lock.stat().st_mtime) > 2 * PLOT_EVERY_MIN * 60:
        lock.unlink(missing_ok=True)
    age_min = (now_ts - last.stat().st_mtime) / 60 if last.exists() else 1e9
    if age_min < PLOT_EVERY_MIN:
        return (f"  🖼  preview: rebuilt {int(age_min)}m ago · next in ~{max(int(PLOT_EVERY_MIN - age_min), 0)}m"
                f" → {_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_plot')}/eval/m13_hero_table.png")
    try:
        _fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(_fd)
    except FileExistsError:
        return "  🖼  preview already rebuilding — leaving it."
    try:
        last.write_text("")
        all_encs = " ".join([enc_name("frozen")] + [enc_name(e) for e in ARM2ENC.values()])
        chain = (f"SKIP_STAGES={S3_SKIP_PERENC} CACHE_POLICY_ALL=1 "
                 f"./scripts/run_eval.sh --{mode} --encoders \"{all_encs}\"")
        plog = REPO / "logs" / f"plot_preview_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.log"
        with open(plog, "wb") as fh:
            # 2026-06-24: refresh the metrics_watch set (eval_scorecard.pdf/png + kept/paper/tcc/validity +
            # eval_metrics.csv/json) FIRST + via the venv python — so a plain `ngpu_run_status.py` (system
            # python, no --plots) keeps the live scorecard fresh for whatever eval has finished, and the slow
            # §3 hero-table chain below can never block/stale it. m13 reads JSONs → no GPU.
            fh.write(b"=== metrics_watch refresh (eval_scorecard + eval_metrics.csv) ===\n"); fh.flush()
            mw_rc = subprocess.run(_metrics_watch_cmd(mtag, mode), cwd=str(REPO), env=os.environ.copy(),
                                   stdout=fh, stderr=subprocess.STDOUT).returncode
            # iter19 2026-07-09: the TOP-LEVEL cross-backbone figures (forest_plot_{best,frozen} +
            # scale_replication + eval_scorecard_combined.{pdf,png}) are now emitted INSIDE the
            # metrics_watch refresh above (m13 --metrics-watch-out folds cross_backbone_report), so the
            # separate combined-PDF re-stack step is gone — one recipe, every watch figure tracks live evals.
            fh.write(b"\n=== S3 hero-table preview ===\n"); fh.flush()
            rc = subprocess.run(chain, shell=True, executable="/bin/bash", cwd=str(REPO),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        scard = REPO / f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_plot')}/metrics_watch/{BACKBONE}/eval_scorecard.png"
        if scard.exists() or hero.exists():
            return (f"  🖼  preview REBUILT (metrics_watch rc={mw_rc} · hero rc={rc}) → "
                    f"metrics_watch/{BACKBONE}/{{eval_scorecard,eval_metrics.csv}} + probe_plot/metrics_watch/"
                    f"{{forest_plot_*,scale_replication,scale_poc_vs_full_*,eval_scorecard_combined}} · next in ~{PLOT_EVERY_MIN}m")
        return f"  🖼  preview rc={rc}/{mw_rc} — partial/blocked, see {plog.name} · next in ~{PLOT_EVERY_MIN}m"
    finally:
        lock.unlink(missing_ok=True)


def maybe_metrics_plots(mtag, mode):
    """`--plots`: refresh the WHOLE metrics_watch figure+data set via m13's self-contained
    --metrics-watch refresh (so `ngpu_run_status.py --plots` = one command for status + every figure).
    Reuses THIS process's ITER18_BACKBONE / ITER18_SKIP_ARMS env (m13 reads the same vars: _MW_BACKBONE +
    _mw_skip_arms). Read-only on the eval/train artifacts; safe next to a live run."""
    out = REPO / _eval_root(mtag, BACKBONE, EVAL_CORPUS)
    cmd = _metrics_watch_cmd(mtag, mode)
    print(f"\n  📊 --plots: refreshing metrics_watch figures+data → {out}/probe_plot/metrics_watch/{BACKBONE}/")
    rc = subprocess.run(cmd, cwd=str(REPO), env=os.environ.copy()).returncode
    if rc == 0:
        print(f"  📊 metrics_watch refreshed (rc=0) → {_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_plot')}/metrics_watch/{BACKBONE}/ "
              f"+ outputs/{mtag}/probe_plot/metrics_watch/{{forest_plot_*,scale_replication,scale_poc_vs_full_*,eval_scorecard_combined}}")
    else:
        print(f"  📊 metrics_watch refresh rc={rc} — partial/blocked (partial json under a live run is normal)")


_BURN_RE = re.compile(r"^ngpu_run_(?P<mode>[a-z]+)_(?P<rest>.+)_(?P<ts>\d{8}_\d{6})\.log$")
_BURN_METRICS = ("teacher_free", "maskratio", "rollout", "causal", "tdist", "order",
                 "all", "aot", "tov", "pace", "tcc")


def _logs_burn():
    """Audited GPU burn (iter19 2026-07-09, user order: the research-group compute bill).
    EVERY logs/ngpu_run_*.log is one GPU-job execution; its wall = mtime − filename start-stamp
    (validated: tov 162457 → 5h50m45s vs its own pbar 5:50:15). Summing ALL segments prices the
    TRUE burn — aborted attempts, retries, redone evals, sanity smokes — unlike the projection
    table (current roster, done+estimated). Buckets: arm → {train,eval} (frozen under 'frozen';
    eval col = E:+P:+F: wrappers); 'labels+merges' pooled; every non-FULL mode pooled under
    'sanity'. The m09a seed trained on Box A; 07-08 its wrappers were copied into logs/ (from
    iter/iter19_train_115kclips/logs/seed_training) with mtimes RESTORED from log content (the copy
    had stamped all files 08:25:35) — so the seed's 18h05m train wall is now priced like the rest."""
    enc2arm = {enc_name(e): a for a, e in ARM2ENC.items()}
    enc2arm[enc_name("frozen")] = "frozen"
    burn = {}

    def _add(bucket, col, secs):
        burn.setdefault(bucket, {"train": 0.0, "eval": 0.0})[col] += secs

    for p in (REPO / "logs").glob("ngpu_run_*.log"):
        m = _BURN_RE.match(p.name)
        if not m:
            continue
        try:
            start = datetime.strptime(m["ts"], "%Y%m%d_%H%M%S").timestamp()
            secs = max(0.0, p.stat().st_mtime - start)   # still-running logs → burn-to-now
        except (ValueError, OSError):
            continue
        mode, rest = m["mode"], m["rest"]
        if mode != "full":
            _add("sanity", "train" if rest.startswith("train_") else "eval", secs)
        elif rest.startswith("train_"):
            _add(rest[6:], "train", secs)
        elif rest.startswith("eval_"):
            _add(enc2arm.get(rest[5:], rest[5:]), "eval", secs)
        elif rest.startswith(("pt_", "et_")):
            body = rest.split("_", 1)[1]
            for met in _BURN_METRICS:
                if body.endswith("_" + met):
                    body = body[: -len(met) - 1]
                    break
            _add(enc2arm.get(body, body), "eval", secs)
        elif rest == "labels" or rest.startswith("merge_"):
            _add("labels+merges", "eval", secs)
        else:
            _add("other", "eval", secs)
    return burn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["POC", "SANITY", "FULL"], default="POC")
    ap.add_argument("--log", default=None,
                    help="explicit main log (default: latest logs/ngpu_run_<mode>*.log "
                         "excluding per-job _train_/_eval_/_s3_ logs)")
    ap.add_argument("--plots", action="store_true",
                    help="after the ASCII status, refresh ALL metrics_watch figures (3 base + WiSE-FT sweep "
                         "+ paper scorecard + TCC chart) + {train,eval}_metrics.{json,csv} via m13's "
                         "self-contained --metrics-watch refresh (one command = status + every figure).")
    args = ap.parse_args()

    jobs, mtag = build_jobs(args.mode)
    log = Path(args.log) if args.log else _latest_log(mtag)
    if log is None or not log.exists():
        sys.exit(f"No {args.mode} main log found (logs/ngpu_run_{mtag}*.log). "
                 f"Has the scheduler started?")

    text = log.read_text(errors="replace")

    # regen runs (--taxheads-only / --etheads-only) emit per-encoder X:/Y: jobs — a DIFFERENT job set
    # than the full DAG. Detect from the launch markers + rebuild jobs to match, else every X:/Y: marker
    # is filtered out below and the table reads all-pending on stale priors (iter18 2026-06-21).
    _marker_ids = re.findall(r"GPU\d+ ◀ (\S+)", text)
    taxheads_only = any(j.startswith("X:") for j in _marker_ids)
    etheads_only = any(j.startswith("Y:") for j in _marker_ids)
    regen = taxheads_only or etheads_only
    if regen:
        jobs, mtag = build_jobs(args.mode, taxheads_only=taxheads_only, etheads_only=etheads_only)

    # iter18 2026-06-08: the run's backbone is in the banner (backbone=…). If THIS watch pane was
    # launched without the matching ITER18_BACKBONE, our imported BACKBONE (+ all job-ids/paths)
    # are for the wrong family → every cell would read pending. Warn LOUDLY rather than mislead.
    _bm = re.search(r"backbone=(\S+)", text)
    if _bm and _bm.group(1) != BACKBONE:
        print(f"  ⚠️  BACKBONE MISMATCH — run is '{_bm.group(1)}' but this watch pane is '{BACKBONE}'. "
              f"Re-run:  ITER18_BACKBONE={_bm.group(1)} python -u scripts/ngpu_run_status.py", flush=True)
    # Same guard for the scored corpus (banner prints corpus=…). Both the run and this pane derive it from
    # pipeline.yaml data.local_data_dir, so they agree unless one exported EVAL_CORPUS/TRAINED_CORPUS — in
    # which case our eval/<corpus>/ paths would read the wrong tree. Warn rather than silently mislead.
    # (Old pre-corpus-banner logs have no corpus= → the check no-ops, like the backbone one.)
    _cm = re.search(r"corpus=(\S+)", text)
    if _cm and _cm.group(1) != EVAL_CORPUS:
        print(f"  ⚠️  CORPUS MISMATCH — run scored '{_cm.group(1)}' but this watch pane resolves '{EVAL_CORPUS}'. "
              f"Re-run with  EVAL_CORPUS={_cm.group(1)}  exported.", flush=True)

    # An --only run restricts the DAG — mirror that restriction so this table matches reality.
    om = re.search(r"\[--only\] restricted to \[(.*?)\]", text)
    only_arms = set(re.findall(r"'([^']+)'", om.group(1))) if om else None
    if only_arms:
        keep = {f"T:{BACKBONE}:{a}" for a in only_arms}
        jobs = {jid: j for jid, j in jobs.items() if jid in keep}

    # --skip-arms runs drop arms (train+eval) from the DAG — mirror it (iter18 2026-06-07).
    sm = re.search(r"\[--skip-arms\] dropped \[(.*?)\]", text)
    if sm:
        skip = set(re.findall(r"'([^']+)'", sm.group(1)))
        # 2026-06-24: forward the run's dropped arms to the metrics_watch regen so the AUTO-rendered
        # eval_scorecard hides them too. m13 (maybe_plot / --plots) reads ITER18_SKIP_ARMS via
        # _mw_skip_arms() (whitespace-split arm names). The watch command sets ITER18_BACKBONE but NOT
        # ITER18_SKIP_ARMS, so without this the 8 --skip-arms showed up as empty N/A bars. Single source
        # = the log (what the run actually skipped); setdefault honors an explicit operator override
        # (the runbook --plots path that sets ITER18_SKIP_ARMS="$SKIP" itself).
        os.environ.setdefault("ITER18_SKIP_ARMS", " ".join(sorted(skip)))
        drop = ({f"T:{BACKBONE}:{a}" for a in skip}
                | {f"E:{enc_name(ARM2ENC[a])}" for a in skip if a in ARM2ENC}
                | {f"P:{enc_name(ARM2ENC[a])}:all" for a in skip if a in ARM2ENC}   # iter19: 8b is one --metric all job
                | {f"F:{enc_name(ARM2ENC[a])}:{m}" for a in skip if a in ARM2ENC for m in ET_METRICS}
                | {f"X:{enc_name(ARM2ENC[a])}" for a in skip if a in ARM2ENC}   # --taxheads-only regen
                | {f"Y:{enc_name(ARM2ENC[a])}" for a in skip if a in ARM2ENC})  # --etheads-only regen
        jobs = {jid: j for jid, j in jobs.items() if jid not in drop}

    launched = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ◀ (\S+)", text)}
    done = {m.group(2): m.group(1) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✓ (\S+)", text)}
    # The scheduler's one-shot RETRY (2026-06-24) prints a non-terminal "✗ <jid> … RETRY 1/1" on the FIRST
    # failure before re-queueing; only a SECOND failure ("retry also failed") is terminal. Exclude the RETRY
    # marker so a job mid-retry classifies as running (re-launched ◀) → done/failed by its OUTCOME, not ❌.
    failed = {m.group(2): m.group(1)
              for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] GPU\d+ ✗ (\S+)([^\n]*)", text)
              if "RETRY" not in m.group(3)}
    rm = re.search(r"\[resume --cache 1\] skipping \d+ already-trained arms: \[(.*?)\]", text)
    if rm:
        # iter18 prints BARE arm names (iter17 printed bb:arm) → rebuild the full jid.
        for tok in re.findall(r"'([^']+)'", rm.group(1)):
            done.setdefault(f"T:{BACKBONE}:{tok}", "resume")
    # P: (Stage-8b single-metric) jobs are 'done' when their aggregate_<metric>.json exists on disk.
    # The scheduler resume-skips them silently (no per-job GPU ✓ marker logged), so mirror that from
    # disk — and it also surfaces 8b metrics produced by a prior monolithic eval as already-done.
    for jid in jobs:
        if jid.startswith("P:"):
            enc_nm, metric = jid[2:].rsplit(":", 1)   # NOT 'enc_name' — that's the imported helper (shadowing it makes it a main()-local → UnboundLocalError)
            if (REPO / f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'predictor_temporal')}/{enc_nm}/aggregate_{metric}.json").exists():
                done.setdefault(jid, "resume")
        elif jid.startswith("F:"):    # Stage-8c (m12f encoder_temporal) — same done-marker pattern
            enc_nm, metric = jid[2:].rsplit(":", 1)
            if (REPO / f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'encoder_temporal')}/{enc_nm}/aggregate_{metric}.json").exists():
                done.setdefault(jid, "resume")
    # iter18 2026-06-08: drop any log-parsed jid NOT in THIS run's job set. A watch pane reading a
    # DIFFERENT-backbone or OLD (pre-banner) log carries foreign jids (e.g. T:vjepa_2_1_vitG:… while
    # this pane built vjepa_2_1_vitg jobs); feeding one into jobs[jid] downstream → KeyError. The
    # backbone-mismatch warning above flags the case when the banner records backbone=.
    launched = {j: t for j, t in launched.items() if j in jobs}
    done = {j: t for j, t in done.items() if j in jobs}
    failed = {j: t for j, t in failed.items() if j in jobs}
    gm = re.search(r"gpus=(\d+)", text)
    gpus = int(gm.group(1)) if gm else 4
    s3 = re.search(r"§3 rc=(-?\d+)", text)
    only_complete = "--only run complete" in text
    now_hms = datetime.now(timezone.utc).strftime("%H:%M:%S")
    now_s = _sod(now_hms)

    _hsd_cache = {}

    def _e_skip_done(enc):
        # a resume-SKIPPED E: emits no ◀/✓ (classify's log markers) yet IS done when its 4 head-metric
        # outputs exist — the same disk-truth the scheduler's skip / the 2-8 fan cell / the E0 gate line
        # use (head_stage_done). Overlaying it HERE makes the arm-table group cell, the Σ totals and every
        # other classify() consumer agree at once — 07-08 frozen read '⬚ ~9h10m' after ~59h of real burn
        # because its E: was skip-resumed with no log marker. Memoized: classify runs 100s×/render.
        if enc not in _hsd_cache:
            _hsd_cache[enc] = head_stage_done(mtag, BACKBONE, EVAL_CORPUS, enc)
        return _hsd_cache[enc]

    def classify(jid):
        if jid in done:
            return "done"
        if jid in failed:
            return "failed"
        if jid in launched:
            return "running"
        if jid.startswith("E:") and _e_skip_done(jid[2:]):
            return "done"
        return "pending"

    def elapsed(jid):
        return (now_s - _sod(launched[jid])) % 86400 if jid in launched else 0

    # ── REAL-ETA inputs (iter18 2026-06-07): static workload ledger + calibration ──
    # The step-ledger prices only RUNNING/PENDING train arms (reads train_pool.json + factor_manifest).
    # An eval-only run — the cross-set retest (all train resume-skipped) or a --taxheads/--etheads regen
    # (0 train jobs) — needs no ledger and has no train_pool.json on the box, so skip it (building it would
    # just FileNotFoundError → priors), iter18 2026-06-21.
    if any(j.startswith("T:") and classify(j) not in ("done", "failed") for j in jobs):
        try:
            ledger = _build_ledger(mtag)
        except Exception as e:   # the watch must survive a ledger failure — fall back LOUDLY
            print(f"  ⚠️  workload ledger FAILED ({type(e).__name__}: {e}) — ETA on priors", flush=True)
            ledger = None
    else:
        ledger = None
    calib = _calibrate(jobs, done, launched, mtag, ledger)
    # eval REAL-ETA: per-stage medians + running-eval stage state + per-job stage plans.
    ecalib = _eval_calibrate(jobs, mtag, now_s)

    # evals stay duration-class-based (homogeneous); their measured mean replaces the prior.
    est = dict({"poc": PRIOR, "full": PRIOR_FULL}.get(mtag, PRIOR_SANITY))
    measured = {}
    for jid, t in done.items():
        if t != "resume" and jid in launched:
            measured.setdefault(_arm_of(jid), []).append((_sod(t) - _sod(launched[jid])) % 86400)
    for arm, vals in measured.items():
        est[arm] = sum(vals) / len(vals)

    # Σ secs per jid across all log segments — reused for the P: per-metric medians here and the
    # table's consumed/✅ cells + Σ TOTAL row below.
    consumed = _arm_consumed(jobs, mtag)
    # regen (Y:/X:) jobs are multi-phase per-encoder run_evals. Price the PENDING ones from the MEASURED
    # per-job time — the median over completed peers' walls + the live total-wall the RUNNING peers project
    # from their own m12f phase bars (_regen_running_total). No hardcoded warm-cache factor: as later waves
    # run on the warm frame-cache, their live projections come in shorter, so this median self-corrects DOWN
    # on its own (iter18 2026-06-21).
    _regen_kind = "taxheads" if taxheads_only else "etheads"
    _regen_cold = _REGEN_COLD_PRIOR[_regen_kind].get(mtag, 90 * 60)
    _regen_perjob = sorted(
        [consumed[j] for j in jobs
         if j.startswith(("Y:", "X:")) and classify(j) == "done" and j in consumed]
        + [_regen_running_total(jobs, j, elapsed(j), _regen_cold) for j in jobs
           if j.startswith(("Y:", "X:")) and classify(j) == "running"])
    _regen_perjob_med = _regen_perjob[len(_regen_perjob) // 2] if _regen_perjob else None
    # P: (Stage-8b single-metric) per-metric medians from COMPLETED P: jobs → priors for pending P:
    # (metrics differ a lot in cost — teacher_free ≫ causal — so keep them per-metric).
    _pt_by = {}
    for jid in jobs:
        if jid.startswith(("P:", "F:")) and classify(jid) == "done" and jid in consumed:
            # one shared per-metric dict: PT names (rollout/causal/…) and ET names (aot/tov/…) are disjoint
            _pt_by.setdefault(jid.rsplit(":", 1)[-1], []).append(consumed[jid])
    pt_med = {m: sorted(v)[len(v) // 2] for m, v in _pt_by.items()}
    # LIVE projection seeding (iter18 2026-06-12): a RUNNING P:/F: job's own clip bar projects
    # its full wall (elapsed + remaining×rate) — PENDING siblings of the SAME metric inherit
    # the median of those projections until a real completion lands in pt_med. Without this,
    # pending F: jobs sat on the cold prior while their running siblings were measurably 3×
    # slower → the bottom-line run ETA read ~3h for ~8h of queued work.
    _proj = {}
    for jid in jobs:
        if not jid.startswith(("P:", "F:")) or classify(jid) != "running":
            continue
        cands = sorted((p for p in REPO.glob(jobs[jid]["log"].format(ts="*")) if p.exists()),
                       key=lambda p: p.stat().st_mtime)
        txt_ = _tail(cands[-1]) if cands else ""
        cp = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", txt_)
        rr = re.findall(r"recent=([\d.]+)s/clip", txt_)
        if cp and rr and int(cp[-1][1]) and int(cp[-1][0]) >= _MIN_EVAL_POINTS:
            cur, tot, rate = int(cp[-1][0]), int(cp[-1][1]), float(rr[-1])
            _proj.setdefault(jid.rsplit(":", 1)[-1], []).append(elapsed(jid) + (tot - cur) * rate)
    for _m, _v in _proj.items():
        pt_med.setdefault(_m, sorted(_v)[len(_v) // 2])

    # ── per-job remaining time ──
    remaining = {}
    for jid in jobs:
        st = classify(jid)
        arm = _arm_of(jid)
        prior = est.get(arm, est["eval"])
        if arm in MERGE_ARMS:                  # WiSE-FT: post-hoc weight merge, ~minutes, no step plan
            prior = _MERGE_PRIOR.get(mtag, 3 * 60)
        if st in ("done", "failed"):
            remaining[jid] = 0.0
        elif jid.startswith(("Y:", "X:")):
            # regen job. RUNNING → live multi-phase projection from its OWN m12f bars (elapsed / progress).
            # PENDING → the measured per-job median (completed walls + running projections), which already
            # reflects the warm-cache speedup once warm waves are live — no hardcoded factor.
            if st == "running":
                tot = _regen_running_total(jobs, jid, elapsed(jid), _regen_cold)
                remaining[jid] = max(tot - elapsed(jid), 60)
            else:
                remaining[jid] = _regen_perjob_med or _regen_cold
        elif jid.startswith(("P:", "F:")):
            # Stage-8b/8c single-metric job — estimated from its OWN clip bar (running) or the
            # per-metric median (pending), NOT the stage ledger. Cold prior is per-FAMILY
            # (_ET_COLD_PRIOR for the heavier F: 8c jobs), NOT est["eval"] (whole-eval, ~10× off).
            el = elapsed(jid) if st == "running" else 0.0
            cold = (_ET_COLD_PRIOR if jid.startswith("F:") else _PT_COLD_PRIOR).get(mtag, 20 * 60)
            tot = _pt_total(jobs, jid, el, cold, pt_med)
            remaining[jid] = max(tot - el, 60) if st == "running" else tot
        elif st == "running":
            remaining[jid] = max(_running_total(jobs, jid, elapsed(jid), prior, calib, ecalib)
                                 - elapsed(jid), 60)
        elif arm == "eval":
            # PENDING E: eval = stages 2-8 plan × measured/projected stage walls (Stage 8b is the
            # separate P: jobs above) — its OWN honest total, not a class prior.
            remaining[jid] = _eval_total(jid, 0.0, prior, ecalib)
        elif jid.startswith("T:") and ledger and arm in ledger and calib["rate"] is not None:
            # PENDING train = full ledger plan × measured pure rate + every pad —
            # the REAL ETA (known work ÷ measured throughput), not a hopeful prior.
            # `arm in ledger` guards the kind=merge job (wiseft): it has NO ledger entry, so without
            # this it would KeyError the whole watch the moment calib["rate"] goes non-None (the first
            # train arm completes). The merge instead falls through to `prior` (the small _MERGE_PRIOR).
            led = ledger[arm]
            remaining[jid] = (led["total"] * calib["rate"]
                              + (len(led["stages"]) + 2) * calib["pad"])
        else:
            remaining[jid] = prior

    # ── forward DAG sim over the GPU pool → finish time (secs from now) per job ──
    done_set = {j for j in jobs if classify(j) in ("done", "failed")}
    running = {j for j in jobs if classify(j) == "running"}
    pending = {j for j in jobs if classify(j) == "pending"}
    finish = {j: 0.0 for j in done_set}
    run_end = {j: remaining[j] for j in running}
    free = max(gpus - len(running), 0)
    t = 0.0
    guard = 0
    while (running or pending) and guard < _SIM_GUARD:
        guard += 1
        for j in [x for x in jobs if x in pending and jobs[x]["deps"] <= done_set]:
            if free <= 0:
                break
            pending.discard(j)
            running.add(j)
            run_end[j] = t + remaining[j]
            free -= 1
        if not running:
            break
        nxt = min(running, key=lambda j: run_end[j])
        t = run_end[nxt]
        running.discard(nxt)
        done_set.add(nxt)
        finish[nxt] = t
        free += 1
    # §3 finale: MEASURED from the 15-min preview runs (maybe_plot — the EXACT same
    # run_eval recipe/stages). A preview lasting >120s had real eval caches to aggregate,
    # so its wall ≈ the finale's; the 7s previews (caches missing, early exit) are
    # excluded. Until a substantive preview exists, the documented prior is used and
    # the printout SAYS so — no unlabeled fake numbers (user order, 06-07).
    pv = []
    for p in (REPO / "logs").glob("plot_preview_2*.log"):
        if not p.exists():        # skip dangling symlinks
            continue
        m2 = re.search(r"_(\d{8}_\d{6})\.log$", p.name)
        if m2:
            st = datetime.strptime(m2.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            d = p.stat().st_mtime - st.timestamp()
            if d > 120:
                pv.append(d)
    pv.sort()
    if pv:
        s3_pad, s3_src = pv[len(pv) // 2], "measured"
    else:
        s3_pad, s3_src = {"poc": 15, "full": 25}.get(mtag, 4) * 60, "prior, unmeasured"
    eta_secs = (max(finish.values()) if finish else 0) + s3_pad

    # ── render: emoji table (rows = arm, cols = train | eval) ──
    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0}
    for jid in jobs:
        counts[classify(jid)] += 1

    # ── regen runs (--taxheads-only X: / --etheads-only Y:) get a dedicated compact table: one row per
    #    encoder (state · ET-metric heads saved · ETA). The train|eval grid + 8b/8c fan below don't apply
    #    (a regen DAG has no T:/E:/P:/F: jobs), so render this and return (iter18 2026-06-21). ──
    if regen:
        kind_label = ("et-heads · Stage 8c (aot·tov·pace·tcc)" if etheads_only else "tax-heads · Stage 11")
        prefix = "Y" if etheads_only else "X"
        order = [enc_name(e) for _a, e in [("frozen", "frozen")] + list(ARM2ENC.items())]
        SW2, MW2, CW2 = 40, 12, 24
        print(f"═══ {args.mode} REGEN: {kind_label} · {log.name} · now {now_hms} UTC · {gpus} GPU ═══")
        print("┌" + "─" * SW2 + "┬" + "─" * MW2 + "┬" + "─" * CW2 + "┐")
        print("│" + " encoder".ljust(SW2) + "│" + (" heads" if etheads_only else " stage").ljust(MW2)
              + "│" + " state".ljust(CW2) + "│")
        print("├" + "─" * SW2 + "┼" + "─" * MW2 + "┼" + "─" * CW2 + "┤")
        for enc in order:
            jid = f"{prefix}:{enc}"
            if jid not in jobs:
                continue
            st = classify(jid)
            if etheads_only:
                # the regen's PRODUCT is the reuse HEADS (head_<metric>.pt for aot/tov/pace; tcc is
                # training-free → no head). Count those — NOT aggregate_*.json, which may pre-exist from
                # the original eval and would show a stale "done" on a not-yet-run encoder.
                _etd = REPO / f"{_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'encoder_temporal')}/{enc}"
                _hm = ("aot", "tov", "pace")
                nd = sum(1 for m in _hm if (_etd / f"head_{m}.pt").exists())
                mcell = f"{nd}✓/{len(_hm)}"
            else:
                mcell = "—"
            if st == "done":
                scell = f"✅ {_dur(consumed.get(jid, 0))}"
            elif st == "failed":
                scell = "❌ FAILED"
            elif st == "running":
                scell = f"🔄 {_dur(elapsed(jid))}·~{_dur(finish.get(jid, remaining[jid]))}"
            else:
                scell = f"⬚ ~{_dur(finish.get(jid, remaining[jid]))}"
            short = enc.replace(f"{enc_prefix()}_", "")
            print("│ " + short.ljust(SW2 - 1) + "│ " + mcell.ljust(MW2 - 1) + "│ " + scell.ljust(CW2 - 1) + "│")
        print("└" + "─" * SW2 + "┴" + "─" * MW2 + "┴" + "─" * CW2 + "┘")
        regen_eta = max(finish.values()) if finish else 0.0    # no §3 finale in a regen run
        print(f"\n  {counts['done']}✅  {counts['running']}🔄  {counts['pending']}⬚  "
              f"{counts['failed']}❌  / {len(jobs)} {prefix}: jobs")
        settled = counts["done"] + counts["failed"] == len(jobs)
        if settled and not counts["failed"]:
            print(f"  🏁 all {len(jobs)} regen jobs DONE — heads saved on disk. "
                  f"Next: the 200-clip smoke, then the full retest.")
        elif settled and counts["failed"]:
            flag = "--etheads-only" if etheads_only else "--taxheads-only"
            print(f"  ⚠️  {counts['failed']}❌ FAILED — recover: python -u scripts/ngpu_run.py "
                  f"--mode {args.mode} --gpus {gpus} {flag} --cache 1 --skip-arms $SKIP")
            print("  ❌ failed: " + ", ".join(j for j in jobs if classify(j) == "failed"))
        else:
            _end = datetime.now(timezone.utc) + timedelta(seconds=regen_eta)
            print(f"  🏁 regen ETA  ~{_dur(regen_eta)} from now  →  {_end:%H:%M} UTC · "
                  f"{(_end - timedelta(hours=7)):%H:%M} PDT")
        print("\n  legend: ✅ done · 🔄 running · ⬚ pending · ❌ failed"
              + ("  ·  heads ✓ = reuse head saved (head_{aot,tov,pace}.pt; tcc is training-free)"
                 if etheads_only else ""))
        return

    # `consumed` (Σ GPU-seconds per jid across ALL log segments) was computed above for pt_med.

    def cell(jid):
        if jid not in jobs:
            return "—"
        st = classify(jid)
        if st == "done":
            # TOTAL duration consumed across ALL the arm's log segments (every launch,
            # incl. interrupted ones) — one number, no per-run breakdown (user 06-07).
            tot = consumed.get(jid, 0)
            if tot:
                return f"✅ {_dur(tot)}"
            if done.get(jid) == "resume":
                _pd = _prior_run_dur(jid.split(":")[-1])   # 2.1: real span from the seed logs, not '(prior run)'
                return f"✅ {_dur(_pd)} (prior)" if _pd else "✅ (prior run)"
            return f"✅ {_dur((_sod(done[jid]) - _sod(launched[jid])) % 86400)}"
        if st == "failed":
            return "❌ FAILED"
        if st == "running":
            note = _STAGE_NOTE.get(jid)
            stg = f"·{note}" if note else ""
            return f"🔄 {_dur(elapsed(jid))}{stg}·~{_dur(finish[jid])}"
        return f"⬚ ~{_dur(finish[jid])}"

    def _eval_group_cell(enc):   # param is 'enc' (a full encoder name) — NOT 'enc_name' (shadowing the imported helper is the bug that bit twice)
        # iter18 2026-06-07: an encoder's eval = ONE E: job (stages 2-8) + 6 P: jobs (Stage 8b, one
        # metric each). They run in PARALLEL across the GPU pool, so the group's remaining is the
        # MAX finish, not the sum. `·8b D✓R▶/6` = D done, R running of the 6 metric jobs — so the
        # one rolled-up cell reconciles with the 🔄 job counter (3 parallel metrics here = 3▶).
        group = [j for j in ([f"E:{enc}", f"P:{enc}:all"]   # iter19: 8b = ONE --metric all job (shared encode)
                             + [f"F:{enc}:{m}" for m in ET_METRICS]) if j in jobs]
        if not group:
            return "—"
        sts = [classify(j) for j in group]
        if all(s == "done" for s in sts):
            return f"✅ {_dur(sum(consumed.get(j, 0) for j in group))}"
        if any(s == "failed" for s in sts):
            return "❌ FAILED"
        # per-CELL OWN duration (user 2026-07-07): [completed · ~estimated] for THIS encoder's eval
        # ONLY — NOT the DAG-cumulative queue finish (finish[] counts every encoder ahead in the pool,
        # so a pending cell reads ~34h when its OWN eval is ~13h). The queue ETA lives in the Σ TOTAL
        # row. Basis = each job's standalone consumed/remaining, so the cells SUM to Σ TOTAL (eval).
        completed = (sum(consumed.get(j, 0) for j in group if classify(j) == "done")
                     + sum(elapsed(j) for j in group if classify(j) == "running"))
        rem = sum(remaining.get(j, 0.0) for j in group if classify(j) in ("running", "pending"))
        if any(s == "running" for s in sts):
            return f"🔄 {_dur(completed)}·~{_dur(rem)}"
        return f"⬚ ~{_dur(rem)}"

    def kemoji(arm):
        if arm in MERGE_ARMS:
            return "🔀"   # post-hoc weight merge (WiSE-FT) — not a training run
        if arm.startswith("pretrain"):
            return "🚂"
        if arm.startswith(("surgery", "surgical")):
            return "🔧"
        return "🔩"   # FT-technique baselines (full_ft/lpft/peft/cassle/ewc)

    SW, CW = 32, 22
    bar = "─"
    print(f"═══ {args.mode} · {log.name} · now {now_hms} UTC · {gpus} GPU ═══")
    # iter19 (2026-07-04): draw a row only if it has a job in THIS invocation's (already --only/--skip-
    # filtered) `jobs` — so a pretrain-only / train-phase run no longer paints the frozen anchor row + an
    # empty 8b+8c fan for evals it isn't running. frozen shows iff it has an E:/P:/F: job here.
    _eval_encs = set()
    for _j in jobs:
        if _j.startswith("E:"):
            _eval_encs.add(_j[2:])
        elif _j.startswith(("P:", "F:")):
            _eval_encs.add(_j[2:].rsplit(":", 1)[0])
    _show_frozen = enc_name("frozen") in _eval_encs
    print("┌" + bar * SW + "┬" + bar * CW + "┬" + bar * CW + "┐")
    print("│" + " arm".ljust(SW) + "│" + " train".ljust(CW) + "│" + " eval".ljust(CW) + "│")
    print("├" + bar * SW + "┼" + bar * CW + "┼" + bar * CW + "┤")
    if _show_frozen:
        print("│ " + "📊 frozen (eval-only)".ljust(SW - 1) + "│" + " —".ljust(CW) + "│ "
              + _eval_group_cell(enc_name("frozen")).ljust(CW - 1) + "│")
    # iter19 2026-07-09 (user order): the SEED row — pretrain_encoder is skip-arms'd out of this
    # run's jobs, but the research-group bill needs it visible. eval = audited wall of its prior-run
    # E:/P:/F: wrapper logs on THIS box (_logs_burn); train ran on Box A whose logs never came here
    # → shown unpriced ("Box A, unlogged"), never an invented number.
    _burn = _logs_burn()
    _pt = "pretrain_encoder"
    _pt_extra = {"train": 0.0, "eval": 0.0}   # the seed row's audited walls — folded into Σ TOTAL below
    if (_pt in ARM2ENC and f"T:{BACKBONE}:{_pt}" not in jobs
            and f"E:{enc_name(ARM2ENC[_pt])}" not in jobs and _pt in _burn):
        _ptb = _pt_extra = _burn[_pt]
        _tr = f"✅ {_dur(_ptb['train'])} (audit)" if _ptb["train"] else "✅ (Box A, unlogged)"
        # eval verdict from disk, not from the log's exit code: the seed's E: was killed mid-run when it
        # was SKIP'd post-gate — its 4 head-metric outputs never landed, so head_stage_done=False → the
        # burn is real but the eval is ABORTED/incomplete, and the group table must say so (user 07-08).
        _pt_ok = head_stage_done(mtag, BACKBONE, EVAL_CORPUS, enc_name(ARM2ENC[_pt]))
        _ev = ((f"✅ {_dur(_ptb['eval'])} (audit)" if _pt_ok else f"❌ {_dur(_ptb['eval'])} (aborted)")
               if _ptb["eval"] else " —")
        print("│ " + f"🚂 {_pt} (seed)".ljust(SW - 1) + "│ "
              + _tr.ljust(CW - 1) + "│ " + _ev.ljust(CW - 1) + "│")
    for arm in DISPLAY_ORDER:        # incl. the wiseft merge — its row is built like any other
        tj, ej = f"T:{BACKBONE}:{arm}", f"E:{enc_name(ARM2ENC[arm])}"
        if tj not in jobs and ej not in jobs:
            continue
        print("│ " + f"{kemoji(arm)} {arm}".ljust(SW - 1) + "│ "
              + cell(tj).ljust(CW - 1) + "│ " + _eval_group_cell(enc_name(ARM2ENC[arm])).ljust(CW - 1) + "│")

    # Σ TOTAL row (iter18 2026-06-07): per column, completed compute (consumed across
    # every log segment) + estimated remaining for running/pending jobs — the full
    # GPU-time bill of the table, done + still-to-come.
    def _col_total(kind):
        tot = 0.0
        for jid in jobs:
            if (_arm_of(jid) == "eval") != (kind == "eval"):
                continue
            st = classify(jid)
            if st in ("done", "failed"):
                tot += consumed.get(jid, 0) or (
                    (_sod(done[jid]) - _sod(launched[jid])) % 86400
                    if jid in done and done[jid] != "resume" and jid in launched else 0)
            elif st == "running":
                tot += consumed.get(jid, elapsed(jid)) + remaining[jid]
            else:
                tot += remaining[jid]
        return tot

    t_tot, e_tot = _col_total("train"), _col_total("eval")
    # iter19 2026-07-08 (user task2/3): fold the seed row into the Σ bill — _col_total prices only this
    # run's `jobs`, so the skip-arms'd seed was missing: Σ train read 59h59m when the true trained-hours
    # are ≈78h (18h05m seed + 30h11m peft_lora + 29h47m diheavy). Its aborted 10h21m eval burn counts
    # too (real GPU spend). Gated on the seed row actually rendering (_pt_extra stays 0 otherwise), so a
    # run that trains the seed itself (T: job in `jobs`) is never double-counted.
    t_tot += _pt_extra["train"]
    e_tot += _pt_extra["eval"]
    print("├" + bar * SW + "┼" + bar * CW + "┼" + bar * CW + "┤")
    print("│ " + "Σ TOTAL (done + estimated)".ljust(SW - 1) + "│ "
          + f"Σ {_dur(t_tot)}".ljust(CW - 1) + "│ " + f"Σ {_dur(e_tot)}".ljust(CW - 1) + "│")
    print("└" + bar * SW + "┴" + bar * CW + "┴" + bar * CW + "┘")
    print(f"  Σ compute bill (train+eval, done+estimated): ~{_dur(t_tot + e_tot)} GPU-time")

    # ── 💰 audited GPU burn — the ACTUAL bill from logs/ (all runs: aborted, retried, redone,
    # sanity), vs the projection table above. iter19 2026-07-09, user order (research-group bill).
    _blabel = {"frozen": "📊 frozen", "labels+merges": "🏷 labels+merges",
               "sanity": "🧪 sanity smokes (all)", "other": "❓ other"}
    _border = ([_pt] if _pt in _burn else []) \
        + [a for a in DISPLAY_ORDER if a in _burn and a != _pt] \
        + [k for k in ("frozen", "labels+merges", "sanity", "other") if k in _burn]
    print("\n  💰 audited GPU burn from logs/ (every run segment on this box, incl. aborted/redone)")
    print("  ┌" + bar * SW + "┬" + bar * CW + "┬" + bar * CW + "┐")
    print("  │" + " burn bucket".ljust(SW) + "│" + " train (actual)".ljust(CW) + "│" + " eval (actual)".ljust(CW) + "│")
    print("  ├" + bar * SW + "┼" + bar * CW + "┼" + bar * CW + "┤")
    _bt = _be = 0.0
    for k in _border:
        v = _burn[k]
        _bt, _be = _bt + v["train"], _be + v["eval"]
        lbl = _blabel.get(k, f"{kemoji(k)} {k}" + (" (seed)" if k == _pt else ""))
        tr = f"{_dur(v['train'])}" if v["train"] else ("(Box A, unlogged)" if k == _pt else "—")
        print("  │ " + lbl.ljust(SW - 1) + "│ " + tr.ljust(CW - 1) + "│ "
              + (_dur(v["eval"]) if v["eval"] else "—").ljust(CW - 1) + "│")
    print("  ├" + bar * SW + "┼" + bar * CW + "┼" + bar * CW + "┤")
    print("  │ " + "Σ audited so far".ljust(SW - 1) + "│ " + f"Σ {_dur(_bt)}".ljust(CW - 1)
          + "│ " + f"Σ {_dur(_be)}".ljust(CW - 1) + "│")
    print("  └" + bar * SW + "┴" + bar * CW + "┴" + bar * CW + "┘")
    _rem_tot = sum(remaining.get(j, 0.0) for j in jobs if classify(j) in ("running", "pending"))
    # the Box-A footnote self-retires: 07-08 the seed's Box-A wrappers were copied into logs/ (from
    # iter/iter19_train_115kclips/logs/seed_training, mtimes restored from log content) → counted.
    _note = "" if _burn.get(_pt, {}).get("train") else "  (excl. Box-A seed train — unlogged here)"
    print(f"  💰 audited so far ≈ {_dur(_bt + _be)} GPU-time · + est. remaining {_dur(_rem_tot)}"
          f" → grand ≈ {_dur(_bt + _be + _rem_tot)}{_note}")

    # ── Stage-8b metric fan: encoder × 6 predictor-temporal metrics, LIVE grid (iter18 2026-06-07) ──
    # Makes the metric-parallel fan-out visible — each cell = the P:<enc>:<metric> job's state.
    # "2-8" = the encoder's E: job (the non-8b eval stages: features/probe/taxonomy/motion/future).
    _GLYPH = {"done": "✓", "running": "🔄", "pending": "·", "failed": "✗"}   # 🔄 not ▶ (user order 06-12: the running marker must be the same emoji everywhere)
    _ABBR = {"rollout": "roll", "causal": "caus", "tdist": "tdis",
             "teacher_free": "t-fr", "maskratio": "mask", "order": "ordr",
             "aot": "aot", "tov": "tov", "pace": "pace", "tcc": "tcc", "all": "8b·6"}
    # iter19 2026-07-07: 8b is now ONE P:<enc>:all job (m12e --metric all, shared encode) → one "8b·6"
    # column (the 6 pt-metrics run inside it; per-metric progress is in per_clip_<metric>.npy). 8c (F:,
    # encoder-temporal m12f) still fans into 4 columns. tcc carries BOTH tcc_cycle and tcc_tau.
    _FAN = [("P", "all")] + [("F", m) for m in ET_METRICS]
    enc_rows = ([("frozen", enc_name("frozen"))] if _show_frozen else []) \
        + [(a, enc_name(ARM2ENC[a])) for a in DISPLAY_ORDER
           if f"E:{enc_name(ARM2ENC[a])}" in jobs]
    EW, PW = 26, 5

    def _gl(jid):
        return _GLYPH[classify(jid)] if jid in jobs else " "

    def _e_gl(enc):
        # the 2-8 (E:) cell: a resume-SKIPPED E: emits no log ◀/✓ so classify()='pending', but if its 4
        # head-metric outputs exist it IS done → check them directly (head_stage_done, the SAME 'is E: done'
        # source the scheduler resume-skip uses, so cell and scheduler can't disagree). iter19 2026-07-08.
        jid = f"E:{enc}"
        if jid not in jobs:
            return " "
        st = classify(jid)
        if st == "pending" and head_stage_done(mtag, BACKBONE, EVAL_CORPUS, enc):
            st = "done"
        return _GLYPH[st]
    g_top = "┌" + bar * EW + "┬" + bar * 5 + ("┬" + bar * PW) * len(_FAN) + "┐"
    g_mid = "├" + bar * EW + "┼" + bar * 5 + ("┼" + bar * PW) * len(_FAN) + "┤"
    g_bot = "└" + bar * EW + "┴" + bar * 5 + ("┴" + bar * PW) * len(_FAN) + "┘"
    if not enc_rows:
        print("\n  (no eval jobs in this run — --only / train phase; the 8b+8c metric fan renders on the eval box)")
    else:
        print(f"\n  Stage-8b+8c metric fan (encoder × {len(_FAN)} metric jobs · ✓ done · 🔄 run · · pend)")
        print("  " + g_top)
        print("  │" + " encoder".ljust(EW) + "│" + "2-8".center(5)
              + "│" + "│".join(_ABBR[m].center(PW) for _, m in _FAN) + "│")
        print("  " + g_mid)
        for label, enc in enc_rows:
            cells = "│".join(_gl(f"{p}:{enc}:{m}").center(PW) for p, m in _FAN)
            print("  │ " + label.ljust(EW - 1) + "│" + _e_gl(enc).center(5) + "│" + cells + "│")
        print("  " + g_bot)
        _pall = [j for j in jobs if j.startswith(("P:", "F:"))]
        _pc = {k: sum(1 for j in _pall if classify(j) == k) for k in ("done", "running", "pending")}
        print(f"  {_pc['done']}✓ done · {_pc['running']}🔄 running · {_pc['pending']}· pending"
              f"  of {len(_pall)} metric jobs (8b + 8c)")

    # ── summary + run ETA ──
    end_utc = datetime.now(timezone.utc) + timedelta(seconds=eta_secs)
    end_pdt = end_utc - timedelta(hours=7)   # user reads PDT
    print(f"\n  {counts['done']}✅  {counts['running']}🔄  {counts['pending']}⬚  "
          f"{counts['failed']}❌  / {len(jobs)} jobs")
    # iter19 --eval-first: if the run launched with the SSL-head gate, surface its progress here so the
    # pre-training go/no-go is visible at a glance (arm training is HELD until every gate eval finishes).
    _gate_m = re.search(r"SSL-head GATE:.*?for \[([^\]]*)\]", text)
    if _gate_m:
        _garms = re.findall(r"'([^']+)'", _gate_m.group(1))
        _gencs = [enc_name("frozen") if a == "frozen" else enc_name(ARM2ENC[a])
                  for a in _garms if a == "frozen" or a in ARM2ENC]
        _gjobs = [j for j in jobs if any(
            j == f"E:{e}" or j.startswith(f"P:{e}:") or j.startswith(f"F:{e}:") for e in _gencs)]
        # a resume-SKIPPED gate E: emits no log ◀/✓ so classify()='pending' — count it done when its head-metric
        # outputs exist (head_stage_done, same fix as the 2-8 cell), else the gate reads 5/6 forever (iter19 07-08).
        _gdone = sum(1 for j in _gjobs if classify(j) == "done"
                     or (j.startswith("E:") and head_stage_done(mtag, BACKBONE, EVAL_CORPUS, j[2:])))
        _grun = sum(1 for j in _gjobs if classify(j) == "running")
        # the gate holds ONLY train jobs (ngpu_run.py:568). If 0 are pending (both arms already trained on a
        # resume) it holds NOTHING — never print "ARM TRAINING HELD", which reads as a deadlock when it is not.
        _ng_m = re.search(r"→ (\d+) pending train job", text)
        _ngated = int(_ng_m.group(1)) if _ng_m else None
        if _gjobs and _gdone == len(_gjobs):
            print(f"  🚦 E0 SSL-head gate ({', '.join(_garms)}): ✅ CLEARED → arm training released")
        elif _ngated == 0:
            print(f"  🚦 E0 SSL-head gate ({', '.join(_garms)}): ✅ moot — 0 arm-train jobs to hold "
                  f"(both arms already trained); {_gdone}/{len(_gjobs)} gate evals done, {_grun} running")
        elif _gjobs:
            print(f"  🚦 E0 SSL-head gate ({', '.join(_garms)}): 🔄 {_gdone}/{len(_gjobs)} evals done, "
                  f"{_grun} running — ARM TRAINING HELD until it clears")
    fully_done = bool(s3 and s3.group(1) == "0") or only_complete
    settled = counts["done"] + counts["failed"] == len(jobs)
    plot_msg = (maybe_plot(mtag, args.mode)
                if (not settled or counts["failed"]) and not fully_done and not only_arms else None)
    backup_msg = maybe_backup(fully_done, mtag)
    if fully_done:
        print("\a\n" + "█" * 78)
        print(f"  🏁 RUN COMPLETE — plots in {_eval_dir(mtag, BACKBONE, EVAL_CORPUS, 'probe_plot')}/eval/"
              if not only_complete else "  🏁 --only RUN COMPLETE (no evals/finale by design)")
        print(backup_msg)
        print("█" * 78)
    elif settled and counts["failed"]:
        print("\a\n" + "█" * 78)
        print(f"  ⚠️  ALL {len(jobs)} JOBS SETTLED · {counts['failed']}❌ FAILED — DO NOT KILL YET.")
        print("  The scheduler skips the §3 finale on any failure. Fix, then resume the survivors:")
        print(f"       python -u scripts/ngpu_run.py --mode {args.mode} --gpus {gpus} --cache 1")
        if plot_msg:
            print(plot_msg)
        print(backup_msg)
        print("█" * 78)
    elif settled:
        print(f"  🏁 all {len(jobs)} jobs settled · §3 finale {'🔄 running' if '§3 finale' in text else '⬚ pending'}")
        print(backup_msg)
    else:
        print(f"  🏁 run ETA  ~{_dur(eta_secs)} from now  →  {end_utc:%H:%M} UTC · "
              f"{end_pdt:%H:%M} PDT  (incl. §3 finale ~{_dur(s3_pad)} · {s3_src})")
        if plot_msg:
            print(plot_msg)
        print(backup_msg)
    if counts["failed"]:
        print("  ❌ failed (recover with --cache 1): "
              + ", ".join(j for j in jobs if classify(j) == "failed"))
    print("\n  legend: 🚂 pretrain · 🔧 surgery · 🔩 FT baseline · 🔀 merge · 📊 eval │ "
          "✅ done · 🔄 running · ⬚ pending · ❌ failed")

    # --plots: AFTER the ASCII status, refresh the full metrics_watch figure+data set (consolidated path —
    # one command for status + every figure). mtag/mode come from the parsed --mode above.
    if args.plots:
        maybe_metrics_plots(mtag, args.mode)


if __name__ == "__main__":
    main()
