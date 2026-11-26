#!/usr/bin/env python3
"""Clear regenerable resume anchors (`*ckpt_latest*.pt` + `*ckpt_stage*.pt`) for arms whose
training is FINISHED — reclaiming disk before an HF upload — WITHOUT touching the final
deliverables (`student_encoder.pt`, `*ckpt_best*.pt`, `motion_aux_head.pt`, eval `.npy/.json`,
`probe.pt`, plots) or the resume state of PARKED arms.

WHY a dedicated guarded helper (not `find -delete`):
  · delete-protection (src/CLAUDE.md): destructive deletes live in .py behind `--cache-policy`,
    never as shell `rm`/`find -delete` (the checkpoint-protection hook blocks those anyway).
  · the parked-arm carve-out is the whole point: an arm with NO `student_encoder.pt` (training
    interrupted / parked — e.g. iter18's cassle_encoder, ewc_encoder per runbook §0.C C4) STILL
    needs its `*ckpt_latest*.pt` to resume. A blanket anchor-wipe would force it to retrain from
    scratch. This helper PRESERVES anchors for any arm missing its final exported encoder.

RULE: clear an arm's resume anchors IFF `<arm>/student_encoder.pt` exists (training done →
      `ckpt_best.pt` + `student_encoder.pt` are exported → the latest/stage anchors are pure
      regenerable scaffolding). Otherwise preserve them and say so.

USAGE:
  # DRY-RUN (policy=1=keep, default): list every anchor that WOULD be cleared + what's preserved.
  python -u src/utils/clear_resume_anchors.py --root outputs/poc/vjepa_2_1_vitG --cache-policy 1

  # DELETE (policy=2=recompute): actually remove the DONE-arm anchors via guarded_delete.
  python -u src/utils/clear_resume_anchors.py --root outputs/poc/vjepa_2_1_vitG --cache-policy 2 \
      2>&1 | tee logs/clear_resume_anchors_$(date +%Y%m%d_%H%M%S).log

  # also drop the predictor-bearing best ckpts (only if you will NEVER re-run eval):
  python -u src/utils/clear_resume_anchors.py --root outputs/poc/vjepa_2_1_vitG --also-best --cache-policy 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cache_policy import (  # noqa: E402
    add_cache_policy_arg,
    guarded_delete,
    is_recompute,
    resolve_cache_policy_interactive,
)

# Regenerable resume scaffolding — safe to drop once an arm has exported its final encoder.
_ANCHOR_SUBSTRINGS = ("ckpt_latest", "ckpt_stage")
# The predictor-bearing best ckpt — needed ONLY to re-run eval (stage 8); opt-in via --also-best.
_BEST_SUBSTRING = "ckpt_best"
# Final deliverables that are NEVER touched (here for the docstring/self-check, not matched).
_KEEP_ALWAYS = ("student_encoder.pt", "motion_aux_head.pt", "probe.pt")


def _arm_anchors(arm_dir: Path, include_best: bool):
    """(anchor .pt paths, total_bytes) for one arm dir — latest/stage always, best if requested."""
    subs = _ANCHOR_SUBSTRINGS + ((_BEST_SUBSTRING,) if include_best else ())
    hits = [p for p in arm_dir.rglob("*.pt") if any(s in p.name for s in subs)]
    return hits, sum(p.stat().st_size for p in hits)


def _gb(n):
    return n / 1024 ** 3


def clear_resume_anchors(root: Path, policy: str, include_best: bool) -> dict:
    """Walk `root`'s arm dirs; clear (policy=2) or preview (policy=1) anchors for DONE arms only.
    Returns {"cleared_bytes", "preserved_bytes", "done": [...], "parked": [...]}."""
    if not root.is_dir():
        sys.exit(f"FATAL: --root not a directory: {root}")
    arms = sorted(d for d in root.iterdir() if d.is_dir())
    if not arms:
        sys.exit(f"FATAL: no arm subdirectories under {root}")
    target_bytes = preserved_bytes = 0   # target = anchors on DONE arms (cleared@2 / would-clear@1)
    done, parked = [], []
    recompute = is_recompute(policy)
    print(f"  root: {root}  ·  arms: {len(arms)}  ·  policy={policy} "
          f"({'DELETE' if recompute else 'dry-run/keep'})  ·  also-best={include_best}\n")
    for arm in arms:
        hits, nbytes = _arm_anchors(arm, include_best)   # measured BEFORE any delete
        if not (arm / "student_encoder.pt").exists():
            # PARKED / interrupted — its anchors are the only way to resume → preserve, loudly.
            preserved_bytes += nbytes
            parked.append(arm.name)
            print(f"  ⏸️  PRESERVE  {arm.name:34s} no student_encoder.pt (parked/resumable) "
                  f"— kept {_gb(nbytes):.1f} G of anchors")
            continue
        done.append(arm.name)
        target_bytes += nbytes
        if not hits:
            print(f"  ✅ clean    {arm.name:34s} DONE · no anchors left")
            continue
        print(f"  🧹 {'deleting' if recompute else 'WOULD clear':12s} {arm.name:34s} DONE · "
              f"{len(hits)} anchor(s) · {_gb(nbytes):.1f} G")
        for p in hits:                                   # guarded_delete: no-op unless policy=2
            guarded_delete(p, policy, label=f"resume-anchor {arm.name}/{p.name}")
    return {"target_bytes": target_bytes, "preserved_bytes": preserved_bytes,
            "done": done, "parked": parked}


# ON-COMPLETION (in-trainer) sweep — latest/stage are pure scaffolding once the final encoder is exported;
# add ckpt_step (the keep_last_n survivors) so a completed arm leaves ZERO resume anchors on disk.
_COMPLETION_ANCHOR_SUBSTRINGS = _ANCHOR_SUBSTRINGS + ("ckpt_step",)


def clear_anchors_on_completion(arm_dir: Path) -> int:
    """Delete ONE arm's resume anchors (ckpt_latest / ckpt_stage* / ckpt_step*) right after a SUCCESSFUL
    training run — called from inside the trainer, AFTER export_student_for_eval. Returns bytes freed.

    UN-GATED by cache_policy (unlike the manual `clear_resume_anchors` CLI above): a resume anchor is
    single-arm training SCRATCH, not a durable cross-module CACHE, so the 'Clean all intermediates after
    training' policy (src/CLAUDE.md) applies on SUCCESS regardless of --cache 1/2. This is the ONE
    sanctioned exception to the 'no .py delete without cache_policy=2' rule — narrow by construction:

    SAFETY — refuses unless BOTH durable eval artifacts already exist in arm_dir: student_encoder.pt AND a
    *ckpt_best*.pt. So it can NEVER strip the resume state of an incomplete/failed/parked run (which has no
    student_encoder.pt → keeps every anchor), and it only ever removes latest/stage/step (never best/student)."""
    student = arm_dir / "student_encoder.pt"
    has_best = any(_BEST_SUBSTRING in p.name for p in arm_dir.glob("*.pt"))
    if not (student.exists() and has_best):
        raise RuntimeError(
            f"clear_anchors_on_completion: refusing in {arm_dir} — durable artifacts missing "
            f"(student_encoder.pt={student.exists()}, *{_BEST_SUBSTRING}*={has_best}); call ONLY after a "
            f"successful export, otherwise the resume anchors are still needed to resume.")
    freed = 0
    for p in sorted(arm_dir.glob("*.pt")):
        if any(s in p.name for s in _COMPLETION_ANCHOR_SUBSTRINGS):
            sz = p.stat().st_size
            p.unlink()
            freed += sz
            print(f"  [anchor-cleanup] removed {p.name} ({_gb(sz):.1f} G)")
    print(f"  [anchor-cleanup] {arm_dir.name}: freed {_gb(freed):.1f} G of resume anchors "
          f"(kept student_encoder.pt + *{_BEST_SUBSTRING}*)")
    return freed


# ── Training-completion GATE (iter19 2026-07-08, user order) ─────────────────────────────────────────────
# The SINGLE definition of "did an m09 arm finalize cleanly?" — reused (same anchor patterns as
# clear_anchors_on_completion above, one source) by BOTH each m09 trainer (assert_finalized at its own exit →
# FAIL LOUD) AND the ngpu_run orchestrator's --cache 1 resume (is_finalized → RE-RUN an incomplete arm, never
# skip it). The POLICY of what "trained" means lives HERE with the trainers, not scattered in the orchestrator.
# A CLEAN finalize leaves student_encoder.pt + a *ckpt_best* (predictor-bearing, needed for eval Stage 8/8b)
# AND ZERO resume anchors. peft_lora's 2026-07-06 run FATAL'd at the post-loop _best save → student present but
# NO _best and a SURVIVING ckpt_stage0 anchor ⇒ NOT finalized; the old student-only check silently shipped it
# minus its 7 predictor metrics.


def finalize_missing(arm_dir) -> list:
    """Finalize artifacts ABSENT / not-cleaned in arm_dir — EMPTY list ⇒ a clean finalize. Checks the two
    durable eval deliverables (student_encoder.pt + any *ckpt_best*.pt) AND that no resume anchor survived."""
    arm_dir = Path(arm_dir)
    missing = []
    if not (arm_dir / "student_encoder.pt").exists():
        missing.append("student_encoder.pt")
    if not any(_BEST_SUBSTRING in p.name for p in arm_dir.glob("*.pt")):
        missing.append(f"*{_BEST_SUBSTRING}*.pt (predictor-bearing best ckpt)")
    missing += [f"UNCLEANED-anchor:{p.name}" for p in sorted(arm_dir.glob("*.pt"))
                if any(s in p.name for s in _COMPLETION_ANCHOR_SUBSTRINGS)]
    return missing


def is_finalized(arm_dir) -> bool:
    """True iff arm_dir is a clean m09 finalize (student_encoder.pt + *ckpt_best* present, no surviving resume
    anchors). The orchestrator's resume-skip gate: an arm returning False is RE-RUN (resume→finalize), never
    laundered into 'already-trained' off student_encoder.pt alone (that file is written BEFORE the _best save)."""
    return not finalize_missing(arm_dir)


def assert_finalized(arm_dir, label: str = ""):
    """FAIL LOUD (SystemExit) if arm_dir did not finalize cleanly. Each m09 trainer calls this as its LAST step,
    right after clear_anchors_on_completion, so a botched finalize (missing _best, uncleaned anchor) is caught AT
    TRAINING TIME by the trainer itself — never silently inherited as 'trained' by a downstream resume."""
    miss = finalize_missing(arm_dir)
    if miss:
        sys.exit(f"FATAL [train-completion{': ' + label if label else ''}]: {Path(arm_dir)} did NOT finalize "
                 f"cleanly — missing/uncleaned {miss}. Eval needs student_encoder.pt + the predictor-bearing "
                 f"*{_BEST_SUBSTRING}*.pt, and clear_anchors_on_completion must have removed every resume anchor.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="training output root for ONE backbone, e.g. outputs/poc/vjepa_2_1_vitG")
    ap.add_argument("--also-best", action="store_true",
                    help="ALSO clear *ckpt_best*.pt (the predictor-bearing best ckpt) — only safe if "
                         "you will never re-run eval (stage 8 needs it). OFF by default.")
    add_cache_policy_arg(ap)
    args = ap.parse_args()
    policy = resolve_cache_policy_interactive(args.cache_policy)

    res = clear_resume_anchors(args.root, policy, args.also_best)
    print()
    if is_recompute(policy):
        print(f"  ✅ CLEARED ~{_gb(res['target_bytes']):.1f} G of resume anchors "
              f"from {len(res['done'])} DONE arms.")
    else:
        print(f"  🔎 DRY-RUN: would clear ~{_gb(res['target_bytes']):.1f} G from "
              f"{len(res['done'])} DONE arms — rerun with --cache-policy 2 to delete.")
    print(f"  ⏸️  preserved ~{_gb(res['preserved_bytes']):.1f} G of anchors for "
          f"{len(res['parked'])} PARKED arm(s): {res['parked'] or '—'}")


if __name__ == "__main__":
    main()
