#!/usr/bin/env python3
"""m18_vlm_data — build the VLM instruction JSONL (stage-1 align captions + stage-2 MOTION QA) and
the TempCompass gate set, from the ONLY HF-reachable sources (verified 2026-07-19):
  • lmms-lab/LLaVA-Video-178K  — LLaVA `conversations` format: captions + MC/OE QA + video tars.
  • lmms-lab/TempCompass       — temporal MC-QA parquets (`dim` = action/direction/speed/order/…) + videos zip.

Emits, per line: {"video": <abspath>, "prompt": "<video>\\n<question>", "answer": <text>, "task": <tag>}.
Stage-2 keeps MOTION/temporal QA only (config keep/drop keywords) — NO scene/appearance QA
(OURS loses scene 0/15 → would dilute/invert the encoder gap). Gold ref: github.com/haotian-liu/LLaVA
(instruction JSONL) + the V-JEPA2 paper's LLaVA-style alignment (arxiv 2506.09985).

Video files are large (tars/zip) → downloaded+extracted ONLY with --download-videos (a 96GB-box step);
without it the JSONL still builds with the expected local paths so training can be staged.

USAGE (96GB box · /venv/main):
    python src/m18_vlm_data.py --config configs/vlm.yaml --stage align    --download-videos --cache-policy 1
    python src/m18_vlm_data.py --config configs/vlm.yaml --stage instruct --download-videos --cache-policy 1
    python src/m18_vlm_data.py --config configs/vlm.yaml --stage gate      --download-videos --cache-policy 1
"""
import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive, is_recompute

VIDEO_TOKEN = "<video>"
LLAVA_REPO = "lmms-lab/LLaVA-Video-178K"
TC_REPO = "lmms-lab/TempCompass"
# TempCompass `dim` values that are MOTION/temporal (visible + coarse — OURS's strength); scene/attr excluded.
TC_TEMPORAL_DIMS = {"action", "direction", "speed", "order", "event_order", "attribute_change"}


def _hf(repo, fname):
    return Path(hf_hub_download(repo, fname, repo_type="dataset"))


def _extract_tars(repo, files, dest, policy):
    """Download + extract a list of tar.gz video files into dest (idempotent — skips if present)."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in tqdm(files, desc=f"videos[{dest.name}]"):
        marker = dest / (Path(f).name + ".done")
        if marker.exists() and not is_recompute(policy):   # str policy — is_recompute handles '2'/'recompute'
            continue
        tp = _hf(repo, f)
        with tarfile.open(tp) as tf:
            tf.extractall(dest)                       # LLaVA video tars carry academic_source/... paths
        marker.write_text("ok")


def _parse_llava(rec, keep, drop):
    """LLaVA conversation record → (question, answer) or None if it fails the motion filter."""
    conv = rec["conversations"]
    human = next((c["value"] for c in conv if c["from"] == "human"), None)
    gpt = next((c["value"] for c in conv if c["from"] == "gpt"), None)
    if human is None or gpt is None:
        return None
    q = human.replace("<image>", "").replace("<video>", "").strip()
    ql = q.lower()
    if keep and not any(k in ql for k in keep):
        return None
    if drop and any(k in ql for k in drop):
        return None
    return q, gpt.strip()


def stage_align(cfg, out, dl, policy):
    """Stage-1 captions from LLaVA-Video (cap json). Projector-alignment data (no motion filter)."""
    d = cfg["data"]["align"]
    files = list_repo_files(LLAVA_REPO, repo_type="dataset")
    cap_jsons = [f for f in files if f.endswith("_cap_processed.json")]
    if not cap_jsons:
        sys.exit("FATAL: no *_cap_processed.json in LLaVA-Video-178K")
    rows, subsets_used = [], set()
    for cj in cap_jsons:
        recs = json.load(open(_hf(LLAVA_REPO, cj)))
        sub = cj.split("/")[0]
        for r in recs:
            pr = _parse_llava(r, keep=[], drop=[])
            if pr is None:
                continue
            rows.append({"video": str(out / "videos" / r["video"]),
                         "prompt": f"{VIDEO_TOKEN}\n{pr[0]}", "answer": pr[1], "task": "caption"})
            subsets_used.add(sub)
            if len(rows) >= d["max_samples"]:
                break
        if len(rows) >= d["max_samples"]:
            break
    if dl:
        vids = [f for f in files if f.endswith(".tar.gz") and f.split("/")[0] in subsets_used]
        _extract_tars(LLAVA_REPO, vids, out / "videos", policy)
    _write(out / "align.jsonl", rows, policy)


def stage_instruct(cfg, out, dl, policy, max_tars, max_samples):
    """Stage-2 MOTION/temporal QA from LLaVA-Video (mc + oe json), keyword-filtered.

    COST-BOUNDED: pulls ONE subset and at most --max-tars video shards (~5.3 GB each), then keeps only rows
    whose clip actually landed on disk. The EARLY GATE needs ~3k samples → 1 shard (~5 GB), not the subset's
    full 38 GB. Raise --max-tars / --max-samples for the full stage-2 run."""
    d = cfg["data"]["instruct"]
    keep, drop = d["keep_task_keywords"], d["drop_task_keywords"]
    cap = max_samples or d["max_samples"]
    files = list_repo_files(LLAVA_REPO, repo_type="dataset")
    qa_jsons = sorted(f for f in files if f.endswith(("_mc_v0_1_qa_processed.json", "_oe_v0_1_qa_processed.json")))
    if not qa_jsons:
        sys.exit("FATAL: no *_qa_processed.json in LLaVA-Video-178K")
    subset = qa_jsons[0].split("/")[0]
    if dl:
        tars = sorted(f for f in files if f.endswith(".tar.gz") and f.split("/")[0] == subset)
        if max_tars:
            tars = tars[:max_tars]
        print(f"[m18 data] subset={subset} · pulling {len(tars)} video shard(s) (~5.3 GB each)")
        _extract_tars(LLAVA_REPO, tars, out / "videos", policy)
    vroot = out / "videos"
    rows, missing = [], 0
    for qj in [f for f in qa_jsons if f.split("/")[0] == subset]:
        for r in json.load(open(_hf(LLAVA_REPO, qj))):
            pr = _parse_llava(r, keep, drop)
            if pr is None:
                continue
            vp = vroot / r["video"]
            if not vp.exists():
                missing += 1
                continue
            rows.append({"video": str(vp), "prompt": f"{VIDEO_TOKEN}\n{pr[0]}",
                         "answer": pr[1], "task": "motion_qa"})
            if len(rows) >= cap:
                break
        if len(rows) >= cap:
            break
    if not rows:
        sys.exit(f"FATAL: 0 motion-QA rows with a downloaded clip (skipped {missing} for missing video) — "
                 f"raise --max-tars or loosen keep/drop keywords in vlm.yaml")
    print(f"[m18 data] kept {len(rows)} rows · skipped {missing} whose clip is not downloaded")
    _write(out / "instruct.jsonl", rows, policy)


def _mp4_root(vdir):
    """tempcompass_videos.zip NESTS its clips (…/tempcompass_videos/videos/*.mp4) — return the dir that
    actually holds the mp4s, so the JSONL paths resolve (else eval FileNotFoundErrors on every row)."""
    if list(vdir.glob("*.mp4")):
        return vdir
    for d in sorted(p for p in vdir.rglob("*") if p.is_dir()):
        if list(d.glob("*.mp4")):
            return d
    return vdir


def stage_gate(cfg, out, dl, policy):
    """TempCompass temporal MC-QA + yes_no → gate JSONL. Videos from tempcompass_videos.zip."""
    files = list_repo_files(TC_REPO, repo_type="dataset")
    vdir = out / "tempcompass_videos"
    if dl:
        zf = _hf(TC_REPO, "tempcompass_videos.zip")
        vdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zf) as z:
            z.extractall(vdir)
    vroot = _mp4_root(vdir)
    have = {p.stem for p in vroot.glob("*.mp4")}          # keep ONLY rows whose clip is on disk
    rows, skipped = [], 0
    for pq in [f for f in files if f.endswith(".parquet") and ("multi-choice" in f or "yes_no" in f)]:
        df = pd.read_parquet(_hf(TC_REPO, pq))
        for _, r in df.iterrows():
            if r["dim"] not in TC_TEMPORAL_DIMS:
                continue
            vid = str(r["video_id"])
            if have and vid not in have:
                skipped += 1
                continue
            # TempCompass mixes MULTI-CHOICE (answer "A. dunking…") and YES/NO (answer "yes"/"no") rows.
            # They need DIFFERENT instructions and different answer parsing — telling a yes/no question to
            # "answer with the option's letter" is nonsense, and a letter-only extractor can never score it
            # (2026-07-19: 60% of the gate set was structurally unanswerable until this fix). `fmt` carries
            # the format through to m18_vlm_eval.extract_answer.
            ans = str(r["answer"]).strip()
            yn = ans.lower().startswith(("yes", "no"))
            instr = "Answer yes or no." if yn else "Answer with the option's letter."
            rows.append({"video": str(vroot / f"{vid}.mp4"),
                         "prompt": f"{VIDEO_TOKEN}\n{r['question'].strip()}\n{instr}",
                         "answer": ans, "task": r["dim"], "fmt": "yesno" if yn else "mc"})
    if not rows:
        sys.exit("FATAL: 0 TempCompass temporal rows — check TC_TEMPORAL_DIMS vs the parquet `dim` values")
    if skipped:
        print(f"[m18 data] skipped {skipped} rows whose video is absent from the zip ({len(have)} clips on disk)")
    _write(out / "gate_tempcompass.jsonl", rows, policy)


def _write(path, rows, policy):
    # JSONL is a cheap DERIVED output (not a protected cache) → always rebuild (open 'w' overwrites).
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"[m18 data] {path} : {len(rows)} rows · tasks {dict(Counter(r['task'] for r in rows))}")


def main():
    p = argparse.ArgumentParser(description="m18 — VLM instruction/gate data builder")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", required=True, choices=["align", "instruct", "gate"])
    p.add_argument("--download-videos", action="store_true", help="also fetch+extract the video tars/zip (96GB-box step)")
    p.add_argument("--max-tars", type=int, default=None, help="instruct: cap video shards pulled (~5.3 GB each)")
    p.add_argument("--max-samples", type=int, default=None, help="instruct: cap rows (overrides cfg max_samples)")
    add_cache_policy_arg(p)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["vlm"]
    policy = resolve_cache_policy_interactive(args.cache_policy)
    out = Path(cfg["data"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "instruct":
        stage_instruct(cfg, out, args.download_videos, policy, args.max_tars, args.max_samples)
    elif args.stage == "gate":
        stage_gate(cfg, out, args.download_videos, policy)
    else:
        stage_align(cfg, out, args.download_videos, policy)


if __name__ == "__main__":
    main()
