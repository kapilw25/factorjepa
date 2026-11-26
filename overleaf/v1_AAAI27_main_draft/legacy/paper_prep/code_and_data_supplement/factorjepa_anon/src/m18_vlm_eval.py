#!/usr/bin/env python3
"""m18_vlm_eval — eval a trained VLM arm on a benchmark, then GATE OURS vs FROZEN. Runs on 96GB box.

  --stage eval  --arm {frozen,ours} [--early] : greedy-decode the MC answer per clip → preds_<arm>.json
                                                (--early = small held-out TempCompass subset, cheap).
  --stage gate  [--early]                     : read both arms' preds → accuracy + BCa-style bootstrap
                                                95% CI → verdict. Dumps gate_report.json + heroes_vlm.json
                                                (OURS-right / FROZEN-wrong → the demo_cosmos cards).

Honest gate (CLAUDE.md 95% CI mandatory):
  • EARLY : PASS iff (acc_OURS − acc_FROZEN) ≥ gate.early_min_delta_pp. Fail → STOP before full pretrain.
  • FULL  : PASS iff acc_OURS CI-low > acc_FROZEN CI-high (non-overlapping 95% CIs). Fail → forest plots.

bs=1 generation (correct left-context; no left-pad hazard with inputs_embeds).
USAGE (96GB box):
    for ARM in frozen ours; do python src/m18_vlm_eval.py --config configs/vlm.yaml --stage eval --arm $ARM --early; done
    python src/m18_vlm_eval.py --config configs/vlm.yaml --stage gate --early     # → continue? or STOP
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.vlm_model import VJepaLlavaVLM, build_chat
from utils.frozen_features import resize_and_normalize
from utils.demo_video import decode_all_frames
from utils.config import get_model_config


def extract_answer(text, fmt):
    """Format-aware answer parsing. MC → first standalone A-E; YES/NO → first standalone yes|no (→ 'Y'/'N',
    matching gt_letter's first-char convention). '?' when unparseable (scored WRONG, equally for both arms).
    A letter-only extractor silently zeroed every yes/no row — 60% of the gate set (2026-07-19)."""
    t = text.strip()
    if fmt == "yesno":
        m = re.search(r"\b(yes|no)\b", t.lower())
        return m.group(1)[0].upper() if m else "?"
    m = re.search(r"\b([A-E])\b", t)
    if m:
        return m.group(1)
    c = t[:1].upper()
    return c if c in "ABCDE" else "?"


def gt_letter(answer):
    return answer.strip()[:1].upper()


def _bench_jsonl(v):
    """v = the INNER cfg["vlm"] dict (not the full yaml) — indexing the wrong level is a runtime KeyError
    that lint cannot catch (2026-07-19)."""
    return Path(v["data"]["out_dir"]) / "gate_tempcompass.jsonl"


def eval_arm(cfg, arm, early, limit=None):
    v = cfg["vlm"]
    out = Path(v["render"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    model = VJepaLlavaVLM(cfg, arm, load_llm=True, lora=True)
    ck = Path(v["render"]["out_dir"]).parent / "vlm_ckpt" / arm
    pj = ck / "instruct_projector.pt"
    if not pj.exists():
        sys.exit(f"FATAL: {pj} missing — run m18_vlm_train --stage instruct --arm {arm} first")
    model.projector.load_state_dict(torch.load(pj, map_location="cuda"))
    from peft import PeftModel
    model.llm = PeftModel.from_pretrained(model.llm, ck / "lora").to("cuda")
    model.projector.eval()

    rows = [json.loads(x) for x in open(_bench_jsonl(v))]
    rng = np.random.default_rng(0)
    if early:                                                    # small disjoint subset by video
        vids = sorted({r["video"] for r in rows})
        keep = set(rng.choice(vids, min(v["gate"]["early_subset_videos"], len(vids)), replace=False))
        rows = [r for r in rows if r["video"] in keep]
    if limit:                                                    # smoke: validate generate() cheaply
        rows = rows[:limit]
        print(f"[m18 eval] LIMITED to {len(rows)} rows (--limit)")
    crop = get_model_config(v["encoder"]["model_config"])["model"]["crop_size"]
    nf = v["encoder"]["num_frames"]
    preds = []
    for r in tqdm(rows, desc=f"eval[{arm}{'/early' if early else ''}]"):
        vp = Path(r["video"])
        if not vp.exists():
            raise FileNotFoundError(f"video missing: {vp} (run m18_vlm_data --stage gate --download-videos)")
        frames = decode_all_frames(vp)
        idx = np.linspace(0, len(frames) - 1, nf).round().astype(int)
        pixels = resize_and_normalize(frames[idx], crop).permute(1, 0, 2, 3).contiguous().unsqueeze(0).to("cuda", torch.bfloat16)
        p_ids, _ = build_chat(model.tokenizer, r["prompt"], None, model.enable_thinking)   # chat-templated, non-thinking
        ids = torch.tensor([p_ids], device="cuda")
        attn = torch.ones_like(ids)
        gen = model.generate(pixels, ids, attn, max_new_tokens=8)[0]
        pred = extract_answer(gen, r["fmt"])
        gt = gt_letter(r["answer"])
        # `raw` = the VLM's VERBATIM generation, kept so a human can eyeball INPUT prompt vs OUTPUT text
        # (without it you only see the parsed letter and cannot tell a wrong answer from a parse failure).
        preds.append({"video": r["video"], "task": r["task"], "fmt": r["fmt"], "question": r["prompt"],
                      "answer": r["answer"], "gt": gt, "pred": pred, "raw": gen,
                      "correct": int(pred == gt)})
    tag = "early" if early else "full"
    fp = out / f"preds_{arm}_{tag}.json"
    json.dump(preds, open(fp, "w"))
    print(f"[m18 eval] {arm}/{tag}: acc {np.mean([p['correct'] for p in preds]):.3f} over {len(preds)} → {fp}")


def _boot_ci(correct, iters, ci):
    c = np.asarray(correct, float)
    n = len(c)
    rng = np.random.default_rng(0)
    means = c[rng.integers(0, n, (iters, n))].mean(1)          # vectorized bootstrap (CLAUDE.md)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(c.mean()), float(lo), float(hi)


def gate(cfg, early):
    v = cfg["vlm"]; g = v["gate"]; out = Path(v["render"]["out_dir"])
    tag = "early" if early else "full"
    fr = json.load(open(out / f"preds_frozen_{tag}.json"))
    ou = json.load(open(out / f"preds_ours_{tag}.json"))
    if len(fr) != len(ou):
        sys.exit(f"FATAL: arm pred counts differ ({len(fr)} vs {len(ou)}) — re-run eval for both arms")
    af, lf, hf = _boot_ci([p["correct"] for p in fr], g["bootstrap_iters"], g["ci"])
    ao, lo, ho = _boot_ci([p["correct"] for p in ou], g["bootstrap_iters"], g["ci"])
    if early:
        passed = (ao - af) * 100 >= g["early_min_delta_pp"]
        rule = f"(OURS−FROZEN) {(ao-af)*100:+.1f}pp ≥ {g['early_min_delta_pp']}pp"
    else:
        passed = lo > hf
        rule = f"OURS CI-low {lo:.3f} > FROZEN CI-high {hf:.3f} (non-overlap)"
    heroes = [{**o, "frozen_pred": f["pred"]} for f, o in zip(fr, ou) if o["correct"] and not f["correct"]]
    report = {"stage": tag, "frozen": {"acc": af, "ci": [lf, hf]}, "ours": {"acc": ao, "ci": [lo, ho]},
              "delta_pp": (ao - af) * 100, "rule": rule, "PASS": bool(passed), "n": len(fr), "n_heroes": len(heroes)}
    json.dump(report, open(out / f"gate_report_{tag}.json", "w"))
    json.dump(heroes, open(out / f"heroes_vlm_{tag}.json", "w"))
    verdict = "✅ PASS" if passed else "⛔ FAIL"
    print(f"[m18 gate/{tag}] FROZEN {af:.3f} [{lf:.3f},{hf:.3f}]  ·  OURS {ao:.3f} [{lo:.3f},{ho:.3f}]")
    print(f"[m18 gate/{tag}] {verdict} — {rule} · {len(heroes)} OURS-right/FROZEN-wrong heroes")
    if not passed:
        print(f"[m18 gate/{tag}] TRUTHFUL NEGATIVE — do NOT render; "
              + ("STOP before full pretrain." if early else "ship the forest plots."))


def main():
    p = argparse.ArgumentParser(description="m18 — VLM eval + OURS-vs-FROZEN gate")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", required=True, choices=["eval", "gate"])
    p.add_argument("--arm", choices=["frozen", "ours"])
    p.add_argument("--early", action="store_true", help="small held-out subset (early cheap gate)")
    p.add_argument("--limit", type=int, default=None, help="cap eval rows (smoke-test generate cheaply)")
    args = p.parse_args()
    if args.stage == "eval" and not torch.cuda.is_available():
        sys.exit("FATAL: CUDA required (96GB box)")
    cfg = yaml.safe_load(args.config.read_text())
    if args.stage == "eval":
        if not args.arm:
            sys.exit("FATAL: --stage eval requires --arm")
        eval_arm(cfg, args.arm, args.early, args.limit)
    else:
        gate(cfg, args.early)


if __name__ == "__main__":
    main()
