#!/usr/bin/env python3
"""vlm_eyeball — dump VLM eval records into a HUMAN-READABLE form so a person can manually verify
(1) the INPUT prompt actually sent to the model and (2) the OUTPUT text it generated, next to the
ground truth and the parsed answer.

Why this exists: preds_<arm>_<tag>.json stores the PARSED answer ('A'/'Y'/'?'), which cannot distinguish
"the model answered wrongly" from "the model answered fine but our parser missed it" — that distinction
silently zeroed 60% of the gate set once already (2026-07-19). `raw` is the verbatim generation.

Writes BOTH:
  <out>.txt   — flat, scannable: PROMPT / GT / RAW OUTPUT / PARSED / verdict per record
  <out>.json  — the same records, machine-readable

USAGE:
    python src/utils/vlm_eyeball.py --preds outputs/demo/vlm/preds_ours_early.json \
        --out outputs/demo/vlm/eyeball_ours_early --n 40
    python src/utils/vlm_eyeball.py --preds ... --out ... --n 40 --only wrong   # inspect failures only
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="human-readable dump of VLM eval records")
    p.add_argument("--preds", type=Path, required=True, help="preds_<arm>_<tag>.json from m18_vlm_eval")
    p.add_argument("--out", type=Path, required=True, help="output stem (writes .txt and .json)")
    p.add_argument("--n", type=int, required=True, help="how many records to dump")
    p.add_argument("--only", choices=["all", "wrong", "right", "unparsed"], default="all")
    args = p.parse_args()

    recs = json.load(open(args.preds))
    if args.only == "wrong":
        recs = [r for r in recs if not r["correct"]]
    elif args.only == "right":
        recs = [r for r in recs if r["correct"]]
    elif args.only == "unparsed":
        recs = [r for r in recs if r["pred"] == "?"]
    recs = recs[: args.n]
    if not recs:
        raise SystemExit(f"FATAL: no records matched --only {args.only} in {args.preds}")

    lines = []
    for i, r in enumerate(recs, 1):
        raw = r.get("raw", "<not captured — re-run eval with the current m18_vlm_eval.py>")
        lines += [
            "=" * 100,
            f"[{i}/{len(recs)}]  clip={Path(r['video']).name}   dim={r['task']}   fmt={r.get('fmt', '?')}",
            "-" * 100,
            "INPUT PROMPT (verbatim, as sent to the VLM):",
            r["question"],
            "",
            f"GROUND TRUTH : {r['answer']}        (scored as '{r['gt']}')",
            f"VLM RAW OUTPUT: {raw!r}",
            f"PARSED ANSWER : '{r['pred']}'",
            f"VERDICT       : {'CORRECT' if r['correct'] else 'WRONG'}"
            + ("   <-- PARSE FAILURE (model said something we could not read)" if r["pred"] == "?" else ""),
            "",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".txt").write_text("\n".join(lines))
    json.dump(recs, open(str(args.out) + ".json", "w"), indent=2)
    nwrong = sum(1 for r in recs if not r["correct"])
    nunp = sum(1 for r in recs if r["pred"] == "?")
    print(f"[vlm_eyeball] {len(recs)} records ({nwrong} wrong, {nunp} unparsed) → "
          f"{args.out}.txt  +  {args.out}.json")


if __name__ == "__main__":
    main()
