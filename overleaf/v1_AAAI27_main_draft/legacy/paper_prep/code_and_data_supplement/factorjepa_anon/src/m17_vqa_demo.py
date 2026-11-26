"""m17 — VQA card demo: OURS answers RIGHT where FROZEN answers WRONG (mimics demo_cosmos.png).

Renders a video of multiple-choice cards over DenseWorld clips. For each hero clip the card shows the
QUESTION + options A-D + GROUND TRUTH, then reveals the two model answers: FROZEN (red, wrong) vs OURS
(green, right). "Answer" = an ATTENTIVE probe head (V-JEPA 2's own eval protocol — AttentiveClassifier,
NOT a linear layer) on FROZEN vs OURS V-JEPA features (SAME probe architecture, only the
backbone differs) — computed leakage-safe (GroupKFold-by-video OOF) in scratchpad/anticip_precheck.py +
hero_extract.py; this module ONLY renders the precomputed predictions (no GPU, no model here).

Honesty (feedback_no_hallucinated_victory): the demo shows SELECTED clips, so every frame carries the
AGGREGATE caption ("OURS X% vs FROZEN Y% over N clips"). Heroes are ranked by how far FROZEN missed
(most convincing first) and diversified across videos.

USAGE (CPU-only; run from repo root):
  source venv_denseworld/bin/activate
  python src/m17_vqa_demo.py \
    --heroes outputs/demo/vqa/heroes.json \
    --config configs/demo.yaml \
    --output-dir outputs/demo/vqa \
    --sanity            # (optional) render only 2 clips for a fast smoke
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import utils.*` resolves when run directly
from utils.demo_video import decode_all_frames  # noqa: E402

_FONT_REG = font_manager.findfont("DejaVu Sans")
_FONT_BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
_MAG_ORDER = ["still", "slow", "medium", "fast"]  # optical-flow magnitude quartile order (still<...<fast)
_LETTERS = ["A", "B", "C", "D"]


def _class_order(qcfg):
    """Class-id order for THIS question, so m17 is not welded to the magnitude task.

    The module hardcoded _MAG_ORDER, which silently mislabels any question whose classes
    aren't still/slow/medium/fast (a 2-way L/R turn card would index the wrong letters).
    Questions may declare `classes:`; magnitude keeps its historical order as the default.
    """
    return list(qcfg.get("classes", _MAG_ORDER))


def _font(sz, bold=True):
    return ImageFont.truetype(_FONT_BOLD if bold else _FONT_REG, sz)


def _rank_and_select(heroes, n_clips, max_per_video, order):
    """Rank by FROZEN's error magnitude (bins off) desc — the most layman-convincing misses first —
    then diversify: at most `max_per_video` per video, take n_clips.

    `order` is the question's own class order; using the module-level magnitude list here
    raised ValueError on any other label set (e.g. a 2-way L/R turn question).
    """
    def bins_off(h):
        return abs(order.index(h["frozen"]) - order.index(h["true"]))
    ranked = sorted(heroes, key=lambda h: (-bins_off(h), h["key"]))
    seen, out = {}, []
    for h in ranked:
        v = h["video_id"]
        if seen.get(v, 0) >= max_per_video:
            continue
        seen[v] = seen.get(v, 0) + 1
        out.append(h)
        if len(out) >= n_clips:
            break
    if len(out) < n_clips:  # not enough distinct videos → relax the per-video cap, keep ranking
        for h in ranked:
            if h not in out:
                out.append(h)
            if len(out) >= n_clips:
                break
    return out[:n_clips]


def _resample(frames, n_out):
    idx = np.linspace(0, len(frames) - 1, n_out).round().astype(int)
    return frames[idx]


def _draw_card(rgb, card, hero, reveal, agg, opts, question, order):
    """Draw the MCQ card onto one RGB frame (uint8 HxWx3). reveal=False → question+GT only;
    reveal=True → also the two model answers. Returns a new uint8 frame."""
    im = Image.fromarray(rgb).convert("RGBA")
    W, H = im.size
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    col = card["colors"]
    f_q, f_o, f_a = _font(card["font_title"]), _font(card["font_opt"]), _font(card["font_ans"])
    pad, margin = card["pad_px"], card["margin_px"]
    box_w = int(W * card["width_frac"])
    line_q = f_q.getbbox("Ag")[3] + 8
    line_o = f_o.getbbox("Ag")[3] + 8
    line_a = f_a.getbbox("Ag")[3] + 10
    n_ans = 3 if reveal else 1  # GROUND TRUTH always; +FROZEN +OURS on reveal
    # WRAP the question. It was drawn as one d.text() line, so any question longer than
    # box_w ran off the card and was clipped mid-word ("...(ignoring cam") — the card looked
    # broken, which is fatal for a research page. Wrap to the box and grow the box to match.
    q_lines, cur = [], ""
    for word in question.split():
        trial = f"{cur} {word}".strip()
        if d.textlength(trial, font=f_q) <= box_w - 2 * pad or not cur:
            cur = trial
        else:
            q_lines.append(cur); cur = word
    if cur:
        q_lines.append(cur)
    box_h = pad * 2 + len(q_lines) * line_q + len(opts) * line_o + 8 + n_ans * line_a
    x0, y0 = margin, margin
    d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=14,
                        fill=(12, 14, 18, card["bg_alpha"]))
    x, y = x0 + pad, y0 + pad
    for ql in q_lines:
        d.text((x, y), ql, font=f_q, fill=tuple(col["text"])); y += line_q
    for i, opt in enumerate(opts):
        is_gt = (order[i] == hero["true"])
        c = tuple(col["gt"]) if is_gt else tuple(col["text"])
        d.text((x, y), f"{_LETTERS[i]}) {opt}", font=f_o, fill=c); y += line_o
    y += 8
    gt_letter = _LETTERS[order.index(hero["true"])]
    d.text((x, y), f"GROUND TRUTH: {gt_letter}  ({hero['true']})", font=f_a, fill=tuple(col["gt"])); y += line_a
    if reveal:
        fz = _LETTERS[order.index(hero["frozen"])]
        ou = _LETTERS[order.index(hero["ours"])]
        d.text((x, y), f"FROZEN ANSWER:  {fz}   ✗", font=f_a, fill=tuple(col["wrong"])); y += line_a
        d.text((x, y), f"OURS   ANSWER:  {ou}   ✓", font=f_a, fill=tuple(col["correct"]))
    # Footer band (bottom): aggregate + probe disclosure, on EVERY frame — auto-fit to width.
    # OPTIONAL: a heroes doc with no "aggregate" renders no footer, so the card asserts only
    # what is true of THIS clip (its GT and the two real model answers) and makes no population
    # claim at all. Printing a favourable aggregate for a hand-picked clip would be the
    # dishonest option; printing none claims nothing.
    if agg is None:
        return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))
    foot = (f"OURS {agg['ours']*100:.1f}% vs FROZEN {agg['frozen']*100:.1f}% over {agg['n']} clips"
            f"   ·   selected example   ·   answer = attentive probe head (identical for both arms)")
    fsz = card["font_foot"]; f_foot = _font(fsz)
    while d.textlength(foot, font=f_foot) > W - 20 and fsz > 9:
        fsz -= 1; f_foot = _font(fsz)
    fb = d.textbbox((0, 0), foot, font=f_foot)
    d.rectangle([0, H - (fb[3] - fb[1]) - 12, W, H], fill=(12, 14, 18, 205))
    d.text((10, H - (fb[3] - fb[1]) - 6), foot, font=f_foot, fill=tuple(col["text"]))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def _title_frame(wh, text_lines, card, dur_frames):
    W, H = wh
    im = Image.new("RGB", (W, H), (10, 12, 16))
    d = ImageDraw.Draw(im)
    fonts = [_font(card["font_title"] + 8), _font(card["font_opt"]), _font(card["font_foot"] + 4)]
    y = H // 2 - 60
    for line, fnt in zip(text_lines, fonts + [fonts[-1]] * 8):
        # Shrink-to-fit: these lines are centred, so an over-wide one is clipped at BOTH
        # ends (the closing card read "...cene motion?' ... over 1825 Wa"). Mirror the
        # footer's auto-fit instead of assuming the caller keeps captions short.
        sz = fnt.size
        while d.textlength(line, font=fnt) > W - 40 and sz > 10:
            sz -= 1; fnt = _font(sz)
        w = d.textbbox((0, 0), line, font=fnt)[2]
        d.text(((W - w) // 2, y), line, font=fnt, fill=(236, 239, 241))
        y += fnt.getbbox("Ag")[3] + 18
    return [np.asarray(im)] * dur_frames


def _contact_sheet(frames_dir, out_png, tiles):
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        sys.exit("FATAL: no frames to build contact sheet")
    r, c = tiles
    pick = np.linspace(0, len(pngs) - 1, r * c).round().astype(int)
    ims = [Image.open(pngs[i]).convert("RGB") for i in pick]
    w, h = ims[0].size
    scale = 480 / w
    tw, th = int(w * scale), int(h * scale)
    sheet = Image.new("RGB", (tw * c, th * r), (0, 0, 0))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((tw, th)), ((i % c) * tw, (i // c) * th))
    sheet.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heroes", required=True, help="heroes.json from hero_extract (question, aggregate, heroes[])")
    ap.add_argument("--config", required=True, help="configs/demo.yaml (m17 block)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sanity", action="store_true", help="render only 2 clips for a fast smoke")
    ap.add_argument("--n-clips", type=int, default=None,
                    help="override m17.n_clips (e.g. 1 for a single research-page card)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["m17"]
    hero_doc = json.loads(Path(args.heroes).read_text())
    heroes = hero_doc["heroes"]
    ag = hero_doc.get("aggregate")
    # The denominator MUST come from the same document as the accuracies it describes. Taking
    # `n` from the yaml while ours/frozen/chance come from heroes.json lets the footer advertise
    # a stale N for a freshly-regenerated aggregate (silently wrong, and it is the one number a
    # reader uses to judge the claim). heroes.json wins; yaml is only the legacy fallback.
    # A doc with NO aggregate renders a clean card with no footer and no closing card.
    agg = None if ag is None else {"ours": ag["ours"], "frozen": ag["frozen"], "chance": ag["chance"],
                                   "n": ag.get("n", cfg["n_probe_clips"])}
    q = hero_doc.get("question")
    if q not in cfg["questions"]:
        sys.exit(f"FATAL: no display config for question {q!r}; add m17.questions.{q} to the demo config")
    qcfg = cfg["questions"][q]
    order = _class_order(qcfg)

    n_clips = 2 if args.sanity else (args.n_clips if args.n_clips is not None else cfg["n_clips"])
    sel = _rank_and_select(heroes, n_clips, cfg["max_per_video"], order)
    if len(sel) < n_clips:
        sys.exit(f"FATAL: only {len(sel)} hero clips available, need {n_clips}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "_frames"
    frames_dir.mkdir(exist_ok=True)
    for p in frames_dir.glob("*.png"):  # clean prior frames (truncate-safe, no rm of durable data)
        p.unlink()
    card, opts, question = cfg["card"], qcfg["options"], qcfg["text"]
    fps, spc = cfg["display_fps"], cfg["seconds_per_clip"]
    n_disp = int(round(spc * fps))
    reveal_at = int(round(cfg["reveal_delay_frac"] * n_disp))

    fi = 0
    wh = None
    t0 = time.time()
    for hi, h in enumerate(sel):
        frames = decode_all_frames(Path(h["path"]))  # (N,H,W,3) uint8 native
        frames = _resample(frames, n_disp)
        if wh is None:
            wh = (frames.shape[2], frames.shape[1])
            if agg is not None:   # no aggregate → clip-only card, skip the intro title
                for tfr in _title_frame(wh, ["Which model reads the scene right?",
                                             "same question -> FROZEN vs OURS (surgery)",
                                             "answer = attentive probe head (identical for both arms)"],
                                        card, int(fps * cfg["title_seconds"])):
                    Image.fromarray(tfr).save(frames_dir / f"{fi:06d}.png"); fi += 1
        for j, rgb in enumerate(frames):
            drawn = _draw_card(rgb, card, h, j >= reveal_at, agg, opts, question, order)
            Image.fromarray(drawn).save(frames_dir / f"{fi:06d}.png"); fi += 1
        print(f"[m17] clip {hi+1}/{len(sel)} {Path(h['path']).name}  GT={h['true']} "
              f"FROZEN={h['frozen']} OURS={h['ours']}  ({fi} frames, {(time.time()-t0):.0f}s)")

    # closing aggregate card — skipped when the doc carries no aggregate
    if agg is not None:
        for tfr in _title_frame(wh, [f"OURS {agg['ours']*100:.1f}%   vs   FROZEN {agg['frozen']*100:.1f}%",
                                     f"{qcfg['short']} over {agg['n']} DenseWorld clips",
                                     f"chance {agg['chance']*100:.1f}%  ·  leakage-safe (video-held-out) probe"],
                                card, int(fps * cfg["title_seconds"])):
            Image.fromarray(tfr).save(frames_dir / f"{fi:06d}.png"); fi += 1

    mp4 = out / (f"demo_vqa_{q}_sanity.mp4" if args.sanity else f"demo_vqa_{q}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(frames_dir / "%06d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", str(cfg["crf"]), str(mp4)], check=True)
    _contact_sheet(frames_dir, out / "contact_sheet.png", cfg["sheet_tiles"])
    (out / "demo_vqa_metrics.json").write_text(json.dumps({
        "question": question, "options": opts, "aggregate": agg,
        "clips": [{"file": Path(h["path"]).name, "video_id": h["video_id"], "ground_truth": h["true"],
                   "frozen_answer": h["frozen"], "ours_answer": h["ours"]} for h in sel]}, indent=1))
    print(f"[m17] DONE -> {mp4} ({fi} frames @ {fps}fps = {fi/fps:.0f}s) · contact_sheet.png · demo_vqa_metrics.json")


if __name__ == "__main__":
    main()
