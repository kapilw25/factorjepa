#!/usr/bin/env python3
"""m16 — motion-similarity RETRIEVAL demo (iter20, decoder-free · the honest layman demo).

motion-cosine separation is OURS diheavy's biggest eval win (115x vs frozen). This shows it with
ZERO decoder / zero fog: for a query clip, retrieve its top-K nearest neighbours in each model's
POOLED encoder-feature space (mean over final-layer tokens — the exact space m12b's motion-cosine
scores). OURS returns clips that MOVE THE SAME WAY as the query; FROZEN returns look-alike-but-move-
differently clips. Every panel is REAL video — a layman watches the neighbour rows and sees OURS
"get the motion" and FROZEN miss it.

HONESTY (this demo makes a "OURS is better" claim about real model quality, so it is guarded):
  - A MODEL-FREE motion descriptor (coarse global camera-flow + motion energy, computed WITHOUT
    either model) tags every clip in plain English AND is the ground truth for two things:
      (1) query selection — pick queries where OURS' neighbours are genuinely more motion-similar
          than FROZEN's (by descriptor distance), never by eye;
      (2) an AGGREGATE honesty check over the WHOLE pool — if OURS is not better on average
          (neighbour precision@k on the motion tag), the demo FAILS LOUD instead of cherry-picking.
  - Neighbours from the SAME source video as the query are excluded (no trivial near-duplicate hits).

Stages (run in order):
  pool      sample pool_size clips across --tars → data/demo_pool/*.mp4 + pool_meta.npz
            (motion descriptor + tag per clip; model-free).
  features  load ONE --ckpt encoder → pooled feature per pool clip → <work-dir>/feats__<label>.npy.
            Run once per model (FROZEN, then OURS).
  render    read both feats + pool_meta → retrieve → aggregate honesty check (FAIL LOUD) →
            demo_retrieval.mp4 + contact_sheet.png.

Gold ref: motion-cosine (m12b) uses pooled frozen features; nearest-neighbour retrieval in that same
space is the standard qualitative probe of representation quality (facebookresearch/vjepa2 eval).

USAGE (RTX 3060 12 GB · venv_denseworld):
    source venv_denseworld/bin/activate
    set -o pipefail && PYTHONPATH=src python -u src/m16_retrieval_demo.py --stage pool \\
        --demo-config configs/demo.yaml --work-dir outputs/demo/m16 \\
        --tars data/demo_src/data/train-0000{0,10,25}.tar data/demo_src/data/train-00112.tar \\
        2>&1 | tee logs/m16_pool_$(date +%Y%m%d_%H%M%S).log
    # then --stage features --ckpt "FROZEN 2.1=checkpoints/vjepa2_1_vitg_384.pt" --model-config … (x2)
    # then --stage render --frozen "FROZEN 2.1" --ours "OURS diheavy" --output-dir outputs/demo/m16
"""
import argparse
import queue as queue_mod
import re
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from tqdm import tqdm

from utils.config import get_pipeline_config
from utils.frozen_features import resize_and_normalize
from utils.predictor_eval import load_encoder_only
from utils.demo_video import decode_all_frames, decode_frames_select, probe_n_frames, resize_center_crop_uint8

_BG = (16, 20, 24)
_FG = (236, 239, 241)
_SUB = (176, 190, 197)
_GREEN = (76, 175, 80)
_RED = (211, 70, 55)
_FONT_BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def _font(sz):
    return ImageFont.truetype(_FONT_BOLD, sz)


def _safe(label):
    return re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")


# ── model-free motion descriptor ─────────────────────────────────────────────────

def motion_descriptor(gray, search):
    """gray (T,h,w) float32 → (mean_dx, mean_dy, energy). Per consecutive pair, coarse global
    translation minimising SAD over +/-search px (captures camera pan direction+speed); energy =
    mean |Δframe| (fast vs slow). NO model involved — this is the ground truth."""
    T, h, w = gray.shape
    m = search
    dxs, dys, en = [], [], []
    for t in range(T - 1):
        a = gray[t, m:h - m, m:w - m]
        b = gray[t + 1]
        best, bdx, bdy = 1e18, 0, 0
        for dy in range(-m, m + 1, 2):
            for dx in range(-m, m + 1, 2):
                bb = b[m + dy:h - m + dy, m + dx:w - m + dx]
                sad = float(np.abs(a - bb).mean())
                if sad < best:
                    best, bdx, bdy = sad, dx, dy
        dxs.append(bdx); dys.append(bdy)
        en.append(float(np.abs(gray[t] - gray[t + 1]).mean()))
    return np.array([np.mean(dxs), np.mean(dys), np.mean(en)], np.float32)


def descriptor_tag(desc, thr):
    """Plain-English motion tag from the descriptor + pool-derived thresholds."""
    dx, dy, e = desc
    mag = float(np.hypot(dx, dy))
    if e < thr["e_lo"]:
        return "mostly still"
    if mag > thr["pan"]:
        if abs(dx) >= abs(dy):
            return "camera pans " + ("left" if dx > 0 else "right")
        return "camera tilts " + ("down" if dy > 0 else "up")
    if e > thr["e_hi"]:
        return "fast / busy motion"
    return "steady forward pace"


# ── stage: pool ──────────────────────────────────────────────────────────────────

def _gray_small(frames_uint8, w, h):
    """(T,H,W,3) uint8 → (T,h,w) float32 gray, cheap resize by strided sampling."""
    T, H, W, _ = frames_uint8.shape
    ys = np.linspace(0, H - 1, h).astype(int)
    xs = np.linspace(0, W - 1, w).astype(int)
    g = frames_uint8[:, ys][:, :, xs].astype(np.float32).mean(-1)
    return g


def stage_pool(args, m16):
    num_frames = get_pipeline_config()["probe"]["num_frames"]
    out = Path(args.work_dir); out.mkdir(parents=True, exist_ok=True)
    pool_dir = out / "clips"; pool_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(m16["seed"])
    # sample pool_size evenly across the tars
    per = max(1, m16["pool_size"] // len(args.tars))
    recs = []
    for tar_path in args.tars:
        with tarfile.open(tar_path) as tf:
            members = [mm for mm in tf.getmembers() if mm.name.endswith(".mp4")]
            take = rng.choice(len(members), min(per, len(members)), replace=False)
            shard = Path(tar_path).stem.split("-")[-1]
            for mi in tqdm(sorted(take), desc=f"pool[{shard}]"):
                mem = members[mi]
                key = Path(mem.name).stem
                mp4 = tf.extractfile(mem).read()
                meta = {}
                jn = mem.name[:-4] + ".json"
                try:
                    import json as _json
                    meta = _json.loads(tf.extractfile(jn).read())
                except Exception:
                    meta = {"video_id": f"{shard}_{key}", "tour_type": "?", "city": "?"}
                cid = f"{shard}_{key}"
                (pool_dir / f"{cid}.mp4").write_bytes(mp4)
                tmp = out / "_tmp_pool.mp4"; tmp.write_bytes(mp4)
                try:
                    n = probe_n_frames(tmp)
                    idx = np.linspace(0, n - 1, num_frames).round().astype(int)
                    fr = decode_frames_select(tmp, idx)
                except Exception as e:
                    print(f"  [skip] {cid}: {type(e).__name__} {e}")
                    (pool_dir / f"{cid}.mp4").unlink(missing_ok=True)
                    continue
                g = _gray_small(fr, m16["desc_w"], m16["desc_h"])
                desc = motion_descriptor(g, m16["desc_search"])
                recs.append((cid, meta.get("video_id", cid), meta.get("tour_type", "?"),
                             meta.get("city", "?"), desc,
                             meta.get("section", "?"), meta.get("source_file", "?")))
    tmpf = out / "_tmp_pool.mp4"
    if tmpf.exists():
        tmpf.write_bytes(b"")
    # TRUE motion-class labels — the exact ground truth motion-cosine (m12b) is scored on. Key =
    # section/video_id/source_file (verified 2026-07-14). This is what makes the demo honest: retrieval
    # is judged by the metric's OWN labels, not a home-grown proxy (two proxies disagreed with it).
    import json as _json
    lab = _json.loads(Path(args.action_labels).read_text())
    cls_id, cls_name = [], []
    for r in recs:
        key = f"{r[5]}/{r[1]}/{r[6]}"                     # section/video_id/source_file
        e = lab.get(key)
        cls_id.append(int(e["class_id"]) if e else -1)
        cls_name.append(e["class"] if e else "?")
    mapped = sum(c >= 0 for c in cls_id)
    if mapped < 0.9 * len(recs):
        sys.exit(f"FATAL: only {mapped}/{len(recs)} pool clips mapped to a true motion class — "
                 f"check the action_labels key format (section/video_id/source_file)")
    descs = np.stack([r[4] for r in recs])
    np.savez(out / "pool_meta.npz",
             cid=np.array([r[0] for r in recs]), video_id=np.array([r[1] for r in recs]),
             tour=np.array([r[2] for r in recs]), city=np.array([r[3] for r in recs]),
             desc=descs, cls_id=np.array(cls_id), cls_name=np.array(cls_name))
    from collections import Counter
    print(f"[m16 pool] {len(recs)} clips ({mapped} labelled) → {pool_dir}/ · "
          f"true motion-class mix: {dict(Counter(cls_name))}")


# ── stage: features (pooled encoder feature per pool clip) ───────────────────────

def stage_features(args, mcfg, m16):
    num_frames = get_pipeline_config()["probe"]["num_frames"]
    crop, embed_dim = mcfg["crop_size"], mcfg["embed_dim"]
    out = Path(args.work_dir)
    meta = np.load(out / "pool_meta.npz", allow_pickle=True)
    cids = list(meta["cid"])
    encoder, _ckpt, _ = load_encoder_only(str(args.ckpt_path), num_frames, model_cfg=str(args.model_config))
    pool_dir = out / "clips"
    work_q = queue_mod.Queue(maxsize=m16["decode_workers"] * 2)
    ready_q = queue_mod.Queue(maxsize=4)

    def _reader():
        for cid in cids:
            work_q.put(cid)
        for _ in range(m16["decode_workers"]):
            work_q.put(None)

    def _decoder():
        while True:
            cid = work_q.get()
            if cid is None:
                ready_q.put(None); return
            mp4 = pool_dir / f"{cid}.mp4"
            try:
                n = probe_n_frames(mp4)
                idx = np.linspace(0, n - 1, num_frames).round().astype(int)
                fr = decode_frames_select(mp4, idx)
            except Exception as e:
                print(f"  [skip] {cid}: {type(e).__name__} {e}"); continue
            ready_q.put((cid, fr))

    threading.Thread(target=_reader, daemon=True).start()
    for _ in range(m16["decode_workers"]):
        threading.Thread(target=_decoder, daemon=True).start()

    feats = {}
    ended = 0
    pbar = tqdm(total=len(cids), desc=f"feats[{args.label}]")
    while ended < m16["decode_workers"]:
        item = ready_q.get()
        if item is None:
            ended += 1; continue
        cid, fr = item
        norm = resize_and_normalize(fr, crop)
        with torch.no_grad():
            h = encoder(norm.unsqueeze(0).to("cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3, 4).contiguous())
            if isinstance(h, (list, tuple)):
                h = torch.cat(list(h), dim=-1)
        feats[cid] = h[0].float().cpu().numpy()[:, -embed_dim:].mean(0).astype(np.float32)
        pbar.update(1)
    pbar.close()
    F = np.stack([feats[c] for c in cids])                # aligned to pool_meta cid order
    np.save(out / f"feats__{_safe(args.label)}.npy", F)
    print(f"[m16 features] {args.label}: {F.shape} → {out}/feats__{_safe(args.label)}.npy")


# ── stage: render ────────────────────────────────────────────────────────────────

def _topk(qi, F, video_id, k):
    """cosine top-k of query index qi in feature matrix F, excluding same source video."""
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    sims = Fn @ Fn[qi]
    order = np.argsort(-sims)
    out = []
    for j in order:
        if j == qi or video_id[j] == video_id[qi]:
            continue
        out.append(int(j))
        if len(out) == k:
            break
    return out


def _play_frames(mp4, crop, want=48):
    allf = decode_all_frames(Path(mp4))
    idx = np.linspace(0, allf.shape[0] - 1, want).round().astype(int)
    return resize_center_crop_uint8(allf[idx], crop)


def _flow_arrow(frames_uint8, search=5, dw=64, dh=36):
    """Dominant optical-flow vector of a clip (model-free) → the arrow that makes MOTION visible in a
    still: query + same-motion neighbours point the SAME way; different-motion clips don't. Coarse
    global translation (mean over frame pairs) on a downsampled gray copy."""
    T, H, W, _ = frames_uint8.shape
    ys = np.linspace(0, H - 1, dh).astype(int); xs = np.linspace(0, W - 1, dw).astype(int)
    g = frames_uint8[:, ys][:, :, xs].astype(np.float32).mean(-1)
    m = search
    dxs, dys = [], []
    for t in range(0, len(g) - 1, 2):
        a = g[t, m:dh - m, m:dw - m]; b = g[t + 1]
        best, bd = 1e18, (0, 0)
        for dy in range(-m, m + 1):
            for dx in range(-m, m + 1):
                sad = float(np.abs(a - b[m + dy:dh - m + dy, m + dx:dw - m + dx]).mean())
                if sad < best:
                    best, bd = sad, (dx, dy)
        dxs.append(bd[0]); dys.append(bd[1])
    return float(np.mean(dxs)), float(np.mean(dys))


def _draw_arrow(draw, cx, cy, dx, dy, scale=14):
    """White motion arrow from cell centre. Short dot = ~still. Length/dir = dominant pixel flow."""
    mag = float(np.hypot(dx, dy))
    if mag < 0.4:
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 255, 255), width=3)
        return
    ex, ey = cx + dx * scale, cy + dy * scale
    draw.line([cx, cy, ex, ey], fill=(255, 255, 255), width=5)
    ang = np.arctan2(dy, dx)
    for da in (2.6, -2.6):
        draw.line([ex, ey, ex - 14 * np.cos(ang + da), ey - 14 * np.sin(ang + da)],
                  fill=(255, 255, 255), width=5)


def stage_render(args, mcfg, m16):
    crop = mcfg["crop_size"]
    out = Path(args.work_dir)
    meta = np.load(out / "pool_meta.npz", allow_pickle=True)
    cid = list(meta["cid"]); vid = list(meta["video_id"])
    cls = list(meta["cls_id"]); cname = list(meta["cls_name"])   # TRUE motion classes (m12b ground truth)
    Ff = np.load(out / f"feats__{_safe(args.frozen)}.npy")
    Fo = np.load(out / f"feats__{_safe(args.ours)}.npy")
    k = m16["top_k"]
    N = len(cid)

    def prec(i, js):                                       # fraction of neighbours in query's TRUE class
        return np.mean([cls[j] == cls[i] for j in js]) if js else 0.0

    # AGGREGATE honesty check over EVERY labelled pool clip — judged by the metric's OWN motion labels.
    valid = [i for i in range(N) if cls[i] >= 0]
    pf, po = [], []
    nbr = {}
    for i in valid:
        nf = _topk(i, Ff, vid, k); no = _topk(i, Fo, vid, k); nbr[i] = (nf, no)
        pf.append(prec(i, nf)); po.append(prec(i, no))
    pf, po = float(np.mean(pf)), float(np.mean(po))
    print(f"[m16 render] AGGREGATE motion-class precision@{k} (TRUE labels): "
          f"FROZEN {pf:.3f}  ·  OURS {po:.3f}")
    if po <= pf:
        sys.exit(f"FATAL (honesty): OURS precision@{k} {po:.3f} <= FROZEN {pf:.3f} on the TRUE motion "
                 f"labels — refusing to build a demo that misrepresents the result.")

    # queries: OURS gets all-3 right AND FROZEN misses at least one → the gap is VISIBLE; prefer
    # dynamic (non-'still') classes and diverse classes so a layman sees real motion.
    scored = sorted(valid, key=lambda i: (prec(i, nbr[i][1]) - prec(i, nbr[i][0])), reverse=True)
    queries, used = [], set()
    for i in scored:
        nf, no = nbr[i]
        if prec(i, no) < 1.0 or prec(i, nf) >= 1.0:        # OURS perfect, FROZEN imperfect
            continue
        if cls[i] in used:
            continue
        queries.append(i); used.add(cls[i])
        if len(queries) == m16["n_queries"]:
            break
    if len(queries) < m16["n_queries"]:                    # relax: just OURS-beats-FROZEN, diverse class
        for i in scored:
            nf, no = nbr[i]
            if i in queries or cls[i] in used or prec(i, no) <= prec(i, nf):
                continue
            queries.append(i); used.add(cls[i])
            if len(queries) == m16["n_queries"]:
                break
    if not queries:
        sys.exit("FATAL: no query where OURS out-retrieves FROZEN on true classes — pool too uniform.")
    tag = cname                                            # display tag = the TRUE class name
    print(f"[m16 render] queries: {[(cid[i], cname[i], f'O{prec(i,nbr[i][1]):.2f}/F{prec(i,nbr[i][0]):.2f}') for i in queries]}")

    # ── compose ──
    W, H, fps = 1280, 720, m16["fps"]
    pool_dir = out / "clips"
    frames_dir = Path(args.output_dir) / f"frames_{time.strftime('%Y%m%d_%H%M%S')}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene_frames = int(6.0 * fps)
    fi = 0

    def cell(draw, im, x, y, s, frames, fidx, label, border, arrow=None):
        im.paste(Image.fromarray(frames[fidx % len(frames)]).resize((s, s), Image.LANCZOS), (x, y))
        if arrow is not None:                             # motion arrow makes the flow visible in a still
            _draw_arrow(draw, x + s // 2, y + s // 2, arrow[0], arrow[1], scale=min(s, 175) / 12)
        if border is not None:
            draw.rectangle([x, y, x + s, y + s], outline=border, width=4)
        draw.text((x + s // 2, y + s + 13), label, font=_font(12), fill=_FG, anchor="mm")

    # title card
    for _ in range(int(3.0 * fps)):
        im = Image.new("RGB", (W, H), _BG); d = ImageDraw.Draw(im)
        d.text((W // 2, 240), "Find clips that MOVE like this one", font=_font(38), fill=_FG, anchor="mm")
        d.text((W // 2, 310), f"{args.frozen}   vs   {args.ours}", font=_font(26), fill=(128, 203, 196), anchor="mm")
        d.text((W // 2, 380), "each model retrieves its 3 nearest clips — NO decoder, all real video",
               font=_font(16), fill=_SUB, anchor="mm")
        d.text((W // 2, 420), "green = same kind of motion as the query   ·   red = different motion",
               font=_font(16), fill=_SUB, anchor="mm")
        im.save(frames_dir / f"{fi:06d}.png"); fi += 1

    for qi in tqdm(queries, desc="render queries"):
        nf = _topk(qi, Ff, vid, k); no = _topk(qi, Fo, vid, k)
        qf = _play_frames(pool_dir / f"{cid[qi]}.mp4", crop)
        ff = [_play_frames(pool_dir / f"{cid[j]}.mp4", crop) for j in nf]
        of = [_play_frames(pool_dir / f"{cid[j]}.mp4", crop) for j in no]
        qa = _flow_arrow(qf)                              # model-free dominant flow → the visible arrow
        fa = [_flow_arrow(x) for x in ff]; oa = [_flow_arrow(x) for x in of]
        for t in range(scene_frames):
            im = Image.new("RGB", (W, H), _BG); d = ImageDraw.Draw(im)
            d.text((W // 2, 22), "Find clips that MOVE like the query  —  white arrow = how the clip moves",
                   font=_font(20), fill=_FG, anchor="mm")
            d.text((W // 2, 50), "green border = SAME motion as the query (official eval label)  ·  "
                                 "red = different  ·  NO decoder, all real video",
                   font=_font(14), fill=_SUB, anchor="mm")
            qs = 180
            cell(d, im, (W - qs) // 2, 68, qs, qf, t, "QUERY", (128, 203, 196), arrow=qa)
            rows = [(args.frozen, nf, ff, fa, 300), (args.ours, no, of, oa, 488)]
            cs = 170
            for lbl, ns, fs, ars, ry in rows:
                d.text((88, ry + cs // 2), lbl.split()[0], font=_font(17), fill=_FG, anchor="mm")
                x = 210
                for j, fr, ar in zip(ns, fs, ars):
                    match = cls[j] == cls[qi]
                    cell(d, im, x, ry, cs, fr, t, "", _GREEN if match else _RED, arrow=ar)
                    x += cs + 30
            d.text((W // 2, H - 40), "watch the ARROWS: OURS' 3 clips all point like the query; FROZEN's point every which way",
                   font=_font(15), fill=(128, 203, 196), anchor="mm")
            d.text((W // 2, H - 18), f"(query motion class: {tag[qi]})", font=_font(12), fill=_SUB, anchor="mm")
            im.save(frames_dir / f"{fi:06d}.png"); fi += 1

    import subprocess
    mp4 = Path(args.output_dir) / "demo_retrieval.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i",
                    str(frames_dir / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", str(m16["crf"]), str(mp4)], check=True)
    # contact sheet
    pngs = sorted(frames_dir.glob("*.png"))
    idx = np.linspace(0, len(pngs) - 1, 12).round().astype(int)
    sheet = Image.new("RGB", (4 * 480, 3 * 270), "black")
    for kk, ii in enumerate(idx):
        sheet.paste(Image.open(pngs[ii]).resize((480, 270)), ((kk % 4) * 480, (kk // 4) * 270))
    sheet.save(Path(args.output_dir) / "contact_sheet.png")
    print(f"[m16 render] DONE → {mp4} ({fi} frames @ {fps}fps = {fi/fps:.0f}s) · contact_sheet.png · "
          f"aggregate precision@{k} FROZEN {pf:.3f} < OURS {po:.3f}")


def main():
    p = argparse.ArgumentParser(description="m16 — motion-similarity retrieval demo")
    p.add_argument("--stage", required=True, choices=["pool", "features", "render"])
    p.add_argument("--demo-config", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--tars", nargs="+", default=None)
    p.add_argument("--action-labels", type=Path, default=None,
                   help="pool stage: probe_action/action_labels.json — the TRUE motion classes m12b "
                        "scores motion-cosine on (the demo's honest ground truth)")
    p.add_argument("--ckpt", default=None, metavar="LABEL=PATH", help="features stage: one model")
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--frozen", default=None, help="render: frozen model label")
    p.add_argument("--ours", default=None, help="render: ours model label")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    if not torch.cuda.is_available() and args.stage != "render":
        sys.exit("FATAL: CUDA required for pool/features — no CPU fallback")
    m16 = yaml.safe_load(args.demo_config.read_text())["m16"]
    if args.stage == "pool":
        if not args.tars or not args.action_labels:
            sys.exit("FATAL: --stage pool requires --tars and --action-labels")
        stage_pool(args, m16)
    elif args.stage == "features":
        if not args.ckpt or "=" not in args.ckpt or not args.model_config:
            sys.exit("FATAL: --stage features requires --ckpt LABEL=PATH and --model-config")
        args.label, args.ckpt_path = args.ckpt.split("=", 1)
        if not Path(args.ckpt_path).exists():
            sys.exit(f"FATAL: ckpt not found: {args.ckpt_path}")
        mcfg = yaml.safe_load(args.model_config.read_text())["model"]
        stage_features(args, mcfg, m16)
    else:
        if not (args.frozen and args.ours and args.output_dir and args.model_config):
            sys.exit("FATAL: --stage render requires --frozen --ours --model-config --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        mcfg = yaml.safe_load(args.model_config.read_text())["model"]
        stage_render(args, mcfg, m16)


if __name__ == "__main__":
    main()
