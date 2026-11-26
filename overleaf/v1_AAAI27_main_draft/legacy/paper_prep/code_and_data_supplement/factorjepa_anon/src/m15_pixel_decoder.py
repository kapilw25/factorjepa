#!/usr/bin/env python3
"""m15 — tubelet-inversion PIXEL DECODER for V-JEPA latents (iter20 W1+W2, plan_v2 §2 WOW panel).

Trains a small per-token MLP that inverts a model's FINAL-LAYER encoder token (1408-dim) back into
its OWN tubelet's pixels (tubelet = 2 frames x 16x16 px x RGB = 1536 values). Classic feature
inversion — the honest, 12 GB-friendly member of the latent→pixel family (the V-JEPA paper's own
protocol uses a conditional diffusion decoder instead; see plan_pixel_generation.md for that path).

At DEMO time the decoder is fed the PREDICTOR's outputs for hidden tubelets (the last-1408 slice of
the concat prediction = the predictor's estimate of the final encoder layer) → each model's
"imagined future" in real pixels. NON-NEGOTIABLE caption wherever those pixels appear:
    "pixels from an EXTRA decoder trained by us — V-JEPA predicts descriptions, not pictures"

Stages (run in order; each resumable by re-running):
  precompute    frozen encoder over local denseworld tar shards → fp16 token/pixel pairs
                (tokens_per_clip random tokens per clip, ~6 GB for 2000 clips) + viz_clips clips
                stored with ALL tokens for the decode-sanity gate.
  train         MLP 1408→hidden→1536, L1 pixel loss; prints train/val L1 per epoch.
  decode-sanity EVAL GATE (do NOT feed predictor latents before this passes): decode REAL latents
                of the viz clips → side-by-side real | decoded PNG grid. Frames must be
                recognizable (blurry is expected and honest).

Per-model: the decoder inverts ONE model's feature space — train one per --ckpt (FROZEN first,
then OURS diheavy) into separate --work-dir trees.

Gold refs: feature-inversion (Mahendran & Vedaldi 2015) · V-JEPA decoder appendix
(https://arxiv.org/abs/2404.08471) · facebookresearch/vjepa2.

USAGE (RTX 3060 12 GB · venv_denseworld):
    source venv_denseworld/bin/activate
    set -o pipefail && PYTHONPATH=src python -u src/m15_pixel_decoder.py --stage precompute \\
        --ckpt checkpoints/vjepa2_1_vitg_384.pt \\
        --model-config configs/model/vjepa2_1_vitg.yaml --demo-config configs/demo.yaml \\
        --tars data/demo_src/data/train-00000.tar data/demo_src/data/train-00025.tar \\
        --work-dir outputs/demo/m15_frozen \\
        2>&1 | tee logs/m15_precompute_$(date +%Y%m%d_%H%M%S).log
    …then --stage train, then --stage decode-sanity (same --work-dir).
"""
import argparse
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.config import get_pipeline_config
from utils.frozen_features import resize_and_normalize
from utils.predictor_eval import load_encoder_only, token_grid
from utils.demo_video import TokenDecoder, assemble_frames, tubelet_targets


# ── stage: precompute ────────────────────────────────────────────────────────────

def stage_precompute(args, mcfg, m15):
    """Producer/consumer (/gpu-bottleneck 2026-07-14 — serial decode left the GPU at ~50% duty):
    a READER thread walks the tars (tarfile is not thread-safe → single reader), DECODE workers
    select-decode ONLY the 16 sampled frames (pixel-identical to full-decode+index, measured
    1565→516 ms and proven np.array_equal), the MAIN thread keeps every tensor op (resize/normalize
    + forward — per the src/CLAUDE.md threading rule) and consumes a bounded ready queue so decode
    overlaps the GPU forward. Measured serial rate 2.11 s/clip → expected ~1.0 s/clip."""
    import queue as queue_mod
    import threading
    num_frames = get_pipeline_config()["probe"]["num_frames"]
    crop, patch, tube = mcfg["crop_size"], mcfg["patch_size"], mcfg["tubelet_size"]
    embed_dim = mcfg["embed_dim"]
    encoder, _ckpt, _ = load_encoder_only(str(args.ckpt), num_frames, model_cfg=str(args.model_config))
    Tp, Hp, Wp, S = token_grid(num_frames)
    n_tok = Tp * S
    rng = np.random.default_rng(m15["seed"])
    out = Path(args.work_dir); out.mkdir(parents=True, exist_ok=True)
    n_workers = m15["decode_workers"]
    work_q = queue_mod.Queue(maxsize=n_workers * 2)          # (name, mp4_bytes) — bounded: caps RAM
    ready_q = queue_mod.Queue(maxsize=4)                     # (name, frames16) — decode runs ahead

    def _reader():
        sent = 0
        for tar_path in args.tars:
            if sent >= m15["max_clips"]:
                break
            with tarfile.open(tar_path) as tf:
                for member in tf:
                    if sent >= m15["max_clips"]:
                        break
                    if not member.name.endswith(".mp4"):
                        continue
                    work_q.put((member.name, tf.extractfile(member).read()))
                    sent += 1
        for _ in range(n_workers):
            work_q.put(None)

    def _decoder(wid):
        from utils.demo_video import decode_frames_select, probe_n_frames
        tmp = out / f"_tmp_clip_w{wid}.mp4"                  # per-worker tmp — no cross-thread clobber
        while True:
            item = work_q.get()
            if item is None:
                ready_q.put(None)
                return
            name, mp4_bytes = item
            tmp.write_bytes(mp4_bytes)
            try:
                n = probe_n_frames(tmp)
                idx = np.linspace(0, n - 1, num_frames).round().astype(int)   # SAME sampling as before
                frames16 = decode_frames_select(tmp, idx)
            except Exception as e:                           # corrupt member → skip LOUDLY, keep going
                print(f"  [skip] {name}: {type(e).__name__} {e}")
                continue
            ready_q.put((name, frames16))

    threading.Thread(target=_reader, daemon=True).start()
    for wid in range(n_workers):
        threading.Thread(target=_decoder, args=(wid,), daemon=True).start()

    feats, pix, viz = [], [], []
    n_done, ended, t0 = 0, 0, time.time()
    pbar = tqdm(total=m15["max_clips"], desc="precompute")
    while ended < n_workers:
        item = ready_q.get()
        if item is None:
            ended += 1
            continue
        name, frames16 = item
        norm = resize_and_normalize(frames16, crop)                       # (T,3,crop,crop) — main thread
        with torch.no_grad():
            h = encoder(norm.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
                        .permute(0, 2, 1, 3, 4).contiguous())
            if isinstance(h, (list, tuple)):
                h = torch.cat(list(h), dim=-1)
        z = h[0].float().cpu().numpy()[:, -embed_dim:].astype(np.float16)  # (n_tok, embed_dim)
        tgt = tubelet_targets(norm, num_frames, crop, patch, tube).numpy().astype(np.float16)
        if len(viz) < m15["viz_clips"]:                      # full-token clips for the decode-sanity gate
            viz.append((name, z, tgt))
        else:
            sel = rng.choice(n_tok, m15["tokens_per_clip"], replace=False)
            feats.append(z[sel]); pix.append(tgt[sel])
        n_done += 1
        pbar.update(1)
    pbar.close()
    for wid in range(n_workers):
        tmpf = out / f"_tmp_clip_w{wid}.mp4"
        if tmpf.exists():
            tmpf.write_bytes(b"")                            # truncate, never rm (delete-protection)
    np.savez(out / "train_tokens.npz", feats=np.concatenate(feats), pix=np.concatenate(pix))
    np.savez(out / "viz_clips.npz",
             names=np.array([v[0] for v in viz]),
             feats=np.stack([v[1] for v in viz]), pix=np.stack([v[2] for v in viz]))
    print(f"[m15 precompute] {n_done} clips → {out}/train_tokens.npz "
          f"({np.concatenate(feats).nbytes / 1e9:.1f} GB feats) + viz_clips.npz "
          f"({len(viz)} full clips) in {(time.time() - t0) / 60:.0f} min")


# ── stage: train ─────────────────────────────────────────────────────────────────

def stage_train(args, mcfg, m15):
    out = Path(args.work_dir)
    dat = np.load(out / "train_tokens.npz")
    feats = torch.from_numpy(dat["feats"].astype(np.float32))
    pix = torch.from_numpy(dat["pix"].astype(np.float32))
    n = feats.shape[0]
    g = torch.Generator().manual_seed(m15["seed"])
    perm = torch.randperm(n, generator=g)
    n_val = int(n * m15["val_frac"])
    val_i, tr_i = perm[:n_val], perm[n_val:]
    dec = TokenDecoder(feats.shape[1], m15["hidden"], pix.shape[1]).cuda()
    opt = torch.optim.AdamW(dec.parameters(), lr=m15["lr"], weight_decay=m15["weight_decay"])
    bt = m15["batch_tokens"]
    print(f"[m15 train] {n:,} token pairs ({n - n_val:,} train / {n_val:,} val) · "
          f"decoder {sum(p.numel() for p in dec.parameters()) / 1e6:.1f}M params")
    for ep in range(m15["epochs"]):
        dec.train()
        ep_perm = tr_i[torch.randperm(tr_i.shape[0], generator=g)]
        tot, nb = 0.0, 0
        for i in tqdm(range(0, ep_perm.shape[0], bt), desc=f"epoch {ep + 1}/{m15['epochs']}"):
            b = ep_perm[i:i + bt]
            loss = (dec(feats[b].cuda()) - pix[b].cuda()).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        dec.eval()
        with torch.no_grad():
            vl = float(np.mean([(dec(feats[val_i[i:i + bt]].cuda()) - pix[val_i[i:i + bt]].cuda())
                                .abs().mean().item() for i in range(0, n_val, bt)]))
        print(f"[m15 train] epoch {ep + 1}: train L1 {tot / nb:.4f} · val L1 {vl:.4f}")
    torch.save({"state_dict": dec.state_dict(), "in_dim": feats.shape[1],
                "hidden": m15["hidden"], "out_dim": pix.shape[1]}, out / "decoder.pt")
    print(f"[m15 train] saved → {out}/decoder.pt")


# ── stage: decode-sanity (the EVAL GATE) ─────────────────────────────────────────

def stage_decode_sanity(args, mcfg, m15):
    from PIL import Image
    num_frames = get_pipeline_config()["probe"]["num_frames"]
    crop, patch, tube = mcfg["crop_size"], mcfg["patch_size"], mcfg["tubelet_size"]
    from utils.demo_video import load_token_decoder
    out = Path(args.work_dir)
    dec = load_token_decoder(out / "decoder.pt")
    viz = np.load(out / "viz_clips.npz")
    rows = []
    for ci in tqdm(range(viz["feats"].shape[0]), desc="decode-sanity", unit="clip"):
        z = torch.from_numpy(viz["feats"][ci].astype(np.float32)).cuda()
        with torch.no_grad():
            rec = dec(z).cpu().numpy()
        real = assemble_frames(viz["pix"][ci].astype(np.float32), num_frames // tube, crop, patch, tube)
        fake = assemble_frames(rec, num_frames // tube, crop, patch, tube)
        mid = real.shape[0] // 2
        rows.append(np.concatenate([real[mid], fake[mid]], axis=1))     # real | decoded
    grid = np.concatenate(rows[:8], axis=0)
    png = out / "decode_sanity_real_vs_decoded.png"
    Image.fromarray(grid).save(png)
    l1 = float(np.mean(np.abs(viz["pix"].astype(np.float32) -
                              np.stack([dec(torch.from_numpy(viz["feats"][i].astype(np.float32)).cuda())
                                        .detach().cpu().numpy() for i in range(viz["feats"].shape[0])]))))
    print(f"[m15 decode-sanity] pixel L1 on viz clips = {l1:.4f} → {png}")
    print("[m15 decode-sanity] GATE: eyeball the PNG — decoded column must be RECOGNIZABLE "
          "(blurry is expected). Only then feed PREDICTOR latents (W3).")


def main():
    p = argparse.ArgumentParser(description="m15 — tubelet-inversion pixel decoder (W1+W2)")
    p.add_argument("--stage", required=True, choices=["precompute", "train", "decode-sanity"])
    p.add_argument("--ckpt", type=Path, required=True, help="V-JEPA .pt whose features to invert")
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--demo-config", type=Path, required=True)
    p.add_argument("--tars", nargs="+", default=None, help="webdataset tar shards (precompute)")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="per-model tree, e.g. outputs/demo/m15_frozen")
    args = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("FATAL: CUDA required — no CPU fallback")
    mcfg = yaml.safe_load(args.model_config.read_text())["model"]
    m15 = yaml.safe_load(args.demo_config.read_text())["m15"]
    if args.stage == "precompute":
        if not args.tars:
            sys.exit("FATAL: --stage precompute requires --tars")
        stage_precompute(args, mcfg, m15)
    elif args.stage == "train":
        stage_train(args, mcfg, m15)
    else:
        stage_decode_sanity(args, mcfg, m15)


if __name__ == "__main__":
    main()
