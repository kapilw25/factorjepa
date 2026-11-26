#!/usr/bin/env python3
"""m18_vlm_train — LLaVA 2-stage trainer for VJepaLlavaVLM. IDENTICAL recipe both arms; the ONLY
difference is --arm {frozen,ours} (which encoder ckpt load_encoder_only reads). Runs on the 96GB box.

  stage align    : projector ONLY (encoder + LLM frozen), on stage-1 captions (data/vlm/align.jsonl).
  stage instruct : projector + LLM-LoRA, on stage-2 MOTION QA (data/vlm/instruct.jsonl); loads the
                   align projector first.

Loss = causal-LM over the ANSWER tokens only (video + prompt tokens are -100). Gold ref: LLaVA-1.5
2-stage visual instruction tuning (github.com/haotian-liu/LLaVA) + V-JEPA2 alignment (arxiv 2506.09985).
No hardcoded values — LR/epochs/BS/warmup from configs/vlm.yaml.

USAGE (96GB box · /venv/main):
    for ARM in frozen ours; do
      python src/m18_vlm_train.py --config configs/vlm.yaml --arm $ARM --stage align    --no-wandb
      python src/m18_vlm_train.py --config configs/vlm.yaml --arm $ARM --stage instruct --no-wandb
    done
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.vlm_model import VJepaLlavaVLM, build_chat
from utils.frozen_features import resize_and_normalize
from utils.demo_video import decode_all_frames
from utils.config import get_model_config


class VideoQADataset(Dataset):
    """JSONL rows {video, prompt(<video>…), answer} → (pixels (C,T,H,W), input_ids, labels).
    labels = -100 on the prompt, answer-token-ids on the answer (+eos). The single <video> token
    stays in input_ids; the model expands it to n_video_tokens at fuse time."""

    def __init__(self, jsonl, tokenizer, num_frames, crop, enable_thinking):
        self.rows = [json.loads(x) for x in open(jsonl)]
        if not self.rows:
            sys.exit(f"FATAL: empty dataset {jsonl}")
        self.tok, self.num_frames, self.crop, self.enable_thinking = tokenizer, num_frames, crop, enable_thinking

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        vp = Path(r["video"])
        if not vp.exists():
            raise FileNotFoundError(f"video missing: {vp} (run m18_vlm_data --download-videos)")
        frames = decode_all_frames(vp)
        idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        pixels = resize_and_normalize(frames[idx], self.crop).permute(1, 0, 2, 3).contiguous()  # (C,T,H,W)
        ids, labels = build_chat(self.tok, r["prompt"], r["answer"], self.enable_thinking)   # chat-templated, non-thinking
        return pixels, torch.tensor(ids), torch.tensor(labels)


def collate(batch, pad_id):
    pixels = torch.stack([b[0] for b in batch])
    L = max(b[1].shape[0] for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b[1].shape[0]
        ids[i, :n], lab[i, :n], attn[i, :n] = b[1], b[2], 1
    return pixels, ids, attn, lab


def train(cfg, arm, stage, no_wandb, max_samples, epochs=None, lr=None):
    v = cfg["vlm"]
    st = dict(v["stages"][stage])                    # copy so CLI overrides don't mutate the loaded cfg
    # EARLY-GATE overrides: that path skips `align`, so the projector starts RANDOM and the instruct LR
    # (2e-5, tuned for LoRA fine-tuning AFTER align) is too low to learn it. --lr/--epochs let the gate
    # train a usable projector cheaply. BOTH arms must use the SAME overrides (fair encoder swap).
    if epochs:
        st["epochs"] = epochs
    if lr:
        st["lr"] = lr
    out = Path(v["render"]["out_dir"]).parent / "vlm_ckpt" / arm       # outputs/demo/vlm_ckpt/<arm>
    out.mkdir(parents=True, exist_ok=True)
    model = VJepaLlavaVLM(cfg, arm, load_llm=True, lora=(stage == "instruct"))

    if stage == "instruct":                                            # resume the aligned projector if present
        ap = out / "align_projector.pt"
        if ap.exists():
            model.projector.load_state_dict(torch.load(ap, map_location="cuda"))
            print(f"[m18 train] loaded aligned projector {ap}")
        else:
            print(f"[m18 train] WARN {ap} missing — projector from scratch (OK for the EARLY gate; "
                  f"run --stage align first for the FULL run)")

    jsonl = Path(v["data"]["out_dir"]) / ("align.jsonl" if stage == "align" else "instruct.jsonl")
    crop = get_model_config(v["encoder"]["model_config"])["model"]["crop_size"]   # single source (384)
    ds = VideoQADataset(jsonl, model.tokenizer, v["encoder"]["num_frames"], crop, model.enable_thinking)
    if max_samples:                                                               # EARLY-gate cheap cap
        ds.rows = ds.rows[:max_samples]
        print(f"[m18 train] capped to {len(ds)} samples (--max-samples)")
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    dl = DataLoader(ds, batch_size=st["batch_size"], shuffle=True, num_workers=4,
                    collate_fn=lambda b: collate(b, pad_id), drop_last=True)

    params = model.trainable_parameters(stage)
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=st["lr"], weight_decay=st["weight_decay"])
    steps = st["epochs"] * math.ceil(len(dl) / st["grad_accum"])
    warm = max(1, int(steps * st["warmup_frac"]))

    def lr_at(s):                                                      # linear warmup → cosine decay
        if s < warm:
            return s / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    run = None
    if not no_wandb:
        from utils.wandb_utils import init_wandb
        run = init_wandb(project="factorjepa-vlm", name=f"{arm}_{stage}", config={"arm": arm, "stage": stage, **st})

    print(f"[m18 train] arm={arm} stage={stage} · {len(ds)} ex · {len(dl)} batches · "
          f"{steps} opt-steps · trainable {n_train/1e6:.1f}M")
    gstep = 0
    for ep in range(st["epochs"]):
        model.projector.train()
        pbar = tqdm(dl, desc=f"{arm}/{stage} ep{ep}")
        opt.zero_grad()
        for i, (pixels, ids, attn, lab) in enumerate(pbar):
            pixels = pixels.to("cuda", torch.bfloat16)
            ids, attn, lab = ids.to("cuda"), attn.to("cuda"), lab.to("cuda")
            loss = model(pixels, ids, attn, lab).loss / st["grad_accum"]
            loss.backward()
            if (i + 1) % st["grad_accum"] == 0:
                for g in opt.param_groups:
                    g["lr"] = st["lr"] * lr_at(gstep)
                opt.step(); opt.zero_grad(); gstep += 1
            l = loss.item() * st["grad_accum"]
            pbar.set_postfix(loss=f"{l:.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
            if run:
                run.log({"loss": l, "lr": opt.param_groups[0]["lr"]})

    torch.save(model.projector.state_dict(), out / ("align_projector.pt" if stage == "align" else "instruct_projector.pt"))
    if stage == "instruct":
        model.llm.save_pretrained(out / "lora")                       # peft LoRA adapter
    print(f"[m18 train] DONE arm={arm} stage={stage} → {out}")
    if run:
        run.finish()


def main():
    p = argparse.ArgumentParser(description="m18 — VLM 2-stage trainer")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--arm", required=True, choices=["frozen", "ours"])
    p.add_argument("--stage", required=True, choices=["align", "instruct"])
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--max-samples", type=int, default=None, help="cap training samples (EARLY-gate cheap pass)")
    p.add_argument("--epochs", type=int, default=None, help="override cfg epochs (EARLY gate)")
    p.add_argument("--lr", type=float, default=None, help="override cfg lr (EARLY gate: projector is from scratch)")
    args = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("FATAL: CUDA required (96GB box) — no CPU fallback")
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg, args.arm, args.stage, args.no_wandb, args.max_samples, args.epochs, args.lr)


if __name__ == "__main__":
    main()
