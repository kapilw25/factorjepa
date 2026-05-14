# 🎨 iter15 — HTML Refactor: Retrieval → Motion-Class

> 📅 **Date:** 2026-05-14
> 🎯 **Goal:** pivot `docs/index.html` from spatial scene-retrieval (kNN over 16-axis taxonomy → Prec@K / mAP@K / Cycle@K) to motion-class probing (top-1 accuracy on 8 RAFT-derived classes from the 23-D `motion_features.npy`)
> 💡 **Why "temporal" alone is too vague:** motion class IS the temporal feature — derived from optical flow, camera-subtracted. The 8-class taxonomy is the concrete payoff.
> 🔗 **Sibling plan:** [`planCODE_trainHead_scaleBackbone_curriculum.md`](./planCODE_trainHead_scaleBackbone_curriculum.md) — research code; this doc is paper-deliverable.

---

## 📋 Phase summary

```
┌────────────────────────────────────────────┬────────┬──────────┬────────────────────────┐
│ Stage                                       │ When   │ Wall      │ Blockers                │
├────────────────────────────────────────────┼────────┼──────────┼────────────────────────┤
│ A. Copy iter13 v12 plots → placeholders    │ ✅ NOW │ 5 min     │ none                    │
│ B. Strip retrieval sections from HTML      │ ✅ NOW │ 1-2 hr    │ none                    │
│ C. CSS .placeholder framework               │ ✅ NOW │ 30 min    │ none                    │
│ D. Wire iter13 v12 plots into new sections │ ✅ NOW │ 1-2 hr    │ A + B + C               │
├────────────────────────────────────────────┼────────┼──────────┼────────────────────────┤
│ E. probe_action --stage labels on 23-D     │ ⏳ LATER│ 5 sec     │ none (CPU)              │
│ F. probe_action --stage train+eval (7 enc) │ ⏳ LATER│ ~30 min   │ E                       │
│ G. motion_spectrum_gallery.py (8 classes)  │ ⏳ LATER│ 2 hr dev  │ E + F                   │
│    + 30 min GPU pass for flow_to_image     │        │ 30 min GPU│                          │
│ H. "23 Champions" + "Anatomy" panels       │ ⏳ LATER│ 1 hr dev  │ G's RAFT viz pipeline   │
│    + 15 min GPU                             │        │ 15 min GPU│                          │
│ I. 23-D-correct probe results refresh      │ ⏳ LATER│ 30 min    │ Phase 6 head-only POC   │
└────────────────────────────────────────────┴────────┴──────────┴────────────────────────┘
                                          TOTAL NOW: ~3-5 hr  ·  TOTAL LATER: ~5-7 hr + ~75 min GPU
```

**Why split NOW vs LATER:** iter13 v12 plots use the OLD pre-Phase-0 vec[0] motion taxonomy. Numerically they're slightly stale (binning axis changed from `mean_magnitude` → `fg_mean_mag`). The STRUCTURE of the page (motion-class narrative) is durable; the SPECIFIC NUMBERS get refreshed in Stage I when Phase 6 head-only POC results land. NOW = ship the structure with explicit placeholders.

---

## ✅ NOW — Stage A: Copy iter13 v12 plots into `docs/static/images/`

**Goal:** seed `docs/` with the motion-class plots that exist on disk. Zero GPU.

```bash
cd /workspace/factorjepa

# 📂 Source: iter13 v12 results (FROZEN baselines + adapted m09a — all on OLD 13-D motion features)
SRC_PROBE=iter/iter13_motion_probe_eval/result_outputs/v12/full/probe_plot
SRC_TRAIN=iter/iter13_motion_probe_eval/result_outputs/v12/full/probe_pretrain
DST=docs/static/images/

# 🎨 Stage A.1 — probe_action plots (per-encoder motion-class accuracy)
cp $SRC_PROBE/probe_action_acc.png            $DST   # 7-encoder top-1 bar chart
cp $SRC_PROBE/probe_action_loss.png           $DST   # training trajectory
cp $SRC_PROBE/probe_encoder_comparison.png    $DST   # encoder rank ladder

# 🎨 Stage A.2 — m09a pretrain trajectory (already 23-D-aware via motion_aux head)
cp $SRC_TRAIN/m09a_probe_trajectory_trio.png  $DST   # probe top1 climbs across steps
cp $SRC_TRAIN/m09a_loss_decomposition.png     $DST   # JEPA + EWC + motion_aux split
cp $SRC_TRAIN/m09a_block_drift.png            $DST   # encoder weight drift per layer

# ✅ Verify
ls -la $DST | grep -E 'probe_action|m09a_'
```

**Pass criteria:** 6 new files in `docs/static/images/`, all .png.

---

## ✅ NOW — Stage B: Strip retrieval sections from `docs/index.html`

**Goal:** delete the retrieval-specific content + assets. Tab 1 (Denseworld) untouched.

### B.1 Section-by-section edits (Tab 2: FactorJEPA, lines 681-1213)

```
┌────┬──────────────────────────────┬─────────┬───────────────────────────────────────────────────────┐
│ #  │ Section (lines)               │ Verdict │ Action                                                 │
├────┼──────────────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ 1  │ Abstract (684-702)            │ REWRITE │ Replace retrieval-metric language with motion-class:  │
│    │                               │         │ "8-class motion taxonomy from 23-D camera-subtracted  │
│    │                               │         │ optical flow. V-JEPA top-1: 0.84 (frozen) — leading." │
│ 2  │ Key Findings 4 cards (705-730)│ REWRITE │ Each card → motion-class top-1 + paired-Δ R1 winner. │
│ 3  │ Six-Encoder Comparison        │ REPLACE │ DROP: m08b_encoder_comparison.png, m08b_radar.png.    │
│    │   (733-833)                   │         │ INSERT: probe_action_acc.png + probe_encoder_         │
│    │                               │         │   comparison.png. Table cols: motion_top1 /           │
│    │                               │         │   motion_top3 / per_class_macro_F1 / train_min_loss.  │
│ 4  │ Ch10: Frozen vs Adapted V-JEPA│ REWRITE │ Keep narrative (Δ ≈ 0 at 10K POC, full needs 115K).  │
│    │   (835-916)                   │         │ Switch metric from Prec@K to motion_top1. ADD iter14  │
│    │                               │         │   paired-Δ R0-R6 table placeholder.                   │
│ 5  │ kNN Retrieval Demo (918-1049) │ DELETE  │ Entire <section> deleted. Becomes "Motion-Class       │
│    │                               │ +REPLACE│ Spectrum Gallery" — see Stage G (LATER) for assets.   │
│ 6  │ Pipeline (1052-1078)          │ UPDATE  │ Last 2 nodes: drop "m08 retrieval evaluation" → add   │
│    │                               │         │ "m04d 23-D motion features → probe_action top-1".     │
│ 7  │ Detailed Results (1079-1114)  │ REPLACE │ DROP: m06_silhouette_per_key.png, m06_map_per_key.png,│
│    │                               │         │   m08_umap.png, m08_confusion_matrix.png.             │
│    │                               │         │ INSERT: probe_action_loss.png + motion-class confusion│
│    │                               │         │   matrix PLACEHOLDER.                                 │
│ 8  │ Spatial vs Temporal Analysis  │ REPLACE │ DROP all 4: m08b_spatial_temporal_bar.png, m08b_      │
│    │   (1117-1152) — 4 plots       │         │   tradeoff_scatter.png, m08b_temporal_ablation.png,   │
│    │                               │         │   m08b_heatmap.png.                                   │
│    │                               │         │ INSERT: m09a_probe_trajectory_trio.png + m09a_loss_   │
│    │                               │         │   decomposition.png.                                  │
│ 9  │ Spatial-Temporal Gap          │ REWRITE │ Flip narrative: was "V-JEPA fails on retrieval".     │
│    │   (1155-1171)                 │         │ Now: "V-JEPA WINS on motion-class (top-1 0.84)        │
│    │                               │         │   because temporal IS what motion needs."             │
│    │                               │         │ Cite arxiv:2509.21595 (same external validator).      │
│ 10 │ Roadmap (1174-1211)           │ UPDATE  │ Add iter15 rows: head-only freeze, surgery, FG motion│
│    │                               │         │ features (Phase 0/3 ✅ DONE). Update status badges.    │
└────┴──────────────────────────────┴─────────┴───────────────────────────────────────────────────────┘
```

### B.2 Tab 3: Foundations (lines 1216-1665) — mostly KEEP

```
┌────┬──────────────────────────────┬─────────┬───────────────────────────────────────────────────────┐
│ #  │ Section                       │ Verdict │ Action                                                 │
├────┼──────────────────────────────┼─────────┼───────────────────────────────────────────────────────┤
│ 11 │ Where Does V-JEPA Fit? + ML   │ KEEP    │ Pure pedagogy — no retrieval refs.                    │
│    │ Taxonomy (1218-1270)          │         │                                                        │
│ 12 │ SSL methods comparison        │ KEEP    │ Same.                                                  │
│ 13 │ V-JEPA 2 in One Picture       │ KEEP    │ Same.                                                  │
│ 14 │ If You Know LLMs... (1355-96) │ KEEP    │ Same.                                                  │
│ 15 │ System Design (1397+)         │ AUDIT   │ Read tail of file; flag any retrieval refs in System  │
│    │                               │         │ Design diagram or captions.                           │
└────┴──────────────────────────────┴─────────┴───────────────────────────────────────────────────────┘
```

### B.3 Delete retrieval-specific assets

```bash
cd /workspace/factorjepa

# 🗑️ Remove retrieval images (11 files — m08_umap_grid added 2026-05-14
# after Stage A pre-flight audit caught it as a retrieval UMAP variant)
git rm docs/static/images/m06_map_per_key.png
git rm docs/static/images/m06_silhouette_per_key.png
git rm docs/static/images/m08_confusion_matrix.png
git rm docs/static/images/m08_umap.png
git rm docs/static/images/m08_umap_grid.png
git rm docs/static/images/m08b_encoder_comparison.png
git rm docs/static/images/m08b_heatmap.png
git rm docs/static/images/m08b_radar.png
git rm docs/static/images/m08b_spatial_temporal_bar.png
git rm docs/static/images/m08b_temporal_ablation.png
git rm docs/static/images/m08b_tradeoff_scatter.png

# 🗑️ Remove kNN demo videos (24 files)
git rm docs/static/videos/knn/*.mp4

# ✅ Verify nothing retrieval-y remains
ls docs/static/images/ docs/static/videos/ | grep -iE 'knn|m06_|m08' || echo "  ✅ all retrieval assets gone"
```

**Pass criteria:**
- 0 retrieval `<img>` tags in `docs/index.html` (`grep -c m08b\|m06_\|knn docs/index.html` → 0).
- 0 retrieval files in `docs/static/`.
- Tab 1 (Denseworld) untouched — same `<video>` count as before.

---

## ✅ NOW — Stage C: CSS `.placeholder` framework

**Goal:** every future `<img>` that points to a not-yet-built motion plot shows an explicit placeholder, not a broken image icon.

### C.1 Add CSS to `<style>` block in `docs/index.html`

```css
/* Placeholder visual for plots not yet generated.
   Dashed gray border + centered icon + label + ETA caption. */
.plot-container.placeholder {
  border: 2px dashed #999;
  background: linear-gradient(135deg, #f4f4f4 0%, #ececec 100%);
  padding: 3rem 1.5rem;
  text-align: center;
  min-height: 240px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 0.6rem;
}
.placeholder-icon { font-size: 2.4rem; opacity: 0.7; }
.placeholder-label { font-weight: 600; color: #555; font-size: 1rem; }
.placeholder-eta {
  font-style: italic; color: #888; font-size: 0.85rem; max-width: 28rem;
}
```

### C.2 Reusable HTML pattern

```html
<div class="plot-container placeholder">
  <div class="placeholder-icon">⏳</div>
  <div class="placeholder-label">motion-class confusion matrix</div>
  <div class="placeholder-eta">
    pending: probe_action --stage eval after Phase 6 head-only POC
  </div>
</div>
```

**Pass criteria:** at least 2 `.plot-container.placeholder` blocks render correctly in a local browser (gray dashed border + ⏳ icon + caption).

---

## ✅ NOW — Stage D: Wire iter13 v12 plots into refactored HTML

**Goal:** every section that has a plot points to a REAL file (iter13 placeholder) or a `.placeholder` block. No broken `<img src=...>`.

### D.1 Wiring table

```
┌──────────────────────────────────────┬───────────────────────────────────────────────────────┐
│ Section                               │ Plot source (status)                                   │
├──────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Six-Encoder Comparison (#3)           │ probe_action_acc.png        ✅ iter13 v12 placeholder │
│                                       │ probe_encoder_comparison.png ✅ iter13 v12 placeholder │
│ Ch10: Frozen vs Adapted V-JEPA (#4)   │ m09a_probe_trajectory_trio.png ✅ iter13 v12          │
│                                       │ iter14 R0-R6 table          ⏳ placeholder (Stage I)  │
│ Detailed Results (#7)                 │ probe_action_loss.png        ✅ iter13 v12 placeholder │
│                                       │ motion-class confusion       ⏳ placeholder (Stage F)  │
│ Spatial vs Temporal Analysis (#8)     │ m09a_probe_trajectory_trio.png ✅ iter13 v12          │
│                                       │ m09a_loss_decomposition.png    ✅ iter13 v12          │
│ kNN Retrieval Demo replacement (#5)   │ motion-class spectrum gallery ⏳ placeholder (Stage G)│
└──────────────────────────────────────┴───────────────────────────────────────────────────────┘
```

### D.2 Banner: "data caveats" callout near top of Tab 2

Add a one-time visible note explaining the placeholder status — so readers understand which numbers are final and which are pending:

```html
<div class="notification is-warning" style="max-width: 900px; margin: 0 auto 2rem auto;">
  <strong>⏳ iter15 refresh in progress.</strong>
  Plots tagged with the dashed-border placeholder are pending the
  Phase 6 head-only POC. Numerical results currently displayed use iter13 v12
  baselines (13-D motion taxonomy, vec[0] binning). Final figures will use the
  23-D camera-subtracted taxonomy (vec[13] binning, Phase 3 ✅ complete on
  2026-05-14). See <a href="https://github.com/kapilw25/factorjepa/blob/main/iter/iter15_trainHead_freezeEncoder/planCODE_html.md">planCODE_html.md</a>.
</div>
```

### D.3 Smoke test in local browser

```bash
# 🌐 Serve docs/ on localhost:8000 — open and click each tab
python -m http.server 8000 -d docs

# In another shell, sanity-grep for broken img refs that didn't land in Stages A-C
grep -oE 'src="[^"]+\.(png|mp4)"' docs/index.html | sort -u | \
  while read line; do
    path=$(echo "$line" | sed 's|src="||; s|"$||')
    test -f "docs/$path" || echo "  ❌ MISSING: $path"
  done | head -20
```

**Pass criteria:**
- All `<img>` and `<video>` tags resolve (no 404s in browser DevTools).
- Tab 1 (Denseworld) renders identically to pre-refactor.
- Tab 2 (FactorJEPA) shows motion-class narrative with ≥4 real plots and ≥3 visible placeholders.
- Tab 3 (Foundations) renders identically except for any retrieval-ref audit fix.

---

## ⏳ LATER — Stage E: Regenerate `action_labels.json` on 23-D taxonomy

**Goal:** the new vec[13] FG-magnitude binning replaces vec[0] global-magnitude binning. CPU only, ~5 sec.

**Blocker for:** Stages F, G, I.

```bash
# 🏷️ Regenerate action labels from the new 23-D motion_features.npy
python -u src/probe_action.py --FULL \
    --stage labels \
    --eval-subset data/eval_10k.json \
    --motion-features data/eval_10k_local/motion_features.npy \
    --output-dir outputs/full/probe_action \
    2>&1 | tee logs/probe_action_labels_phase3_$(date +%Y%m%d_%H%M%S).log

# ✅ Verify class distribution looks sane (8 classes expected)
python -c "
import json
labels = json.loads(open('outputs/full/probe_action/action_labels.json').read())
from collections import Counter
cnt = Counter(v['class_id'] for v in labels.values())
print(f'  n_classes: {len(cnt)}')
print(f'  per-class counts: {dict(sorted(cnt.items()))}')"
```

**Pass criteria:** 8 classes survive the ≥34-clip filter; per-class counts within 5× of each other.

---

## ⏳ LATER — Stage F: Re-run probe_action on 7 encoders with 23-D labels

**Goal:** refresh `probe_action_acc.png` + `probe_action_loss.png` + the motion-class confusion matrix with the new vec[13] taxonomy.

**Blocker for:** Stage I (final numerical refresh).
**Depends on:** Stage E.

```bash
# 🚀 Run the full probe pipeline against 7 encoders (frozen + 6 from iter13/iter14)
./scripts/run_eval.sh --FULL 2>&1 | tee logs/run_eval_phase3_refresh_$(date +%Y%m%d_%H%M%S).log

# 📂 New plots land at:
#   outputs/full/probe_plot/probe_action_acc.png    (refresh of iter13 v12)
#   outputs/full/probe_plot/probe_action_loss.png
#   outputs/full/probe_plot/probe_encoder_comparison.png
#   outputs/full/probe_action/confusion_matrix.png  (NEW)
```

**Pass criteria:**
- frozen V-JEPA top-1 ≥ 0.80 (sanity check vs iter13 v12 baseline of 0.84).
- 8 surviving classes consistent with Stage E output.
- confusion matrix shows diagonal ≥ 0.65 per class (motion-class is separable).

---

## ⏳ LATER — Stage G: `motion_spectrum_gallery.py` (8 classes × 5 clips + RAFT color wheel)

**Goal:** replace the deleted kNN Retrieval Demo with the "Motion-Class Spectrum" gallery — the strongest narrative pivot in the refactor.

**Depends on:** Stage E (need action_labels.json with 23-D taxonomy).

### G.1 New file: `src/utils/motion_spectrum_gallery.py` (~250 LoC)

```
┌──────────────────────────────────────────────────────────────────────┐
│ INPUTS (all via argparse, required=True, no defaults)                  │
│   --motion-features    data/eval_10k_local/motion_features.npy        │
│   --eval-subset        data/eval_10k.json                             │
│   --tar-dir            data/eval_10k_local                            │
│   --action-labels      outputs/full/probe_action/action_labels.json   │
│   --output-video-dir   docs/static/videos/motion_spectrum             │
│   --output-flow-dir    docs/static/videos/motion_flow                 │
│   --n-per-class        5                                              │
│   --num-frames         16                                             │
│                                                                       │
│ ALGORITHM (3 stages)                                                   │
│ 1. Pick representatives                                               │
│      For each surviving class:                                        │
│        centroid = mean(vec[class_members], axis=0)                    │
│        d_i = ||vec_i - centroid||_2  for each class member            │
│        keep top-N smallest d_i  → "platonic ideal" of the class       │
│                                                                       │
│ 2. Extract mp4 bytes from TAR shards                                  │
│      Reuse utils.data_download.iter_clips_parallel with subset_keys=  │
│      the selected clip_keys; pluck mp4 bytes; write to                 │
│      docs/static/videos/motion_spectrum/<class>_<rank>.mp4            │
│                                                                       │
│ 3. Render Middlebury color-wheel flow visualization                   │
│      Decode the mp4 → RAFT → torchvision.utils.flow_to_image →        │
│      imageio.mimsave as <class>_<rank>_flow.mp4 (8-fps loop).         │
│                                                                       │
│ OUTPUTS                                                                │
│   docs/static/videos/motion_spectrum/<class>_<rank>.mp4         (~40) │
│   docs/static/videos/motion_flow/<class>_<rank>_flow.mp4        (~40) │
│   docs/static/data/motion_spectrum_manifest.json                      │
│     { class_name: [{clip_key, video_path, flow_path, vec23d}, ...] }  │
└──────────────────────────────────────────────────────────────────────┘
```

### G.2 Key code skeleton

```python
"""Motion-class spectrum gallery: 8 classes × 5 representative clips +
RAFT Middlebury color-wheel flow visualization. Output → docs/static/.

USAGE (GPU; ~30 min wall):
    python -u src/utils/motion_spectrum_gallery.py \
        --motion-features data/eval_10k_local/motion_features.npy \
        --eval-subset     data/eval_10k.json \
        --tar-dir         data/eval_10k_local \
        --action-labels   outputs/full/probe_action/action_labels.json \
        --output-video-dir docs/static/videos/motion_spectrum \
        --output-flow-dir  docs/static/videos/motion_flow \
        --n-per-class 5 --num-frames 16 \
        2>&1 | tee logs/motion_spectrum_gallery_$(date +%Y%m%d_%H%M%S).log
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import imageio
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.config import check_gpu
from utils.data_download import iter_clips_parallel
# Reuse m04d's frame-decode + preprocess helpers — same RAFT input format
from m04d_motion_features import decode_video_frames, _preprocess_pairs


def pick_class_representatives(vecs, clip_keys, class_id_by_key, n_per_class):
    """Return {class_id: [clip_key, ...]} — N closest to class centroid."""
    by_class = {}
    for k, cid in class_id_by_key.items():
        by_class.setdefault(cid, []).append(k)
    out = {}
    for cid, members in by_class.items():
        idx = [clip_keys.index(m) for m in members if m in clip_keys]
        if len(idx) < n_per_class:
            continue
        sub = vecs[idx]
        centroid = sub.mean(axis=0, keepdims=True)
        d = np.linalg.norm(sub - centroid, axis=1)
        top = np.argsort(d)[:n_per_class]
        out[cid] = [members[i] for i in top]
    return out


def render_flow_visualization(mp4_bytes, raft_model, transforms,
                              out_path, num_frames):
    """Decode → RAFT → flow_to_image (Middlebury color wheel) → mp4 loop."""
    pairs = decode_video_frames(mp4_bytes, n_pairs=num_frames)
    prev, curr = _preprocess_pairs(pairs, transforms)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        flows = raft_model(prev.cuda(), curr.cuda())[-1]      # (N, 2, H, W)
    rgb = flow_to_image(flows.float()).cpu().numpy()           # (N, 3, H, W) u8
    rgb = rgb.transpose(0, 2, 3, 1)                             # (N, H, W, 3)
    imageio.mimsave(str(out_path), rgb, fps=8, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-features", type=Path, required=True)
    parser.add_argument("--eval-subset",     type=Path, required=True)
    parser.add_argument("--tar-dir",         type=Path, required=True)
    parser.add_argument("--action-labels",   type=Path, required=True)
    parser.add_argument("--output-video-dir", type=Path, required=True)
    parser.add_argument("--output-flow-dir",  type=Path, required=True)
    parser.add_argument("--n-per-class", type=int, required=True)
    parser.add_argument("--num-frames",  type=int, required=True)
    args = parser.parse_args()

    check_gpu()
    print_cgroup_header(prefix="[motion_spectrum]")
    start_oom_watchdog(prefix="[motion_spectrum-oom-watchdog]")

    # ... rest of pipeline: load features+labels, pick reps, extract mp4
    #     bytes, render flow viz, write manifest.json ...
```

### G.3 HTML section that consumes the manifest

```html
<section class="section">
  <div class="container" style="max-width: 1100px;">
    <h2 class="title is-3 has-text-centered">Motion-Class Spectrum</h2>
    <p class="has-text-centered" style="margin-bottom: 1.5rem; color: var(--text-secondary);">
      8 motion classes from 23-D camera-subtracted optical-flow features.
      Each class shows 5 representative clips (closest to class centroid in 23-D space).
      Top row: original mp4. Bottom row: RAFT flow via Middlebury color wheel
      (hue = direction, saturation = magnitude).
    </p>
    <!-- 8 motion-class-row blocks, server-rendered from motion_spectrum_manifest.json -->
  </div>
</section>
```

**Pass criteria:**
- 8 motion-class rows render (or however many survive the 34-clip filter).
- Within each row, all 5 flow.mp4 cells show a CONSISTENT color tint (e.g., all reddish for `fast__rightward`).
- Scene diversity within a row is visible — different cities, different scenes.

---

## ⏳ LATER — Stage H: "23 Champions" + "Anatomy" panels

**Goal:** layered visualizations of WHAT each of the 23 dims captures. Both ride on the same RAFT viz pipeline built in Stage G.

**Depends on:** Stage G's `render_flow_visualization` helper.

### H.1 "23 Dimension Champions" — Tab 2 header showcase

```bash
# 🏆 For each of the 23 dims, find argmax across 9297 clips → render
python -u src/utils/motion_spectrum_gallery.py --mode champions \
    --motion-features data/eval_10k_local/motion_features.npy \
    --output-video-dir docs/static/videos/dim_champions \
    --output-flow-dir  docs/static/videos/dim_champions_flow \
    --num-frames 16 \
    2>&1 | tee logs/dim_champions_$(date +%Y%m%d_%H%M%S).log
```

Add a `--mode {spectrum, champions}` flag to `motion_spectrum_gallery.py` so both visualizations share one entry point. ~50 extra LoC.

### H.2 "Anatomy of One Clip" — pedagogical hero panel

Single static SVG/PNG produced by a small `scripts/render_anatomy_panel.py`:
- 1 hero clip (pick clip with vec[11] strong rightward camera pan + vec[15:23] non-trivial FG)
- 2 RAFT flow panels side-by-side: Global vs FG-subtracted
- 8-axis radar chart for global `dir_hist` + 8-axis radar for FG `fg_dir_hist`
- 7 scalar pips (mean/std/max global + cam_x/cam_y + fg_mean_mag/fg_max_mag)

Output: `docs/static/images/anatomy_hero_panel.png` (one large composite figure).

**Pass criteria:** the anatomy panel makes a non-ML reader say "ah, so vec[13] is the bit-without-the-camera-pan".

---

## ⏳ LATER — Stage I: Numerical refresh from Phase 6 head-only POC

**Goal:** swap iter13 v12 placeholder plots for fresh iter15 head-only training results.

**Depends on:** Phase 6 head-only POC complete (3 cells × ~6-8 hr each — see [`planCODE_trainHead_scaleBackbone_curriculum.md`](./planCODE_trainHead_scaleBackbone_curriculum.md) Phase 6).

```bash
# 🔄 After Phase 6 POC + Phase 4 eval, run probe_action again:
./scripts/run_eval.sh --FULL 2>&1 | tee logs/run_eval_iter15_final_$(date +%Y%m%d_%H%M%S).log

# 📂 Copy iter15 v1 results over iter13 v12 placeholders:
SRC=iter/iter15_trainHead_freezeEncoder/result_outputs/v1/full/probe_plot
cp $SRC/probe_action_acc.png            docs/static/images/probe_action_acc.png
cp $SRC/probe_action_loss.png           docs/static/images/probe_action_loss.png
cp $SRC/probe_encoder_comparison.png    docs/static/images/probe_encoder_comparison.png
# ... etc ...

# 🚧 Remove the "⏳ iter15 refresh in progress" warning banner (Stage D.2).
```

**Pass criteria:**
- All `.plot-container.placeholder` blocks now point at real iter15 v1 plots.
- The warning banner is removed.
- Final paper-ready static page.

---

## 🎯 Pass criteria — overall

```
┌────────────────────────────────────────────────────────────────────┐
│ NOW (Stages A-D) — ship a refactored docs/index.html               │
├────────────────────────────────────────────────────────────────────┤
│ ✅ 0 retrieval `<img>` / `<video>` tags in active scope             │
│ ✅ 10 retrieval images + 24 kNN videos git-rm'd                    │
│ ✅ 6 iter13 v12 plots copied to docs/static/images/                │
│ ✅ Tab 1 (Denseworld) byte-identical to pre-refactor               │
│ ✅ Tab 2 (FactorJEPA) shows motion-class narrative end-to-end       │
│ ✅ Tab 3 (Foundations) renders unchanged except System Design audit│
│ ✅ All non-placeholder `<img src=...>` resolve in browser          │
│ ✅ Warning banner explains placeholder status to readers            │
├────────────────────────────────────────────────────────────────────┤
│ LATER (Stages E-I) — refresh with iter15 v1 numerics               │
├────────────────────────────────────────────────────────────────────┤
│ ✅ 23-D action_labels.json regenerated (8 surviving classes)        │
│ ✅ probe_action --stage train+eval refreshed on 23-D taxonomy       │
│ ✅ motion_spectrum_gallery.py produces 40 video + 40 flow mp4s     │
│ ✅ 23-Champions + Anatomy panels rendered                          │
│ ✅ Phase 6 head-only POC results swapped in                        │
│ ✅ Warning banner removed                                          │
└────────────────────────────────────────────────────────────────────┘
```

---

## 💰 Budget recap

```
┌────────────────────────────────────────┬──────────────┬─────────────────┐
│ Stage                                   │ Wall          │ GPU cost (24GB) │
├────────────────────────────────────────┼──────────────┼─────────────────┤
│ ✅ NOW  A. Copy iter13 plots            │ 5 min         │ $0              │
│ ✅ NOW  B. Strip retrieval sections     │ 1-2 hr        │ $0              │
│ ✅ NOW  C. CSS .placeholder framework   │ 30 min        │ $0              │
│ ✅ NOW  D. Wire plots + smoke test      │ 1-2 hr        │ $0              │
├────────────────────────────────────────┼──────────────┼─────────────────┤
│ ⏳ LATER E. probe_action --stage labels │ 5 sec         │ $0 (CPU)        │
│ ⏳ LATER F. probe_action train+eval ×7  │ ~30 min       │ ~$0.10          │
│ ⏳ LATER G. motion_spectrum_gallery.py  │ 2 hr + 30 min │ ~$0.10          │
│ ⏳ LATER H. 23-Champions + Anatomy      │ 1 hr + 15 min │ ~$0.05          │
│ ⏳ LATER I. Phase 6 results refresh     │ 30 min        │ $0 (file copy)  │
├────────────────────────────────────────┼──────────────┼─────────────────┤
│ 🎯 TOTAL                                │ ~8-10 hr      │ ~$0.25          │
└────────────────────────────────────────┴──────────────┴─────────────────┘
```

NOW work is 100% GPU-free — you can ship Stages A-D from any laptop. LATER work needs ~75 min of GPU time spread across 3 sessions.
