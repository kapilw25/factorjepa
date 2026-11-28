---
license: cc-by-4.0
pretty_name: DENSEWORLD-115k
language:
- en
task_categories:
- video-classification
- other
tags:
- video
- world-models
- jepa
- urban
- global-south
- india
- youtube
- metadata-only
size_categories:
- 100K<n<1M
---

# DENSEWORLD-115k

**A large-scale video benchmark of populous, crowded, and chaotic Global South urban environments**, used to study world models (JEPA) under soft spatial boundaries, extreme agent heterogeneity, persistent occlusion, and rapid social negotiation.

- **115,687** clips (4–14 s each, median 9.3 s; 99.8% under 10.5 s)
- **714** long-form source videos across **22 Indian cities**
- Drive-through, walk-through, and aerial (drone) viewpoints; markets, ghats, junctions, flyovers, beaches, and more

## ⚠️ This is a metadata-only dataset (no videos included)

The source clips are derived from **public YouTube videos** and remain under their creators' copyright. We therefore **do not redistribute any video files**. Instead, this repository ships the **source video list, a clip manifest, and a deterministic reconstruction script** so you can rebuild the exact clips locally from YouTube. This mirrors the standard practice of YouTube-derived datasets (e.g. Panda-70M, HD-VILA-100M, HowTo100M).

## Repository contents

```
denseworld-115k/
├── README.md                     # this card
├── requirements.txt              # yt-dlp, scenedetect[opencv]
├── reconstruct.py                # download → scene-detect → split → rebuild the 115,687 clips
├── sources.json                  # 714 YouTube source videos (id, url, section, n_clips, title)
├── clips.csv                     # 115,687-clip manifest (clip_key, section, base_video_id, chunk, duration…)
└── data/
    └── data_prep/
        ├── clip_durations.json   # per-video clip manifest (ground truth for verification)
        ├── city_matrix.json      # per-city × capture-type coverage counts
        └── word_frequency.json   # source-title word frequencies + scene taxonomy mapping
```

## Reconstructing the clips

```bash
pip install -r requirements.txt          # yt-dlp + PySceneDetect
# install ffmpeg + ffprobe (brew install ffmpeg  /  apt-get install ffmpeg)

python reconstruct.py --out ./denseworld_clips --limit 1   # smoke test (1 video)
python reconstruct.py --out ./denseworld_clips             # full rebuild (all 714 videos)
python reconstruct.py --out ./denseworld_clips --verify-only   # check counts vs the manifest
```

The pipeline is **deterministic** and reproduces the released clips exactly:

1. **Download** each source video at 480p (`yt-dlp`).
2. **Detect scene boundaries** (PySceneDetect `ContentDetector`, threshold 15.0).
3. **Greedy-split** into contiguous clips at scene boundaries, targeting 4–10 s (encoded clips run 4–14 s; see `duration_sec` in `clips.csv`).
4. **Encode** each clip (`ffmpeg` libx264 CRF 28, AAC 128k).

Clips are written to `<out>/<section>/<video_id>-<NNN>.mp4` (e.g. `goa/walking/04YKvC8kAgI-000.mp4`), matching the `clip_key`s in `clips.csv`.

> Some source videos may become unavailable over time (deleted or made private on YouTube); `reconstruct.py` skips these and reports the shortfall in its verification summary. Three very long videos were originally cut in fixed windows rather than whole-video scene detection, so their clip boundaries may differ slightly (<1% of the dataset).

## Data fields

**`sources.json`** → `videos: [ { … } ]`

| field | description |
|---|---|
| `id` | 11-char YouTube video ID (source URL = `https://www.youtube.com/watch?v=<id>`) |
| `url` | full YouTube watch URL |
| `sections` | list of `city/capture-type` sections this video contributes to (e.g. `goa/walking`) |
| `n_clips` | number of clips produced from this video |
| `n_chunks` | >0 only for the 3 pre-chunked long videos |
| `title`, `category` | source title and collection category (`drive_tours`, `walking_tours`, `drone_views`, `tier2_cities`) |

**`clips.csv`** (115,687 rows)

| column | description |
|---|---|
| `clip_key` | `section/video_id/file` — canonical clip identifier |
| `section` | `city/capture-type` (e.g. `mumbai/drive`) |
| `base_video_id` | the 11-char YouTube ID |
| `chunk` | fixed-window chunk index for the 3 long videos, else empty |
| `video_id` | manifest key (`base_video_id` or `base_video_id-<chunk>`) |
| `clip_index` | clip order within its video/chunk |
| `duration_sec`, `size_mb`, `status` | encoded clip stats |

## License & responsible use

- The **metadata and code** in this repository are released under **CC-BY-4.0**.
- The **videos are NOT included** and are **not** covered by this license — they remain the property of their original YouTube uploaders and are subject to YouTube's Terms of Service. Use the reconstructed clips for research purposes and in accordance with those terms.
- **Takedown:** if you are a rights holder and want a source removed from `sources.json`, please open an issue on this repository and we will remove it.

## Citation

Withheld during anonymous review; it will be added here after the review period.
