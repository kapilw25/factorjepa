Plan: Convert WalkIndia-200k to WebDataset TAR Shards + Upload to HF

Context

115,687 mp4 clips (121.2 GB) across 75 sections need uploading to HuggingFace. Individual mp4 upload failed due to:
- HF 10k files/directory limit (kolkata/walking has 20,633 files)
- 256 commits/hour rate limit (stuck at 104k/115k for 12+ hours)
- MerkleDB xet cache errors

Solution: Convert to WebDataset TAR shards (~1GB each). HF sees ~120 files instead of 115k. Enables Dataset Viewer + streaming for V-JEPA evaluation.

New file: src/m05_upload_hf.py

Single script that: (1) packs clips into TAR shards, (2) uploads to HF.

TAR Structure (HF WebDataset convention)

data/
├── train-00000.tar
│   ├── 000000.mp4          # clip video
│   ├── 000000.json         # metadata for this clip
│   ├── 000001.mp4
│   ├── 000001.json
│   └── ...                 # ~1000 clips per shard
├── train-00001.tar
├── ...
└── train-00115.tar         # ~116 shards total

Each JSON sidecar:
{
"video_id": "04YKvC8kAgI",
"section": "goa/walking",
"city": "goa",
"tour_type": "walking",
"tier": "goa",
"duration_sec": 9.73,
"size_mb": 1.57,
"clip_index": 0,
"source_file": "04YKvC8kAgI-000.mp4"
}

Sharding Strategy

- Target: ~1000 clips per shard (~1 GB each)
- 115,687 clips / 1000 = ~116 shards
- Clips sorted by section → video_id → clip number (preserves geographic coherence within shards)
- Global sequential index (000000, 000001, ...) inside each shard for HF compatibility

Implementation: src/m05_upload_hf.py

USAGE:
python -u src/m05_upload_hf.py --SANITY 2>&1 | tee logs/m05_upload_hf_sanity.log
python -u src/m05_upload_hf.py --FULL 2>&1 | tee logs/m05_upload_hf_full.log

Step 1: Build clip manifest from clip_durations.json
- Read outputs_data_prep/clip_durations.json (already has all metadata)
- Build sorted list: [(clip_path, metadata_dict), ...] sorted by section/video/clip
- Total: 115,687 entries

Step 2: Create TAR shards (write to src/data/shards/)
- For each batch of 1000 clips:
- Create train-{shard_idx:05d}.tar
- For each clip in batch:
- Add mp4 file as {global_idx:06d}.mp4
- Add JSON metadata as {global_idx:06d}.json
- Print progress: [shard 42/116] 1000 clips, 1.05 GB
- Disk concern: Shards are written one at a time. Each shard ~1GB. We need ~121GB total for shards BUT clips are 121GB too. Total ~242GB needed.
- Solution: Stream — write shard, upload it, delete it, then write next shard. Only 1 shard (~1GB) on disk at any time.

Step 3: Upload each shard to HF
- For each shard tar file:
- api.upload_file(path_or_fileobj=shard_path, path_in_repo=f"data/{shard_name}", ...)
- On success: delete local shard file
- Print: Uploaded train-00042.tar (1.05 GB) [43/116]
- 116 uploads = 116 commits, well under 256/hour limit

Step 4: Upload README.md
- Reuse generate_readme() from src/utils/hf_upload.py (update to mention WebDataset format)
- Update dataset card tags to include webdataset

SANITY mode

- Process only first 2 shards (2000 clips)
- Upload to same repo (test that viewer works)

Files modified

1. src/m05_upload_hf.py — NEW

Main script. Uses:
- clip_durations.json as data source (from m02b_scene_fetch_duration.py)
- CLIPS_DIR from src/utils/config.py for local clip paths
- _setup_hf_env(), _get_token(), generate_readme() from src/utils/hf_upload.py
- HF_DATASET_REPO from src/utils/config.py
- Standard tarfile module (no webdataset pip package needed — just raw tar creation)

2. src/utils/hf_upload.py — MINOR UPDATE

- Update generate_readme() to mention WebDataset format in the dataset card
- Add webdataset to tags in YAML frontmatter
- Keep existing functions (they may be useful for metadata-only uploads later)

3. src/utils/config.py — MINOR UPDATE

- Add SHARDS_DIR = DATA_DIR / "shards" path constant

Files NOT modified

- src/m02b_scene_fetch_duration.py — already generates clip_durations.json
- src/m02_scene_detect.py — clip generation is done
- Other modules — unaffected

HF Repo Cleanup

Before uploading WebDataset shards, delete the existing 104k individual mp4 files from the repo:
api.delete_folder(path_in_repo="tier1", repo_id=repo_id, repo_type="dataset")
api.delete_folder(path_in_repo="tier2", repo_id=repo_id, repo_type="dataset")
api.delete_folder(path_in_repo="goa", repo_id=repo_id, repo_type="dataset")
api.delete_folder(path_in_repo="monuments", repo_id=repo_id, repo_type="dataset")
Or create a fresh repo.

Disk Space Plan

- Current free: ~60 GB
- Streaming approach: only 1 shard (~1 GB) on disk at a time
- No extra disk needed beyond existing clips (121 GB)

Verification

1. python -m py_compile src/m05_upload_hf.py
2. python src/m05_upload_hf.py --help
3. python -u src/m05_upload_hf.py --SANITY — creates 2 shards, uploads, check HF viewer
4. If viewer works → python -u src/m05_upload_hf.py --FULL
5. Check HF Dataset Viewer shows video grid with metadata columns
6. Test streaming: load_dataset("anonymousML123/walkindia-200k", streaming=True)

ETA

- TAR creation: ~1000 clips/shard, ~1 sec to pack (mostly I/O) → ~2 min for all 116 shards
- Upload: ~1 GB/shard, ~116 shards. At ~10 MB/s upload → ~12 GB/hour → ~10 hours
- But only 116 commits (well under 256/hour) — no rate limit issues
- With parallel upload_file calls: could be faster
- Total: ~10-12 hours (upload-bound, no rate limit blocking)
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌