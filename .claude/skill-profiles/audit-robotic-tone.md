# FactorJEPA profile for /audit-robotic-tone

Read by `.claude/skills/audit-robotic-tone/check_tone.py` (the ```json block) and by the skill (the table). Everything specific to
FactorJEPA lives here; the skill itself stays the same in every project.

| Key | FactorJEPA value | Why |
|---|---|---|
| Workshop folder | `overleaf/v4_AAAI_workshop` | every venue draft is `<folder>/v*_*/0_main.tex` |
| Run from | the workshop folder | `\input{../common/...}` resolves from each venue folder |
| Research words kept | frozen, adapter | "the frozen backbone", "LoRA adapter" are method words here |
| Jargon added | arm, lease, POC, SANITY, shard, ... | operations words of this codebase that leak into prose |
| Build one draft | `python -I build_all.py` (all drafts; no per-draft builder) | paper-variants builder |

```json
{
  "engineering_drop": ["frozen", "adapter"],
  "engineering_add": ["arm", "lease", "leased", "POC", "SANITY", "shard", "sweep", "config", "yaml", "script",
                      "GPU", "rerun", "eval set", "ckpt", "registry", "manifest", "cache"],
  "model_names_add": ["V-JEPA ?2(?:\\.1)?", "DENSEWORLD-115k", "BDD100K", "nuScenes"],
  "macro_roots": [".", ".."],
  "ignore_macros": ["disclosure"]
}
```
