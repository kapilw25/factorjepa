"""Retrofit an ALREADY-PUSHED FactorJEPA repo to be self-contained, WITHOUT re-uploading the multi-GB
weights: upload the inference bundle (loader + vendored V-JEPA 2 arch + requirements) and patch its model
card (replace the old broken Usage block → Quick-start + Architecture, add the Attribution section, refresh
the Files table). The published metrics are PRESERVED (only the load/arch/attribution sections change).

New arms pushed through `hf_finetuned_push.py` are ALREADY self-contained (it ships the bundle via
`_stage_inference_bundle` + generates the full card); this script is ONLY for repos pushed BEFORE that
wiring. It reuses the SAME card sections as the generator (`_quickstart_block` / `_files_table_md` /
`_ARCH_BLOCK` / `_ATTRIB_BLOCK`) so the two never diverge.

USAGE:
  python -m utils.hf_retrofit_cards <repo_id> [<repo_id> ...]            # patch + upload
  python -m utils.hf_retrofit_cards --dry-run <repo_id> [<repo_id> ...]  # patch + validate, NO upload
"""
import argparse
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, upload_folder

from utils.hf_finetuned_push import (
    _ARCH_BLOCK,
    _ATTRIB_BLOCK,
    _files_table_md,
    _get_token,
    _quickstart_block,
    _stage_inference_bundle,
)

_CKPT_RE = re.compile(r"m09[a-z]_ckpt_best\.pt")
# stale tokens the old (pre-fix) card carried — they MUST be gone after a patch (else the card still misleads)
_STALE = ("vit_giant_xformers_rope", "utils.vjepa2_imports", "strict=False", "m09c1_surgery_encoder.py")


def _detect_ckpt(files: list) -> str:
    """The arm's predictor-bearing ckpt name varies (m09a/m09b/m09c). Read it off the repo file list."""
    hits = [f for f in files if _CKPT_RE.fullmatch(f)]
    return hits[0] if hits else "m09c_ckpt_best.pt"


def _patch_card(text: str, repo_id: str, ckpt: str) -> str:
    """Swap the old Usage block for Quick-start+Architecture, refresh the Files table, add Attribution.
    FAIL LOUD if the card is not the expected original-template shape (so we never push a half-patched card)."""
    new_usage = _quickstart_block(repo_id, ckpt) + "\n\n" + _ARCH_BLOCK + "\n\n"
    text, n1 = re.subn(r"## 🚀 Usage\n.*?(?=## 📦 Files in this repo)",
                       lambda _m: new_usage, text, flags=re.DOTALL)
    text, n2 = re.subn(r"## 📦 Files in this repo\n.*?(?=## 🧪 Reproducibility)",
                       lambda _m: _files_table_md(ckpt) + "\n\n", text, flags=re.DOTALL)
    text, n3 = re.subn(r"## 📝 Citation", lambda _m: _ATTRIB_BLOCK + "\n\n## 📝 Citation", text)
    if (n1, n2, n3) != (1, 1, 1):
        raise RuntimeError(f"{repo_id}: card not in expected template shape (usage={n1} files={n2} cite={n3}) — "
                           f"skipped (already retrofitted or hand-edited?)")
    leftover = [b for b in _STALE if b in text]
    if leftover:
        raise RuntimeError(f"{repo_id}: stale tokens still present after patch: {leftover}")
    return text


def retrofit_repo(repo_id: str, token: str, dry_run: bool = False) -> str:
    files = list_repo_files(repo_id, token=token)
    ckpt = _detect_ckpt(files)
    readme = hf_hub_download(repo_id, "README.md", token=token, force_download=True)
    card = _patch_card(Path(readme).read_text(), repo_id, ckpt)
    if dry_run:
        print(f"  [dry-run] {repo_id}  ·  ckpt={ckpt}  ·  patched OK ({len(card)} chars)")
        return ""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _stage_inference_bundle(d)                 # loader + requirements + vjepa2_src/  (from persistent sources)
        (d / "README.md").write_text(card)
        url = upload_folder(repo_id=repo_id, folder_path=str(d), token=token,
                            ignore_patterns=["*.pyc", "__pycache__/*"],
                            commit_message="Self-contained: vendored V-JEPA2 arch + load_factorjepa.py + "
                                           "requirements + corrected card (arch/attribution)")
    print(f"  ✅ {repo_id}  →  {url}")
    return url


def main():
    ap = argparse.ArgumentParser(description="Retrofit already-pushed FactorJEPA repos to self-contained.")
    ap.add_argument("repos", nargs="+", help="HF repo ids, e.g. anon-org/factorjepa-pretrain-vjepa21-vitG-2B-poc")
    ap.add_argument("--dry-run", action="store_true", help="patch + validate, no upload")
    a = ap.parse_args()
    token = _get_token()
    if not token:
        sys.exit("FATAL: HF_TOKEN missing in .env")
    for r in a.repos:
        retrofit_repo(r, token, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
