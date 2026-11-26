"""Resolve HF upload mode (delete-then-reupload vs reuse) for hf_outputs.py.

iter17 (2026-05-27): at the start of `upload` / `upload-data`, ask the user
whether to DELETE the remote folder before re-uploading (clean slate) or REUSE
it (re-upload over existing; HF dedups unchanged LFS objects). Generalizes the
manual scoped-delete used to wipe data/full_local. Mirrors utils.cache_policy's
interactive-with-env-bypass pattern so overnight/tmux runs stay non-interactive.

PUBLIC API
    resolve_upload_mode_interactive(api, repo_id, repo_type, path_in_repo, target_desc) -> "delete"|"reuse"
    delete_repo_folder_scoped(api, repo_id, repo_type, path_in_repo)   # sibling-safe wipe
"""
import os
import sys


def _count_remote(api, repo_id, repo_type, path_in_repo) -> int:
    """Number of repo files under path_in_repo/ (exact-file or prefix match)."""
    prefix = path_in_repo.rstrip("/") + "/"
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    return sum(1 for f in files if f == path_in_repo or f.startswith(prefix))


def resolve_upload_mode_interactive(api, repo_id, repo_type, path_in_repo,
                                    target_desc) -> str:
    """Return 'delete' or 'reuse' for the upload of path_in_repo/.

    - Remote empty (first upload) -> 'reuse' silently (nothing to delete).
    - Else honor HF_UPLOAD_MODE env (delete|reuse) for tmux/overnight runs.
    - Else prompt interactively.
    - FAIL LOUD if non-interactive (no tty) and no env override — never guess
      a destructive 'delete' nor silently 'reuse' over a populated remote.
    """
    n_remote = _count_remote(api, repo_id, repo_type, path_in_repo)
    if n_remote == 0:
        print(f"  [upload-mode] remote '{path_in_repo}/' is empty → fresh upload")
        return "reuse"

    env = os.environ.get("HF_UPLOAD_MODE", "").strip().lower()
    if env in ("delete", "reuse"):
        print(f"  [upload-mode] HF_UPLOAD_MODE={env} (env override) for "
              f"'{path_in_repo}/' ({n_remote} remote files)")
        return env

    if not sys.stdin.isatty():
        sys.exit(f"FATAL: non-interactive upload of '{path_in_repo}/' "
                 f"({n_remote} remote files) needs HF_UPLOAD_MODE=delete|reuse "
                 f"(refusing to guess a destructive default)")

    print(f"\n  HF upload target: {target_desc}")
    print(f"    remote '{path_in_repo}/' already has {n_remote} files. Choose:")
    print("      1 = DELETE remote folder, then re-upload local (clean slate)")
    print("      2 = REUSE — re-upload over existing (HF dedups unchanged files)")
    ans = input("    [1=delete / 2=reuse] (Enter=2): ").strip()
    return "delete" if ans == "1" else "reuse"


def delete_repo_folder_scoped(api, repo_id, repo_type, path_in_repo) -> None:
    """Delete ONLY path_in_repo/ on the repo, verifying every file OUTSIDE that
    prefix is untouched. No-op if nothing is there. Raises (FAIL LOUD) if the
    delete left target files behind OR changed any sibling — a scoped delete
    must never touch siblings (the whole point of the user's safety requirement).
    """
    path_in_repo = path_in_repo.rstrip("/")
    prefix = path_in_repo + "/"
    before = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    target = {f for f in before if f == path_in_repo or f.startswith(prefix)}
    if not target:
        print(f"  [upload-mode] nothing to delete under '{path_in_repo}/'")
        return

    non_target_before = before - target
    print(f"  [upload-mode] deleting '{path_in_repo}/' ({len(target)} files); "
          f"{len(non_target_before)} sibling files must stay intact ...")
    api.delete_folder(
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"upload-mode: wipe {path_in_repo}/ before clean re-upload",
    )

    after = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    still = {f for f in after if f == path_in_repo or f.startswith(prefix)}
    if still:
        raise RuntimeError(
            f"scoped delete FAILED: {len(still)} files still under "
            f"'{path_in_repo}/' after delete_folder")
    if after != non_target_before:
        lost = sorted(non_target_before - after)[:5]
        gained = sorted(after - non_target_before)[:5]
        raise RuntimeError(
            f"scoped delete TOUCHED SIBLINGS for '{path_in_repo}/': "
            f"lost={lost} gained={gained}")
    print(f"  [upload-mode] OK — '{path_in_repo}/' wiped; all "
          f"{len(non_target_before)} sibling files intact")


__all__ = ["resolve_upload_mode_interactive", "delete_repo_folder_scoped"]
