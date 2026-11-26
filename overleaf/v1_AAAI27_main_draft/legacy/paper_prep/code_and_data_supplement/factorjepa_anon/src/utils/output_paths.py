"""output_paths — SINGLE SOURCE for the iter18 backbone-first output tree (2026-06-20).

Spec: iter/iter18_ablations_FTtechniues/plan_output_restructure.md. The 3-dial tree:

    outputs/<mode>/<backbone>_<size>/
    ├── train/   <arm>/student_encoder.pt        (dial 1=mode/scale · dial 2=backbone · ONE copy)
    └── eval/
        └── <corpus>/   <metric>/<arm>/ + plot/  (dial 3 = score corpus: eval_10k / subset_10k)

Every path-building site (run_train.sh, run_eval.sh, iter18_poc_ngpu.py, m13, status tool) MUST route
through this module — py callers import the functions, bash callers use the CLI — so the tree can never
drift. Values come from configs/pipeline.yaml (output_roots / backbone_size_labels / eval_corpora); an
unknown mode / backbone / corpus FATALs (no silent default — a wrong path = garbage metrics).

CLI (bash THIN wrappers):
    python -u src/utils/output_paths.py bb-dir     <mode> <backbone>
    python -u src/utils/output_paths.py train-dir  <mode> <backbone>
    python -u src/utils/output_paths.py eval-root  <mode> <backbone> <corpus>
    python -u src/utils/output_paths.py eval-dir    <mode> <backbone> <corpus> <metric_family>
    python -u src/utils/output_paths.py plot-dir   <mode> <backbone> <corpus>
    python -u src/utils/output_paths.py selftest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path so `utils.*` resolves when run as a bash CLI
from utils.config import get_pipeline_config  # noqa: E402


def _cfg():
    return get_pipeline_config()


def output_root(mode: str) -> str:
    """outputs/<mode> from pipeline.yaml output_roots (FATAL on unknown mode)."""
    roots = _cfg()["output_roots"]
    m = mode.lower()
    if m not in roots:
        sys.exit(f"FATAL [output_paths]: unknown mode '{mode}' — known: {sorted(roots)}")
    return roots[m]


def size_label(backbone: str) -> str:
    """The eyeball-clear size suffix (2B/1B/…) from pipeline.yaml backbone_size_labels (FATAL if unlisted)."""
    labels = _cfg()["backbone_size_labels"]
    if backbone not in labels:
        sys.exit(f"FATAL [output_paths]: backbone '{backbone}' missing from "
                 f"pipeline.yaml backbone_size_labels — known: {sorted(labels)}")
    return labels[backbone]


def _check_corpus(corpus: str) -> str:
    corpora = _cfg()["eval_corpora"]
    if corpus not in corpora:
        sys.exit(f"FATAL [output_paths]: corpus '{corpus}' not in pipeline.yaml eval_corpora {corpora}")
    return corpus


def bb_dir(mode: str, backbone: str) -> str:
    """outputs/<mode>/<backbone>_<size> — the per-(scale,backbone) root."""
    return f"{output_root(mode)}/{backbone}_{size_label(backbone)}"


def train_dir(mode: str, backbone: str) -> str:
    """outputs/<mode>/<backbone>_<size>/train — the encoders (corpus-INDEPENDENT; every corpus reads here)."""
    return f"{bb_dir(mode, backbone)}/train"


def eval_root(mode: str, backbone: str, corpus: str) -> str:
    """outputs/<mode>/<backbone>_<size>/eval/<corpus> — the per-corpus eval root."""
    return f"{bb_dir(mode, backbone)}/eval/{_check_corpus(corpus)}"


def eval_dir(mode: str, backbone: str, corpus: str, metric_family: str) -> str:
    """…/eval/<corpus>/<metric_family> — one metric family's dir (probe_action, predictor_temporal, …)."""
    return f"{eval_root(mode, backbone, corpus)}/{metric_family}"


# The E: (Stage 2-8 head-metric) bundle's 4 result files — the SINGLE 'is E: done' marker set, shared by the
# ngpu_run resume-skip AND the status pane's 2-8 cell so the two can't drift (iter19 2026-07-08). A resume-
# skipped E: emits no log ◀/✓, so the pane must check the outputs directly or it shows a finished arm as pending.
_HEAD_STAGE_MARKERS = (("probe_action", "test_metrics.json"), ("probe_taxonomy", "test_metrics.json"),
                       ("probe_motion_cos", "intra_inter_ratio.json"), ("probe_future_mse", "aggregate_mse.json"))


def head_stage_done(mode: str, backbone: str, corpus: str, enc: str) -> bool:
    """True iff the per-encoder E: bundle produced ALL 4 head-metric result files (action + taxonomy
    test_metrics.json · motion_cos intra_inter_ratio.json · future_mse aggregate_mse.json)."""
    root = Path(__file__).resolve().parents[2]
    return all((root / eval_dir(mode, backbone, corpus, art) / enc / fn).exists()
               for art, fn in _HEAD_STAGE_MARKERS)


def plot_dir(mode: str, backbone: str, corpus: str) -> str:
    """…/eval/<corpus>/plot — m13's figures for this (backbone × corpus)."""
    return f"{eval_root(mode, backbone, corpus)}/plot"


def _selftest() -> None:
    bb, mode, corpus = "vjepa_2_1_vitG", "poc", "eval_10k"
    assert bb_dir(mode, bb) == "outputs/poc/vjepa_2_1_vitG_2B", bb_dir(mode, bb)
    assert train_dir(mode, bb) == "outputs/poc/vjepa_2_1_vitG_2B/train", train_dir(mode, bb)
    assert eval_root(mode, bb, corpus) == "outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k", eval_root(mode, bb, corpus)
    assert eval_dir(mode, bb, corpus, "predictor_temporal") == \
        "outputs/poc/vjepa_2_1_vitG_2B/eval/eval_10k/predictor_temporal"
    assert plot_dir(mode, bb, "subset_10k") == "outputs/poc/vjepa_2_1_vitG_2B/eval/subset_10k/plot"
    assert bb_dir("full", "vjepa_2_1_vitg") == "outputs/full/vjepa_2_1_vitg_1B", bb_dir("full", "vjepa_2_1_vitg")
    # FAIL-LOUD on the three unknown-key paths (each must SystemExit) — explicit flag, no bare except/pass.
    for bad in (lambda: bb_dir("nope", bb), lambda: size_label("nope"), lambda: eval_root(mode, bb, "nope")):
        raised = False
        try:
            bad()
        except SystemExit:
            raised = True
        if not raised:
            sys.exit("FATAL [output_paths selftest]: expected SystemExit on an unknown key")
    print("output_paths selftest: PASS (5 path verbs + 3 FAIL-LOUD guards)")


def _cli(argv) -> None:
    if not argv:
        sys.exit("usage: output_paths.py {bb-dir|train-dir} <mode> <backbone> "
                 "| {eval-root|plot-dir} <mode> <backbone> <corpus> "
                 "| eval-dir <mode> <backbone> <corpus> <metric_family> | selftest")
    cmd, *rest = argv
    if cmd == "selftest":
        _selftest()
    elif cmd == "bb-dir":
        print(bb_dir(*rest))
    elif cmd == "train-dir":
        print(train_dir(*rest))
    elif cmd == "eval-root":
        print(eval_root(*rest))
    elif cmd == "eval-dir":
        print(eval_dir(*rest))
    elif cmd == "plot-dir":
        print(plot_dir(*rest))
    else:
        sys.exit(f"FATAL [output_paths]: unknown command '{cmd}'")


if __name__ == "__main__":
    _cli(sys.argv[1:])
