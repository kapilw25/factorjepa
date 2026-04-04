"""
Shared evaluation suite: m06 → m06b → m07 → m08 per encoder, then m08b compare.
Called by run_frozen.sh, run_pretrain.sh, run_surgery.sh as a single step.

USAGE:
    # Ch9 frozen encoders (5 encoders)
    python -u src/utils/eval_suite.py --encoders vjepa,random,dinov2,clip,vjepa_shuffled \
        --FULL --no-wandb 2>&1 | tee logs/eval_suite_ch9.log

    # Ch10 adapted encoders (1 encoder, compare all 7)
    python -u src/utils/eval_suite.py --encoders vjepa_lambda0_001 \
        --compare-encoders vjepa,random,dinov2,clip,vjepa_shuffled,vjepa_lambda0_001,vjepa_lambda0_001_shuffled \
        --FULL --no-wandb 2>&1 | tee logs/eval_suite_ch10.log
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.progress import make_pbar
from utils.config import add_subset_arg, get_output_dir


# ── Step definitions ──────────────────────────────────────────────────

EVAL_STEPS = [
    {"name": "m06", "script": "src/m06_faiss_metrics.py", "desc": "FAISS metrics"},
    {"name": "m06b", "script": "src/m06b_temporal_corr.py", "desc": "temporal correlation"},
    {"name": "m07", "script": "src/m07_umap.py", "desc": "UMAP"},
    {"name": "m08", "script": "src/m08_plot.py", "desc": "plots"},
]


def _run_one(script: str, args: list, log_file: str) -> tuple:
    """Run a single Python script as subprocess. Returns (success, elapsed_sec)."""
    cmd = [sys.executable, "-u", script] + args
    print(f"\n  CMD: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


def run_eval_suite(encoders: list, mode_flag: str, subset_flag: list,
                   no_wandb: bool, log_dir: str, log_prefix: str,
                   compare_encoders: list = None,
                   true_overlap_encoder: str = None,
                   skip_m06: bool = False):
    """Run m06→m06b→m07→m08 for each encoder, then m08b for compare list."""
    results = []  # (step_name, encoder, success, elapsed)
    total_t0 = time.time()

    wandb_flag = ["--no-wandb"] if no_wandb else []

    total_steps = len(encoders) * len(EVAL_STEPS) + 1  # +1 for m08b
    pbar = make_pbar(total=total_steps, desc="eval_suite", unit="step")

    # Per-encoder evaluation loop
    for enc in encoders:
        print(f"\n{'='*60}")
        print(f"  EVAL SUITE: {enc}")
        print(f"{'='*60}")

        for step in EVAL_STEPS:
            if skip_m06 and step["name"] == "m06":
                print(f"  SKIP: {step['name']} {step['desc']} ({enc}) — already computed")
                results.append((step["name"], enc, True, 0))
                continue

            args = ["--encoder", enc, mode_flag] + subset_flag + wandb_flag

            # m06: add --true-overlap for the designated encoder
            if step["name"] == "m06" and true_overlap_encoder and enc == true_overlap_encoder:
                args.append("--true-overlap")

            log_file = f"{log_dir}/{step['name']}_{log_prefix}_{enc}.log"
            print(f"\n  --- {step['name']} {step['desc']} ({enc}) ---")
            success, elapsed = _run_one(step["script"], args, log_file)

            status = "PASSED" if success else "FAILED"
            print(f"  {status}: {step['name']} {enc} ({elapsed:.0f}s)")
            results.append((step["name"], enc, success, elapsed))
            pbar.update(1)

    # m08b: compare all encoders
    compare_list = compare_encoders or encoders
    print(f"\n{'='*60}")
    print(f"  m08b COMPARE: {','.join(compare_list)}")
    print(f"{'='*60}")

    m08b_args = ["--encoders", ",".join(compare_list), mode_flag] + subset_flag + wandb_flag
    log_file = f"{log_dir}/m08b_{log_prefix}.log"
    success, elapsed = _run_one("src/m08b_compare.py", m08b_args, log_file)
    status = "PASSED" if success else "FAILED"
    print(f"  {status}: m08b compare ({elapsed:.0f}s)")
    results.append(("m08b", "all", success, elapsed))
    pbar.update(1)
    pbar.close()

    # Summary
    total_elapsed = time.time() - total_t0
    n_pass = sum(1 for _, _, s, _ in results if s)
    n_fail = sum(1 for _, _, s, _ in results if not s)

    print(f"\n{'='*60}")
    print(f"  EVAL SUITE COMPLETE")
    print(f"  Encoders: {', '.join(encoders)}")
    print(f"  Steps: {n_pass} passed, {n_fail} failed, {len(results)} total")
    print(f"  Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    if n_fail > 0:
        print(f"\n  FAILURES:")
        for name, enc, success, _ in results:
            if not success:
                print(f"    {name} ({enc})")

    return n_fail == 0


def main():
    parser = argparse.ArgumentParser(
        description="Shared evaluation suite: m06→m06b→m07→m08 per encoder, m08b compare")
    parser.add_argument("--encoders", required=True,
                        help="Comma-separated encoder names to evaluate")
    parser.add_argument("--compare-encoders", default=None,
                        help="Comma-separated encoder names for m08b (default: same as --encoders)")
    parser.add_argument("--true-overlap-encoder", default=None,
                        help="Encoder to pass --true-overlap to m06 (default: none)")
    parser.add_argument("--skip-m06", action="store_true",
                        help="Skip m06 step (if already computed by caller)")
    parser.add_argument("--SANITY", action="store_true")
    parser.add_argument("--POC", action="store_true")
    parser.add_argument("--FULL", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-prefix", default="eval")
    add_subset_arg(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    mode_flag = "--SANITY" if args.SANITY else ("--POC" if args.POC else "--FULL")
    subset_flag = ["--subset", args.subset] if args.subset else []
    encoders = [e.strip() for e in args.encoders.split(",")]
    compare_encoders = ([e.strip() for e in args.compare_encoders.split(",")]
                        if args.compare_encoders else None)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    success = run_eval_suite(
        encoders=encoders,
        mode_flag=mode_flag,
        subset_flag=subset_flag,
        no_wandb=args.no_wandb,
        log_dir=args.log_dir,
        log_prefix=args.log_prefix,
        compare_encoders=compare_encoders,
        true_overlap_encoder=args.true_overlap_encoder,
        skip_m06=args.skip_m06,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
