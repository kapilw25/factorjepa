"""GPU job queue: keep the card busy without overfilling it (ported 2026-11-28 from the SwiPO box queue, cita_ecliptica
scripts/gpu_box/queue.sh, to Python per src/CLAUDE.md). CPU-side orchestrator; it never runs GPU code itself.

Reads --jobs one line at a time, "name|command" (blank lines and lines starting with '#' are skipped), and remembers its
position in --pos, so lines appended later are picked up and a restart resumes where it stopped. Each line starts in its
own tmux session (named after the job) once fewer than gpu.queue.max_jobs GPU jobs run AND nvidia-smi memory.free is at
least gpu.queue.train_free_gb (names starting "train_") or eval_free_gb (all others); after a start it sleeps
settle_train_s / settle_eval_s so the job allocates before the next check. A name starting "wait_" runs in the
foreground and holds the queue until it ends (dependencies, e.g. an eval chain waiting for its training). Every job
appends "EXIT <code> <name> <HH:MM UTC>" to --done; a failed job does not stop the ones after it.
GPU jobs are counted as top-level `python -u src/...` processes (DataLoader workers share the argv and are skipped by
their parent; under CUDA MPS nvidia-smi shows all clients as one server, so it cannot count them).

USAGE (thin wrapper scripts/gpu_queue.sh; run inside tmux on the box):
    python -u src/utils/gpu_queue.py --jobs logs/gpu_queue_jobs.txt --pos logs/gpu_queue_pos \
        --done logs/gpu_queue_done.txt --env-file data/heldout_video/rv_mps.env --venv venv_walkindia --logs-dir logs
    python -u src/utils/gpu_queue.py --selftest
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import get_pipeline_config  # noqa: E402


def parse_line(line: str):
    """'name|command' -> (name, command); None for blank/comment lines. A line without '|' is an error (fail loud)."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if "|" not in s:
        raise ValueError(f"gpu_queue: job line has no '|': {s!r}")
    name, cmd = s.split("|", 1)
    name, cmd = name.strip(), cmd.strip()
    if not name or not cmd or any(c in name for c in " :.'\""):
        raise ValueError(f"gpu_queue: bad job name {name!r} (tmux session names: no spaces, ':', '.', quotes)")
    return name, cmd


def gpu_jobs() -> int:
    """Top-level `python -u src/...` processes (a DataLoader worker's parent is itself such a process: skipped)."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, check=True).stdout
    return count_gpu_jobs(out)


def count_gpu_jobs(ps_out: str) -> int:
    procs = {}
    for row in ps_out.splitlines():
        f = row.split()
        if len(f) >= 5 and f[2].endswith(("python", "python3")) and f[3] == "-u" and f[4].startswith("src/"):
            procs[f[0]] = f[1]
    return sum(1 for pid, ppid in procs.items() if ppid not in procs)


def free_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split()
    return float(out[0]) / 1024


def tmux_cmd(name: str, cmd: str, done: Path, venv: str, env_file: str, logs_dir: str) -> list:
    """The job runs in a subshell whose output is tee'd to <logs_dir>/<name>_<UTC stamp>.log; with pipefail the EXIT
    line carries the job's own exit code, not tee's."""
    log = f"{shlex.quote(logs_dir)}/{name}_$(date -u +%Y%m%d_%H%M%S).log"
    inner = (f"source {shlex.quote(venv)}/bin/activate && "
             + (f"source {shlex.quote(env_file)} && " if env_file else "")
             + "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && set -o pipefail && "
             + f"( {cmd} ) 2>&1 | tee {log}; echo \"EXIT $? {name} $(date -u +%H:%M)\" >> {shlex.quote(str(done))}")
    return ["tmux", "new-session", "-d", "-s", name, "bash", "-c", inner]


def run(args, qcfg):
    jobs, pos_f, done = Path(args.jobs), Path(args.pos), Path(args.done)
    for f in (jobs, done):
        f.touch()
    i = int(pos_f.read_text()) if pos_f.exists() else 0
    print(f"{time.strftime('%H:%M', time.gmtime())} queue up at line {i}: max_jobs={qcfg['max_jobs']} "
          f"train>={qcfg['train_free_gb']} GB eval>={qcfg['eval_free_gb']} GB", flush=True)
    while True:
        lines = jobs.read_text().splitlines()
        if i >= len(lines):
            time.sleep(qcfg["poll_s"])
            continue
        job = parse_line(lines[i])
        if job is not None:
            name, cmd = job
            if name.startswith("wait_"):
                print(f"{time.strftime('%H:%M', time.gmtime())} holding queue: {name}", flush=True)
                rc = subprocess.run(["bash", "-c", cmd]).returncode
                with done.open("a") as fh:
                    fh.write(f"EXIT {rc} {name} {time.strftime('%H:%M', time.gmtime())}\n")
            else:
                train = name.startswith("train_")
                need = qcfg["train_free_gb"] if train else qcfg["eval_free_gb"]
                while gpu_jobs() >= qcfg["max_jobs"] or free_gb() < need:
                    time.sleep(qcfg["poll_s"])
                if subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0:
                    raise RuntimeError(f"gpu_queue: tmux session {name} already exists (duplicate job name?)")
                subprocess.run(tmux_cmd(name, cmd, done, args.venv, args.env_file, args.logs_dir), check=True)
                print(f"{time.strftime('%H:%M', time.gmtime())} started {name} (needs {need} GB, "
                      f"{free_gb():.0f} GB free, {gpu_jobs()} GPU jobs)", flush=True)
                time.sleep(qcfg["settle_train_s"] if train else qcfg["settle_eval_s"])
        i += 1
        pos_f.write_text(str(i))


def selftest() -> int:
    ok = 0
    cases = [("a|echo hi", ("a", "echo hi")), ("# c", None), ("", None), ("train_x | bash run.sh --POC", ("train_x", "bash run.sh --POC")),
             ("wait_x|until grep -q y f; do sleep 1; done", ("wait_x", "until grep -q y f; do sleep 1; done"))]
    for line, want in cases:
        got = parse_line(line)
        ok += got == want
        print(f"{'ok ' if got == want else 'BAD'} {line!r} -> {got}")
    for bad in ("no pipe here", "bad name|x", "a.b|x"):
        try:
            parse_line(bad)
            print(f"BAD accepted {bad!r}")
        except ValueError:
            ok += 1
            print(f"ok  refused {bad!r}")
    cmd = tmux_cmd("train_x", "bash r.sh", Path("logs/d.txt"), "venv_walkindia", "e.env", "logs")
    want_tail = ('( bash r.sh ) 2>&1 | tee logs/train_x_$(date -u +%Y%m%d_%H%M%S).log; '
                 'echo "EXIT $? train_x $(date -u +%H:%M)" >> logs/d.txt')
    ok += cmd[:5] == ["tmux", "new-session", "-d", "-s", "train_x"] and cmd[-1].endswith(want_tail)
    print(f"{'ok ' if cmd[-1].endswith(want_tail) else 'BAD'} tmux command ends with the EXIT line")
    ps = ("10 1 /v/bin/python -u src/m09c1_surgery_encoder.py --POC\n11 10 /v/bin/python -u src/m09c1_surgery_encoder.py --POC\n"
          "12 10 /v/bin/python -u src/m09c1_surgery_encoder.py --POC\n20 1 /v/bin/python3 -u src/m09b_peft_encoder.py\n"
          "30 1 python -c pass\n")
    got = count_gpu_jobs(ps)
    ok += got == 2
    print(f"{'ok ' if got == 2 else 'BAD'} 2 trainers with 2 DataLoader workers count as {got}")
    n = len(cases) + 3 + 1 + 1
    print(f"selftest {ok} of {n}")
    return 0 if ok == n else 1


def main():
    if sys.argv[1:2] == ["--selftest"]:
        sys.exit(selftest())
    ap = argparse.ArgumentParser(description="GPU job queue (see module docstring)")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--venv", required=True, help="venv dir each job activates")
    ap.add_argument("--env-file", required=True, help="file each job sources first (CUDA MPS env), or '' for none")
    ap.add_argument("--logs-dir", required=True, help="each job's output is tee'd to <logs-dir>/<name>_<UTC stamp>.log")
    args = ap.parse_args()
    run(args, get_pipeline_config()["gpu"]["queue"])


if __name__ == "__main__":
    main()
