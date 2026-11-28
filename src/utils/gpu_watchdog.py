"""
GPU utilization watchdog — emails alert if GPU drops below threshold for N minutes.
Run as background process alongside the pipeline.

Also alerts when free disk on --disk-path drops below pipeline.yaml gpu.watchdog.min_free_disk_gb.
Needs in the repo-root .env (never in the repo): GMAIL_APP_PASSWORD, GMAIL_SENDER_EMAIL, GPU_ALERT_EMAIL.

USAGE (on the GPU box, its own tmux session so it outlives every job):
    tmux new -d -s fj_watchdog "source venv_walkindia/bin/activate && python -u src/utils/gpu_watchdog.py --disk-path . --logs-dir logs --done-file logs/gpu_queue_done.txt --alerts-file logs/gpu_watchdog_alerts.log --test-email 2>&1 | tee -a logs/gpu_watchdog_$(date +%Y%m%d_%H%M%S).log"
    python src/utils/gpu_watchdog.py --disk-path . --logs-dir logs --done-file logs/gpu_queue_done.txt --alerts-file logs/gpu_watchdog_alerts.log --dry-run   # no e-mail
"""
import argparse
import os
import shutil
import smtplib
import subprocess
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path


def get_gpu_utilization():
    """Get GPU utilization % via nvidia-smi. Returns -1 on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(result.stdout.strip().split("\n")[0])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return -1


def get_gpu_info():
    """Get GPU name, VRAM, temperature for alert context."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "unknown"


def get_running_process():
    """Get the GPU process name."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.stdout.strip() else "No GPU process"
    except subprocess.TimeoutExpired:
        return "unknown"


def send_alert(to_email, subject, body, gmail_password, from_email):
    """Send email alert via Gmail SMTP.

    iter16 (2026-05-20): removed `from_email="..."` default per CLAUDE.md
    "No hardcoded values in Python". Caller passes from_email explicitly
    (resolved from .env GMAIL_SENDER_EMAIL or CLI flag).
    """
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, gmail_password)
        server.sendmail(from_email, to_email, msg.as_string())


def main():
    parser = argparse.ArgumentParser(description="GPU utilization watchdog with email alerts")
    # iter18 H2: defaults live in pipeline.yaml gpu.watchdog.* (None → yaml).
    parser.add_argument("--threshold", type=int, default=None,
                        help="Alert if GPU util < threshold%% for --window minutes "
                             "(default: pipeline.yaml gpu.watchdog.threshold_pct)")
    parser.add_argument("--window", type=int, default=None,
                        help="Minutes of sustained low util before alert "
                             "(default: pipeline.yaml gpu.watchdog.window_min)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Seconds between GPU checks (default: pipeline.yaml gpu.watchdog.interval_s)")
    parser.add_argument("--cooldown", type=int, default=None,
                        help="Minutes between repeat alerts (default: pipeline.yaml gpu.watchdog.cooldown_min)")
    parser.add_argument("--email", type=str, default=None,
                        help="Recipient address. Omitted → .env GPU_ALERT_EMAIL (FATAL if neither): the "
                             "address is personal data and stays out of the public repo and shell history.")
    parser.add_argument("--disk-path", type=str, required=True,
                        help="Filesystem to watch for free space (the repo root on the box).")
    parser.add_argument("--logs-dir", type=str, required=True,
                        help="Folder of run logs: a hang = no file written for gpu.watchdog.stale_log_min while a GPU job runs.")
    parser.add_argument("--done-file", type=str, required=True,
                        help="Queue exit log (lines 'EXIT <code> <name> <HH:MM>'); a non-zero code is a crash alert.")
    parser.add_argument("--alerts-file", type=str, required=True,
                        help="Every alert is also appended here ('<UTC time> <subject>'), so a reader that cannot "
                             "get e-mail (the Claude session's 15-min check) still sees it.")
    parser.add_argument("--test-email", action="store_true",
                        help="Send one 'watchdog started' e-mail at start so the alert path is proven.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print alerts instead of emailing")
    args = parser.parse_args()

    # iter18 H2: None → pipeline.yaml gpu.watchdog.* (single source).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.config import get_pipeline_config
    _wd = get_pipeline_config()["gpu"]["watchdog"]
    if args.threshold is None:
        args.threshold = _wd["threshold_pct"]
    if args.window is None:
        args.window = _wd["window_min"]
    if args.interval is None:
        args.interval = _wd["interval_s"]
    if args.cooldown is None:
        args.cooldown = _wd["cooldown_min"]

    # Load Gmail credentials from .env (sender email + app password live there
    # alongside other secrets — NEVER hardcoded in src/*.py per CLAUDE.md).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed, checking env directly")

    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    gmail_sender   = os.getenv("GMAIL_SENDER_EMAIL")
    if not gmail_password and not args.dry_run:
        print("FATAL: GMAIL_APP_PASSWORD not found in .env")
        print("Generate at: https://myaccount.google.com/apppasswords")
        print("Add to .env: GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx")
        sys.exit(1)
    if not gmail_sender and not args.dry_run:
        print("FATAL: GMAIL_SENDER_EMAIL not found in .env")
        print("Add to .env: GMAIL_SENDER_EMAIL=you@example.com")
        sys.exit(1)
    if args.email is None:
        args.email = os.getenv("GPU_ALERT_EMAIL")
    if not args.email:
        print("FATAL: no recipient: pass --email or add GPU_ALERT_EMAIL=you@example.com to .env")
        sys.exit(1)
    min_free_gb = _wd["min_free_disk_gb"]
    stale_min = _wd["stale_log_min"]
    done_file = Path(args.done_file)
    done_seen = len(done_file.read_text().splitlines()) if done_file.exists() else 0
    last_stale_alert = 0

    hostname = os.uname().nodename
    checks_needed = max(1, (args.window * 60) // args.interval)
    low_count = 0
    last_alert_time = 0
    last_disk_alert = 0

    def deliver(subject, body):
        with open(args.alerts_file, "a") as f:
            f.write(f"{datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC {subject}\n")
        if args.dry_run:
            print(f"\n{'='*60}\nDRY RUN: {subject}\n{body}\n{'='*60}\n")
            return
        try:
            send_alert(args.email, subject, body, gmail_password, from_email=gmail_sender)
            print(f"[{datetime.now():%H:%M:%S}] ALERT SENT: {subject}")
        except (smtplib.SMTPException, OSError) as e:
            print(f"[{datetime.now():%H:%M:%S}] EMAIL FAILED: {type(e).__name__}: {e}")

    if args.test_email:
        deliver(f"[GPU WATCHDOG] started on {hostname}",
                f"Watchdog running: alert if GPU < {args.threshold}% for {args.window}+ min "
                f"or free disk < {min_free_gb} GB on {args.disk_path}.\n"
                f"Now: GPU {get_gpu_utilization()}% | {get_gpu_info()} | {get_running_process()}\n")

    print(f"GPU Watchdog: threshold={args.threshold}%, window={args.window}min, "
          f"interval={args.interval}s, cooldown={args.cooldown}min")
    print(f"Alert: {args.email} | Checks for alert: {checks_needed} consecutive")

    while True:
        util = get_gpu_utilization()
        free_gb = shutil.disk_usage(args.disk_path).free / 1e9
        if free_gb < min_free_gb and time.time() - last_disk_alert > args.cooldown * 60:
            deliver(f"[DISK ALERT] {hostname}: {free_gb:.0f} GB free on {args.disk_path}",
                    f"Free disk {free_gb:.1f} GB < {min_free_gb} GB. Training writes ~12 GB per arm; "
                    f"a full disk kills runs silently.\nGPU {util}% | {get_running_process()}\n")
            last_disk_alert = time.time()

        # Crash: a queue job that exited non-zero since the last check.
        if done_file.exists():
            lines = done_file.read_text().splitlines()
            for line in lines[done_seen:]:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "EXIT" and parts[1] != "0":
                    deliver(f"[JOB FAILED] {hostname}: {parts[2]} exit {parts[1]}",
                            f"{line}\nCheck: ls -t {args.logs_dir} | head; tmux ls\n")
            done_seen = len(lines)

        # Hang: a GPU process is running but no log was written for stale_min minutes.
        process = get_running_process()
        logs = [p for p in Path(args.logs_dir).iterdir() if p.is_file()]
        newest = max((p.stat().st_mtime for p in logs), default=0.0)
        idle_min = (time.time() - newest) / 60
        if (process != "No GPU process" and idle_min > stale_min
                and time.time() - last_stale_alert > args.cooldown * 60):
            deliver(f"[STALE LOG] {hostname}: no log written for {idle_min:.0f} min, GPU job still running",
                    f"GPU {util}% | {process}\nNewest log is {idle_min:.0f} min old in {args.logs_dir}: "
                    f"the job may be hung.\n")
            last_stale_alert = time.time()

        if util < 0:
            time.sleep(args.interval)
            continue

        if util < args.threshold:
            low_count += 1
            if low_count >= checks_needed:
                now = time.time()
                if now - last_alert_time > args.cooldown * 60:
                    gpu_info = get_gpu_info()
                    process = get_running_process()

                    subject = f"[GPU ALERT] {hostname}: GPU at {util}% for {args.window}+ min"
                    body = (
                        f"GPU utilization below {args.threshold}% for {args.window}+ minutes.\n\n"
                        f"Host:    {hostname}\n"
                        f"Time:    {datetime.now()}\n"
                        f"GPU:     {util}%\n"
                        f"Info:    {gpu_info}\n"
                        f"Process: {process}\n\n"
                        f"Free disk: {free_gb:.0f} GB\n"
                        f"Check: ls -t logs | head; tmux ls; nvidia-smi\n"
                    )
                    deliver(subject, body)
                    last_alert_time = now
                    low_count = 0
        else:
            if low_count > 0:
                print(f"[{datetime.now():%H:%M:%S}] GPU recovered: {util}% "
                      f"(was low for {low_count * args.interval}s)")
            low_count = 0

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
