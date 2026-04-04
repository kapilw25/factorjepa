"""
GPU utilization watchdog — emails alert if GPU drops below threshold for N minutes.
Run as background process alongside the pipeline.

USAGE:
    python scripts/gpu_watchdog.py &                    # default: 50% threshold, 5 min window
    python scripts/gpu_watchdog.py --threshold 30 &     # custom threshold
    python scripts/gpu_watchdog.py --dry-run             # test without sending email
"""
import argparse
import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText


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


def send_alert(to_email, subject, body, gmail_password, from_email="kapilw25@gmail.com"):
    """Send email alert via Gmail SMTP."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, gmail_password)
        server.sendmail(from_email, to_email, msg.as_string())


def main():
    parser = argparse.ArgumentParser(description="GPU utilization watchdog with email alerts")
    parser.add_argument("--threshold", type=int, default=50,
                        help="Alert if GPU util < threshold%% for --window minutes (default: 50)")
    parser.add_argument("--window", type=int, default=5,
                        help="Minutes of sustained low util before alert (default: 5)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between GPU checks (default: 30)")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Minutes between repeat alerts (default: 30)")
    parser.add_argument("--email", type=str, default="kapilw25@gmail.com")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print alerts instead of emailing")
    args = parser.parse_args()

    # Load Gmail app password from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed, checking env directly")

    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail_password and not args.dry_run:
        print("FATAL: GMAIL_APP_PASSWORD not found in .env")
        print("Generate at: https://myaccount.google.com/apppasswords")
        print("Add to .env: GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx")
        sys.exit(1)

    hostname = os.uname().nodename
    checks_needed = max(1, (args.window * 60) // args.interval)
    low_count = 0
    last_alert_time = 0

    print(f"GPU Watchdog: threshold={args.threshold}%, window={args.window}min, "
          f"interval={args.interval}s, cooldown={args.cooldown}min")
    print(f"Alert: {args.email} | Checks for alert: {checks_needed} consecutive")

    while True:
        util = get_gpu_utilization()

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
                        f"Check: tail -f logs/ch9_full*.log\n"
                        f"nvtop or nvidia-smi for live GPU status\n"
                    )

                    if args.dry_run:
                        print(f"\n{'='*60}")
                        print(f"DRY RUN: {subject}")
                        print(body)
                        print(f"{'='*60}\n")
                    else:
                        try:
                            send_alert(args.email, subject, body, gmail_password)
                            print(f"[{datetime.now():%H:%M:%S}] ALERT SENT: GPU {util}% → {args.email}")
                        except Exception as e:
                            print(f"[{datetime.now():%H:%M:%S}] EMAIL FAILED: {e}")

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
