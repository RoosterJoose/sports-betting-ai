#!/usr/bin/env python3
"""Send the latest daily report via email.

Usage:
    # Send via local sendmail (works if Postfix/MTA is configured)
    python scripts/send-report.py

    # Send via SMTP (requires .env config — see below)
    python scripts/send-report.py --smtp

    # Send to a custom recipient
    python scripts/send-report.py --to you@example.com

Configuration (.env):
    # For sendmail mode (default) — no config needed if local MTA works
    # For SMTP mode (--smtp):
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your@gmail.com
    SMTP_PASSWORD=your-app-password
    REPORT_EMAIL_TO=you@example.com
    REPORT_EMAIL_FROM=sports-betting-ai@example.com
"""
import os, sys, subprocess, smtplib, re
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# ── Bootstrap .env ─────────────────────────────────────────────────────
_dotenv = Path(__file__).resolve().parent.parent / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
LATEST_LINK = REPORTS_DIR / "latest.txt"


def get_latest_report() -> Path | None:
    """Return the path to the most recent report file."""
    if LATEST_LINK.exists():
        target = LATEST_LINK.resolve()
        if target.exists():
            return target
    # Fallback: find most recent .txt file
    reports = sorted(REPORTS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def extract_tldr(report_text: str) -> str:
    """Extract the key summary from a report — top plays + parlays."""
    lines = report_text.split("\n")
    tldr_sections = []
    capture = False
    for line in lines:
        # Capture TOP PLAYS and beyond
        if "TOP PLAYS FOR TODAY" in line:
            capture = True
        if capture:
            tldr_sections.append(line)
    return "\n".join(tldr_sections) if tldr_sections else report_text[:5000]


def send_via_sendmail(to_addr: str, subject: str, body: str) -> bool:
    """Send email using /usr/sbin/sendmail (local MTA)."""
    sendmail_path = "/usr/sbin/sendmail"
    if not os.path.exists(sendmail_path):
        print("  sendmail not found at /usr/sbin/sendmail")
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["To"] = to_addr
    msg["From"] = os.environ.get("REPORT_EMAIL_FROM", "sports-betting-ai@localhost")

    try:
        proc = subprocess.run(
            [sendmail_path, "-t", to_addr],
            input=msg.as_string(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            print(f"  Sent via sendmail to {to_addr}")
            return True
        else:
            print(f"  sendmail failed (exit={proc.returncode}): {proc.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  sendmail error: {e}")
        return False


def send_via_smtp(to_addr: str, subject: str, body: str) -> bool:
    """Send email via SMTP (requires .env config)."""
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    from_addr = os.environ.get("REPORT_EMAIL_FROM", user or "sports-betting-ai@example.com")

    if not all([host, user, password]):
        print("  SMTP not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env")
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["To"] = to_addr
    msg["From"] = from_addr

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        print(f"  Sent via SMTP ({host}) to {to_addr}")
        return True
    except Exception as e:
        print(f"  SMTP error: {e}")
        return False


def send_email(to_addr: str, use_smtp: bool = False, dry_run: bool = False) -> bool:
    """Send the latest report via email."""
    report_path = get_latest_report()
    if not report_path:
        print("  No report found in reports/")
        return False

    report_text = report_path.read_text(encoding="utf-8")
    date_match = re.search(r"DAILY REPORT — ([\d\- :]+)", report_text)
    date_str = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")

    subject = f"Sports Betting AI — Daily Report ({date_str})"
    tldr = extract_tldr(report_text)

    # Build a concise body
    body_parts = [
        f"📊 Sports Betting AI — Daily Report",
        f"Date: {date_str}",
        f"",
        f"Full report: {report_path}",
        f"",
        f"{'=' * 60}",
        f"{tldr}",
    ]
    body = "\n".join(body_parts)

    if dry_run:
        print(f"  [DRY RUN] Would email {to_addr}")
        print(f"  Subject: {subject}")
        print(f"  Body preview: {body[:300]}...")
        return True

    if use_smtp:
        return send_via_smtp(to_addr, subject, body)
    else:
        # Try sendmail first, fallback to SMTP if configured
        if send_via_sendmail(to_addr, subject, body):
            return True
        if os.environ.get("SMTP_HOST"):
            print("  Falling back to SMTP...")
            return send_via_smtp(to_addr, subject, body)
        print("  No email delivery method available.")
        print("  Install Postfix or configure SMTP in .env (see header)")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Send daily report via email")
    parser.add_argument("--to", default=os.environ.get("REPORT_EMAIL_TO", ""),
                        help="Recipient email address")
    parser.add_argument("--smtp", action="store_true",
                        help="Use SMTP (default: try sendmail first)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be sent without sending")
    args = parser.parse_args()

    if not args.to:
        print("Error: No recipient. Set REPORT_EMAIL_TO in .env or pass --to")
        sys.exit(1)

    success = send_email(args.to, use_smtp=args.smtp, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
