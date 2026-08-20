"""Fast high-signal secret check; Gitleaks remains the release-grade scanner."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[opurs]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "stripe_live_key": re.compile(r"\b[rs]k_live_[A-Za-z0-9]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "supabase_service_jwt": re.compile(
        r"SUPABASE_SERVICE_ROLE_KEY\s*=\s*eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"
    ),
    "credentialed_database_url": re.compile(
        r"\b(?:postgres(?:ql)?|mysql)://[^\s:/]+:[^\s@]+@[^\s]+",
        re.IGNORECASE,
    ),
}

BINARY_SUFFIXES = {
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pyc",
    ".ttf",
    ".woff",
    ".woff2",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line and Path(line).suffix.lower() not in BINARY_SUFFIXES]


def main() -> int:
    findings: list[tuple[Path, int, str]] = []
    for path in tracked_files():
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((path, line_number, label))

    if findings:
        print("FAIL: potential credentials found (values redacted):")
        for path, line_number, label in findings:
            print(f"- {path}:{line_number}: {label}")
        return 1
    print("PASS: no high-signal credential pattern found in the current tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
