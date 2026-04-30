#!/usr/bin/env python3
"""Quick post preflight checks aligned with BLOG_POST_PROTOCOL.md."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path


def parse_front_matter(text: str):
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1)
    data = {}
    current = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if re.match(r"^[A-Za-z_]+:\s*", line):
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip().strip('"')
            current = key.strip()
        elif current == "image" and ":" in line:
            key, val = line.strip().split(":", 1)
            data[f"image.{key.strip()}"] = val.strip().strip('"')
    return data


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 scripts/post_preflight.py <post.md>")

    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    fm = parse_front_matter(text)
    if not fm:
        raise SystemExit("❌ Invalid or missing front matter")

    errors = []
    warnings = []

    date_str = fm.get("date", "")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S +0900")
        now = datetime.now()
        if dt > now:
            errors.append(f"future-dated post: {date_str}")
    except ValueError:
        warnings.append(f"could not parse date strictly: {date_str}")

    if "```mermaid" in text and fm.get("mermaid", "false").lower() != "true":
        warnings.append("contains mermaid block but front matter mermaid is not true")

    if fm.get("mermaid", "false").lower() == "true" and "```mermaid" not in text:
        warnings.append("mermaid enabled but no mermaid block found")

    if "|---" in text:
        warnings.append("contains markdown table, verify mobile/rendering safety")

    if not fm.get("image.path"):
        warnings.append("missing front matter image.path")

    desc = fm.get("description", "")
    if len(desc) < 80:
        warnings.append(f"description looks short ({len(desc)} chars)")

    print(f"Preflight: {path.name}")
    for e in errors:
        print(f"❌ {e}")
    for w in warnings:
        print(f"⚠️  {w}")

    if errors:
        raise SystemExit(1)
    print("✅ preflight passed")


if __name__ == "__main__":
    main()
