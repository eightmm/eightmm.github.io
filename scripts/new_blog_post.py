#!/usr/bin/env python3
"""Create a new paper-post scaffold using the house protocol.

Example:
  python3 scripts/new_blog_post.py \
    --title "Generative Modeling via Drifting" \
    --slug generative-modeling-via-drifting \
    --description "Short summary" \
    --category AI --subcategory "Generative Models" \
    --tags generative-modeling,diffusion,imagenet \
    --paper-url https://arxiv.org/abs/2602.04770
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "templates" / "paper-post-template.md"
POSTS = ROOT / "_posts"
ASSETS = ROOT / "assets" / "img" / "posts"


def today_kst_string() -> str:
    # Keep simple and stable, local machine timezone is already KST in this setup.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S +0900")


def normalize_tags(tags: str) -> str:
    parts = [t.strip() for t in tags.split(",") if t.strip()]
    return ", ".join(parts)


def fill_template(template: str, mapping: dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Create a new blog post scaffold")
    p.add_argument("--title", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--category", default="AI")
    p.add_argument("--subcategory", default="General")
    p.add_argument("--tags", default="paper-review")
    p.add_argument("--date", default=None, help="Front matter date, default now in +0900")
    p.add_argument("--image-alt", default="Representative figure description")
    p.add_argument("--paper-title", default="")
    p.add_argument("--authors", default="")
    p.add_argument("--affiliations", default="")
    p.add_argument("--venue", default="")
    p.add_argument("--published", default="")
    p.add_argument("--paper-url", default="")
    p.add_argument("--project-url", default="")
    p.add_argument("--code-url", default="")
    args = p.parse_args()

    template = TEMPLATE.read_text(encoding="utf-8")
    date_prefix = (args.date or today_kst_string())[:10]
    post_path = POSTS / f"{date_prefix}-{args.slug}.md"
    asset_dir = ASSETS / args.slug

    if post_path.exists():
        raise SystemExit(f"Post already exists: {post_path}")

    asset_dir.mkdir(parents=True, exist_ok=True)

    filled = fill_template(template, {
        "TITLE": args.title,
        "DATE": args.date or today_kst_string(),
        "DESCRIPTION": args.description,
        "MAIN_CATEGORY": args.category,
        "SUBCATEGORY": args.subcategory,
        "TAGS": normalize_tags(args.tags),
        "SLUG": args.slug,
        "IMAGE_ALT": args.image_alt,
        "PAPER_TITLE": args.paper_title or args.title,
        "AUTHORS": args.authors or "TBD",
        "AFFILIATIONS": args.affiliations or "TBD",
        "VENUE": args.venue or "TBD",
        "PUBLISHED": args.published or "TBD",
        "PAPER_URL": args.paper_url or "TBD",
        "PROJECT_URL": args.project_url or "TBD",
        "CODE_URL": args.code_url or "TBD",
    })

    post_path.write_text(filled, encoding="utf-8")

    print(f"Created: {post_path}")
    print(f"Asset dir: {asset_dir}")
    print("Next steps:")
    print(f"  1) Add figure(s) under {asset_dir}")
    print(f"  2) Edit {post_path}")
    print(f"  3) Run: python3 scripts/validate_blog_post.py {post_path}")


if __name__ == "__main__":
    main()
