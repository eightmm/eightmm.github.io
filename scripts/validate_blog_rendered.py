#!/usr/bin/env python3
"""Blog post rendered validator (Level 2: Web-level).

Builds the Jekyll site via Docker (or connects to live GitHub Pages URL),
then uses Playwright to check each post for:
- Code blocks rendered properly (not raw markdown)
- Mermaid diagrams rendered as SVG (not raw text)
- Math equations rendered (MathJax/KaTeX)
- Images loaded (no broken images)
- No layout/CSS breakage
- Mobile responsiveness (optional)

Usage:
  # Against live site:
  python3 validate_blog_rendered.py --url https://eightmm.github.io

  # Against local Jekyll (Docker):
  python3 validate_blog_rendered.py --docker

  # Specific posts only:
  python3 validate_blog_rendered.py --url https://eightmm.github.io --posts alphafold1 speclig
"""

import argparse
import json
import subprocess
import sys
import time
import re
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def get_post_urls(base_url, post_dir=None, filter_posts=None):
    """Get post URLs from _posts directory filenames."""
    if post_dir is None:
        post_dir = Path(__file__).parent.parent / "_posts"

    urls = []
    for f in sorted(post_dir.glob("*.md")):
        if f.name == ".placeholder":
            continue
        # Parse filename: YYYY-MM-DD-slug.md → /posts/slug/
        match = re.match(r'\d{4}-\d{2}-\d{2}-(.+)\.md', f.name)
        if match:
            slug = match.group(1)
            if filter_posts and not any(fp in slug for fp in filter_posts):
                continue
            urls.append((f.name, f"{base_url.rstrip('/')}/posts/{slug}/"))
    return urls


def validate_page(page, name, url, results):
    """Validate a single rendered blog post page."""
    errors = []
    warnings = []

    try:
        resp = page.goto(url, wait_until="networkidle", timeout=30000)
        if resp and resp.status >= 400:
            errors.append(f"HTTP {resp.status}")
            results.append({"name": name, "url": url, "errors": errors, "warnings": warnings})
            return
    except Exception as e:
        errors.append(f"Failed to load: {e}")
        results.append({"name": name, "url": url, "errors": errors, "warnings": warnings})
        return

    # Wait a bit for JS rendering (mermaid, mathjax)
    page.wait_for_timeout(3000)

    # 1. Code blocks — should be rendered as <pre><code> not raw ```
    raw_fences = page.evaluate("""
        () => {
            const body = document.querySelector('.post-content') || document.body;
            const text = body.innerText;
            const matches = text.match(/^```\\w*$/gm);
            return matches ? matches.length : 0;
        }
    """)
    if raw_fences > 0:
        errors.append(f"Found {raw_fences} raw ``` fences — code blocks not rendering")

    code_blocks = page.query_selector_all("pre code")
    if len(code_blocks) == 0:
        warnings.append("No rendered code blocks found")
    else:
        for i, block in enumerate(code_blocks):
            text = block.inner_text().strip()
            if len(text) < 10:
                warnings.append(f"Code block {i+1} seems empty or too short")

    # 2. Mermaid diagrams — should render as SVG
    mermaid_svgs = page.query_selector_all(".mermaid svg, .mermaid[data-processed] svg")
    mermaid_raw = page.evaluate("""
        () => {
            const elems = document.querySelectorAll('.mermaid, pre code.language-mermaid');
            let raw = 0;
            elems.forEach(el => {
                if (!el.querySelector('svg') && el.textContent.includes('graph') || el.textContent.includes('sequenceDiagram')) {
                    raw++;
                }
            });
            return raw;
        }
    """)
    if mermaid_raw > 0:
        errors.append(f"{mermaid_raw} mermaid diagram(s) not rendered (showing raw text)")

    # Also check for mermaid error messages
    mermaid_errors = page.query_selector_all(".mermaid .error-text, .mermaid .error-icon")
    if mermaid_errors:
        errors.append(f"{len(mermaid_errors)} mermaid diagram(s) have rendering errors")

    # 3. Math equations — MathJax/KaTeX should process them
    raw_latex = page.evaluate("""
        () => {
            const body = document.querySelector('.post-content') || document.body;
            const text = body.innerText;
            // Look for unrendered LaTeX ($$...$$ or \\(...\\) still visible as text)
            const matches = text.match(/\\$\\$[^$]+\\$\\$/g);
            return matches ? matches.length : 0;
        }
    """)
    if raw_latex > 0:
        warnings.append(f"{raw_latex} unrendered LaTeX equation(s) found")

    # 4. Broken images
    broken_images = page.evaluate("""
        () => {
            const imgs = document.querySelectorAll('.post-content img');
            let broken = 0;
            imgs.forEach(img => {
                if (!img.complete || img.naturalWidth === 0) broken++;
            });
            return broken;
        }
    """)
    if broken_images > 0:
        warnings.append(f"{broken_images} broken image(s)")

    # 5. General layout check — post content should exist
    post_content = page.query_selector(".post-content, article")
    if not post_content:
        errors.append("No post content element found — layout may be broken")

    results.append({
        "name": name,
        "url": url,
        "errors": errors,
        "warnings": warnings,
        "code_blocks": len(code_blocks),
        "mermaid_svgs": len(mermaid_svgs),
    })


def start_docker_jekyll(repo_dir):
    """Start Jekyll in Docker and return the process + URL."""
    print("🐳 Starting Jekyll via Docker...")
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "-v", f"{repo_dir}:/srv/jekyll",
         "-p", "4000:4000", "jekyll/jekyll:4", "jekyll", "serve", "--host", "0.0.0.0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Wait for server to start
    for _ in range(30):
        time.sleep(2)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:4000")
            print("✅ Jekyll server ready")
            return proc, "http://localhost:4000"
        except Exception:
            continue
    print("❌ Jekyll server failed to start")
    proc.kill()
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Validate rendered blog posts")
    parser.add_argument("--url", default="https://eightmm.github.io",
                        help="Base URL of the blog")
    parser.add_argument("--docker", action="store_true",
                        help="Start local Jekyll via Docker")
    parser.add_argument("--posts", nargs="*",
                        help="Filter specific posts by slug substring")
    parser.add_argument("--install", action="store_true",
                        help="Install playwright browsers")
    args = parser.parse_args()

    if args.install:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
        return

    if not HAS_PLAYWRIGHT:
        print("❌ playwright not installed. Run:")
        print("  pip install playwright && python3 -m playwright install chromium")
        sys.exit(1)

    docker_proc = None
    base_url = args.url

    if args.docker:
        repo_dir = str(Path(__file__).parent.parent.resolve())
        docker_proc, base_url = start_docker_jekyll(repo_dir)
        if not base_url:
            sys.exit(1)

    try:
        post_urls = get_post_urls(base_url, filter_posts=args.posts)
        if not post_urls:
            print("No posts found to validate")
            sys.exit(1)

        print(f"\n🔍 Validating {len(post_urls)} posts at {base_url}\n")

        results = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for name, url in post_urls:
                print(f"  Checking: {name}...")
                validate_page(page, name, url, results)

            browser.close()

        # Report
        total_pass = 0
        total_fail = 0
        print(f"\n{'='*70}")
        print("📊 RENDERED VALIDATION RESULTS")
        print(f"{'='*70}")

        for r in results:
            status = "✅" if not r["errors"] else "❌"
            if status == "✅":
                total_pass += 1
            else:
                total_fail += 1

            print(f"\n{status} {r['name']}")
            print(f"   URL: {r['url']}")
            if r.get("code_blocks"):
                print(f"   Code blocks: {r['code_blocks']}, Mermaid SVGs: {r.get('mermaid_svgs', 0)}")
            for e in r["errors"]:
                print(f"   ❌ {e}")
            for w in r["warnings"]:
                print(f"   ⚠️  {w}")

        print(f"\n{'='*70}")
        print(f"📊 TOTAL: {total_pass} passed, {total_fail} failed out of {len(results)}")
        print(f"{'='*70}")

        sys.exit(0 if total_fail == 0 else 1)

    finally:
        if docker_proc:
            docker_proc.kill()


if __name__ == "__main__":
    main()
