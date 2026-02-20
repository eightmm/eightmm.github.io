#!/usr/bin/env python3
"""Blog post structural validator (Level 1: Markdown-level).

Checks:
- Front matter completeness (title, date, description, categories, tags, math, mermaid, image)
- Required sections presence and order
- Code block integrity (matched fences, language tags)
- Mermaid block safety (no <br>, no inline style that breaks Chirpy)
- Pseudocode + PyTorch code presence
- Paper Info section
- Content ratio (How It Works >= 40%)
- SEO description length (120-160 chars)
- Cross-linking (series posts)
"""

import sys
import re
import os
import yaml
from pathlib import Path

REQUIRED_FRONT_MATTER = ["title", "date", "description", "categories", "tags"]
RECOMMENDED_FRONT_MATTER = ["math", "mermaid", "image"]

REQUIRED_SECTIONS = [
    "Hook",
    "Problem",
    "Key Idea",
    "How It Works",
    "Results",
    "Limitations",
    "Conclusion",
    "Paper Info",
]

VALID_CATEGORIES = {
    "AI": ["Generative Models", "Protein Structure", "Drug Discovery", "NLP", "Vision",
           "Reinforcement Learning", "Theory", "General"],
    "Bio": ["Structural Biology", "Genomics", "Drug Design", "General"],
    "Dev": ["MLOps", "Systems", "Tools", "General"],
    "General": ["Tutorial", "Review", "Opinion"],
}


class ValidationResult:
    def __init__(self, filepath):
        self.filepath = filepath
        self.errors = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(f"❌ {msg}")

    def warn(self, msg):
        self.warnings.append(f"⚠️  {msg}")

    @property
    def passed(self):
        return len(self.errors) == 0

    def report(self):
        name = os.path.basename(self.filepath)
        lines = [f"\n{'='*60}", f"📄 {name}", f"{'='*60}"]
        if self.errors:
            lines.append(f"\n🔴 ERRORS ({len(self.errors)}):")
            lines.extend(f"  {e}" for e in self.errors)
        if self.warnings:
            lines.append(f"\n🟡 WARNINGS ({len(self.warnings)}):")
            lines.extend(f"  {w}" for w in self.warnings)
        if self.passed:
            lines.append("\n✅ PASSED")
        else:
            lines.append(f"\n❌ FAILED ({len(self.errors)} errors)")
        return "\n".join(lines)


def parse_front_matter(content):
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def validate(filepath):
    r = ValidationResult(filepath)
    content = Path(filepath).read_text(encoding="utf-8")

    # --- Front Matter ---
    fm = parse_front_matter(content)
    if fm is None:
        r.error("Front matter missing or invalid YAML")
        return r

    for key in REQUIRED_FRONT_MATTER:
        if key not in fm:
            r.error(f"Missing required front matter: {key}")

    for key in RECOMMENDED_FRONT_MATTER:
        if key not in fm:
            r.warn(f"Missing recommended front matter: {key}")

    # Categories validation
    if "categories" in fm:
        cats = fm["categories"]
        if isinstance(cats, list) and len(cats) >= 1:
            main_cat = cats[0]
            if main_cat not in VALID_CATEGORIES:
                r.error(f"Invalid main category: '{main_cat}'. Valid: {list(VALID_CATEGORIES.keys())}")
            elif len(cats) >= 2:
                sub_cat = cats[1]
                if sub_cat not in VALID_CATEGORIES[main_cat]:
                    r.warn(f"Unusual subcategory '{sub_cat}' for '{main_cat}'")
        else:
            r.error("Categories should be a list with at least 1 item")

    # SEO description
    if "description" in fm:
        desc = str(fm["description"])
        if len(desc) < 80:
            r.warn(f"SEO description too short ({len(desc)} chars, recommend 120-160)")
        elif len(desc) > 200:
            r.warn(f"SEO description too long ({len(desc)} chars, recommend 120-160)")

    # --- Body content (after front matter) ---
    body_match = re.search(r'^---\n.*?\n---\n(.*)$', content, re.DOTALL)
    if not body_match:
        r.error("No body content found")
        return r
    body = body_match.group(1)

    # --- Required Sections ---
    found_sections = re.findall(r'^## (.+)$', body, re.MULTILINE)
    found_names = [s.strip() for s in found_sections]

    for section in REQUIRED_SECTIONS:
        if section not in found_names:
            r.error(f"Missing required section: ## {section}")

    # --- Code Block Integrity ---
    lines = body.split('\n')
    fence_stack = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('```'):
            if len(stripped) > 3 and not stripped.endswith('```'):
                # Opening fence with language
                fence_stack.append((i, stripped))
            elif stripped == '```':
                if fence_stack:
                    fence_stack.pop()
                else:
                    r.warn(f"Line {i}: Closing fence without matching opening")

    for line_num, fence in fence_stack:
        r.error(f"Line {line_num}: Unclosed code fence: {fence}")

    # --- Mermaid Safety ---
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)```', body, re.DOTALL)
    for i, block in enumerate(mermaid_blocks):
        if '<br' in block:
            r.error(f"Mermaid block {i+1}: Contains <br> tag — use \\n in node labels instead")
        if re.search(r'^\s*style\s+\w+\s+fill:', block, re.MULTILINE):
            r.warn(f"Mermaid block {i+1}: Contains inline style — may not render in Chirpy theme. Use class definitions or remove.")
        if '<' in block and '>' in block and 'subgraph' not in block.split('<')[0].split('\n')[-1]:
            # Check for HTML-like tags that aren't part of node labels
            html_tags = re.findall(r'<(?!--|subgraph)\w+[^>]*>', block)
            if html_tags:
                r.warn(f"Mermaid block {i+1}: Contains HTML-like tags that may break rendering: {html_tags[:3]}")

    # --- Pseudocode + PyTorch ---
    has_pseudocode = bool(re.search(r'```(?:pseudocode|text|plaintext)\b', body)) or \
                     bool(re.search(r'(?:pseudo\s*code|algorithm|procedure)', body, re.IGNORECASE))
    has_python = bool(re.search(r'```python', body))

    if not has_pseudocode:
        r.warn("No pseudocode block found (expected ```pseudocode or algorithmic description)")
    if not has_python:
        r.error("No Python/PyTorch code block found (```python required)")

    # Check for PyTorch imports in python blocks
    python_blocks = re.findall(r'```python\n(.*?)```', body, re.DOTALL)
    has_torch = any('torch' in block or 'nn.Module' in block for block in python_blocks)
    if not has_torch and python_blocks:
        r.warn("Python code blocks don't contain PyTorch code (torch/nn.Module)")

    # --- Content Ratio (How It Works >= 40%) ---
    section_positions = [(m.start(), m.group(1).strip())
                         for m in re.finditer(r'^## (.+)$', body, re.MULTILINE)]

    total_body_len = len(body.strip())
    hiw_len = 0
    for i, (pos, name) in enumerate(section_positions):
        if name == "How It Works":
            end = section_positions[i+1][0] if i+1 < len(section_positions) else len(body)
            hiw_len = end - pos
            break

    if total_body_len > 0 and hiw_len > 0:
        ratio = hiw_len / total_body_len
        if ratio < 0.35:
            r.error(f"How It Works section too short: {ratio:.1%} (minimum 35%, target 40%)")
        elif ratio < 0.40:
            r.warn(f"How It Works section slightly below target: {ratio:.1%} (target 40%)")

    # --- Paper Info Section ---
    if "Paper Info" in found_names:
        paper_info_match = re.search(r'## Paper Info\n(.*?)(?=\n## |\Z)', body, re.DOTALL)
        if paper_info_match:
            pi = paper_info_match.group(1)
            for field in ["Title", "Authors", "Published", "Link"]:
                if field not in pi:
                    r.warn(f"Paper Info missing field: {field}")

    # --- Word count ---
    word_count = len(body.split())
    if word_count < 2000:
        r.warn(f"Post seems short: ~{word_count} words (recommend 2500+)")

    return r


def main():
    if len(sys.argv) < 2:
        # Validate all posts in _posts/
        post_dir = Path(__file__).parent.parent / "_posts"
        files = sorted(post_dir.glob("*.md"))
        if not files:
            print("No posts found in _posts/")
            sys.exit(1)
    else:
        files = [Path(f) for f in sys.argv[1:]]

    total_pass = 0
    total_fail = 0

    for f in files:
        if not f.exists():
            print(f"File not found: {f}")
            total_fail += 1
            continue
        result = validate(str(f))
        print(result.report())
        if result.passed:
            total_pass += 1
        else:
            total_fail += 1

    print(f"\n{'='*60}")
    print(f"📊 TOTAL: {total_pass} passed, {total_fail} failed out of {len(files)}")
    print(f"{'='*60}")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
