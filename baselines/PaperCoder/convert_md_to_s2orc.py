"""
Convert paper.md (MinerU output) to S2ORC-compatible JSON for PaperCoder input.

Uses s2orc-doc2json's data model (Paper, Paragraph, Metadata) to produce
valid S2ORC release JSON without needing Grobid/Java.
"""

import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "s2orc-doc2json"))

from doc2json.s2orc import Paper, Paragraph, Metadata, Author


def parse_markdown(md_path: str, paper_id: str) -> Dict:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    # Extract title: first # heading
    title = ""
    title_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            title_end = i + 1
            break

    # Extract abstract: text between title/author block and first numbered section
    abstract_text = ""
    body_start = 0
    author_lines = []
    in_author_block = True

    for i in range(title_end, len(lines)):
        line = lines[i].strip()

        # Detect author block end: empty line or section start
        if in_author_block and i < title_end + 15:
            if re.match(r"^#\s+\d+", line) or re.match(r"^#\s+[A-Z]", line):
                in_author_block = False
                body_start = i
                break
            if line and not line.startswith("#") and not line.startswith("!"):
                author_lines.append(line)
            elif not line and author_lines:
                in_author_block = False
            continue

        if re.match(r"^#\s+\d+", line) or re.match(r"^#\s+[A-Z]", line):
            body_start = i
            break

    if body_start == 0:
        body_start = title_end

    # Collect abstract text: everything between author block and first section
    abstract_parts = []
    for i in range(title_end + len(author_lines) + 1, body_start):
        line = lines[i].strip()
        if line and not line.startswith("!["):
            abstract_parts.append(line)
    abstract_text = " ".join(abstract_parts)
    # Clean up markdown formatting
    abstract_text = re.sub(r"\*\*(.*?)\*\*", r"\1", abstract_text)
    abstract_text = re.sub(r"\*(.*?)\*", r"\1", abstract_text)

    # Parse authors: first line after title that looks like names/affiliations
    authors = []
    for aline in author_lines:
        # Skip affiliation markers like "1FAIR, Meta"
        if re.match(r"^\d+[A-Za-z\s,]+$", aline) and len(aline) < 60:
            continue
        # Skip email/Correspondence lines
        if "Correspondence:" in aline or "@" in aline:
            continue
        # Simple author parsing
        if "," in aline or ";" in aline:
            for name in re.split(r"[,;]\s*", aline):
                name = name.strip()
                if name and len(name) > 2 and not name.startswith("http"):
                    parts = name.split()
                    if len(parts) >= 2:
                        authors.append(
                            {"first": parts[0], "middle": parts[1:-1] if len(parts) > 2 else [],
                             "last": parts[-1], "suffix": "", "affiliation": {}, "email": ""}
                        )
        elif aline and len(aline) > 2 and not aline.startswith("http"):
            parts = aline.split()
            if len(parts) >= 2:
                authors.append(
                    {"first": parts[0], "middle": parts[1:-1] if len(parts) > 2 else [],
                     "last": parts[-1], "suffix": "", "affiliation": {}, "email": ""}
                )

    # Extract body text by sections
    body_text = []
    current_section = ""
    current_sec_num = ""
    current_paragraphs = []

    for i in range(body_start, len(lines)):
        line = lines[i].strip()

        # Section header
        section_match = re.match(r"^#\s+(\d+\.?\d*)\s+(.+)", line)
        if section_match:
            if current_paragraphs and current_section:
                body_text.append({
                    "text": " ".join(current_paragraphs),
                    "cite_spans": [],
                    "ref_spans": [],
                    "eq_spans": [],
                    "section": current_section,
                    "sec_num": current_sec_num
                })
            current_sec_num = section_match.group(1).rstrip(".")
            current_section = section_match.group(2).strip()
            current_paragraphs = []
            continue

        # Subsection header
        sub_match = re.match(r"^##\s+(.+)", line)
        if sub_match:
            if current_paragraphs and current_section:
                body_text.append({
                    "text": " ".join(current_paragraphs),
                    "cite_spans": [],
                    "ref_spans": [],
                    "eq_spans": [],
                    "section": f"{current_section}::{sub_match.group(1).strip()}",
                    "sec_num": current_sec_num
                })
            current_paragraphs = []
            continue

        # Skip images, empty lines, math blocks (they get inline in text)
        if line.startswith("![") or line.startswith(">"):
            continue
        if line in ("$$", "```"):
            continue

        # Clean inline formatting
        cleaned = re.sub(r"\$\$(.+?)\$\$", r"[\1]", line)
        cleaned = re.sub(r"\$(.+?)\$", r"\1", cleaned)
        cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\!\[.*?\]\(.*?\)", "", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)

        if cleaned:
            current_paragraphs.append(cleaned)

    # Don't forget the last section
    if current_paragraphs and current_section:
        body_text.append({
            "text": " ".join(current_paragraphs),
            "cite_spans": [],
            "ref_spans": [],
            "eq_spans": [],
            "section": current_section,
            "sec_num": current_sec_num
        })

    # Build metadata
    metadata = {
        "title": title,
        "authors": authors if authors else [{"first": "Unknown", "middle": [], "last": "", "suffix": "", "affiliation": {}, "email": ""}],
        "year": None,
        "venue": None,
        "identifiers": {}
    }

    # Build paragraphs for abstract
    abstract_paragraphs = []
    if abstract_text:
        # Split abstract into sentences chunks
        abstract_paragraphs = [{
            "text": abstract_text,
            "cite_spans": [],
            "ref_spans": [],
            "eq_spans": [],
            "section": "Abstract",
            "sec_num": None
        }]

    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "abstract": abstract_paragraphs,
        "body_text": body_text,
        "back_matter": [],
        "bib_entries": {},
        "ref_entries": {}
    }


def build_s2orc_json(parsed: Dict, paper_id: str) -> Dict:
    """Build S2ORC release JSON using s2orc-doc2json data classes."""
    paper = Paper(
        paper_id=paper_id,
        pdf_hash="",
        metadata=parsed["metadata"],
        abstract=parsed["abstract"],
        body_text=parsed["body_text"],
        back_matter=parsed["back_matter"],
        bib_entries=parsed["bib_entries"],
        ref_entries=parsed["ref_entries"]
    )
    return paper.release_json("pdf")


def convert_paper(md_path: str, output_path: str, paper_id: str = None):
    """Convert a single paper.md to S2ORC JSON."""
    if paper_id is None:
        paper_id = Path(md_path).parent.name

    print(f"Converting: {paper_id}")
    parsed = parse_markdown(md_path, paper_id)
    s2orc_json = build_s2orc_json(parsed, paper_id)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(s2orc_json, f, indent=4, sort_keys=False, ensure_ascii=False)

    print(f"  → saved: {output_path}")
    print(f"    body_text paragraphs: {len(parsed['body_text'])}")
    return output_path


def convert_all_papers(papers_dir: str, output_dir: str):
    """Convert all paper.md files under papers_dir to S2ORC JSON."""
    papers_dir = Path(papers_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for paper_dir in sorted(papers_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        md_path = paper_dir / "paper.md"
        if not md_path.exists():
            print(f"SKIP {paper_dir.name}: no paper.md")
            continue

        paper_id = paper_dir.name
        output_path = output_dir / f"{paper_id}.json"
        convert_paper(str(md_path), str(output_path), paper_id)
        converted.append(paper_id)

    print(f"\nDone. {len(converted)} papers converted.")
    return converted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers_dir", type=str,
                        default="/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/data/papers")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/data/s2orc_jsons")
    parser.add_argument("--paper", type=str, help="Single paper name to convert")
    args = parser.parse_args()

    if args.paper:
        md_path = Path(args.papers_dir) / args.paper / "paper.md"
        output_path = Path(args.output_dir) / f"{args.paper}.json"
        convert_paper(str(md_path), str(output_path), args.paper)
    else:
        convert_all_papers(args.papers_dir, args.output_dir)
