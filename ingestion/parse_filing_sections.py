"""
Extracts the "Item 1. Business" and "Item 1A. Risk Factors" sections from a
raw 10-K HTML file. These two sections are where competitor names, supplier
relationships, and topic-level risks (supply chain, regulatory, geopolitical)
are described in plain language — they're the richest source for entity
extraction in this project.

10-K formatting varies a lot between filers, so this uses a heuristic
regex-based split on section headers rather than assuming a fixed structure.
Expect to spot-check and adjust the patterns for filers that don't match.

Usage:
    python -m ingestion.parse_filing_sections
"""
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sec_filings"
PARSED_DIR = DATA_DIR / "parsed"

# Matches variants like "Item 1.", "Item 1 .", "ITEM 1:", etc.
SECTION_PATTERNS = {
    "business": re.compile(r"item\s*1\.?\s+business", re.IGNORECASE),
    "risk_factors": re.compile(r"item\s*1a\.?\s+risk\s*factors", re.IGNORECASE),
    "next_section": re.compile(r"item\s*1b\.?\s+|item\s*2\.?\s+properties", re.IGNORECASE),
}


def html_to_text(html: str) -> str:
    """Strip HTML down to clean, whitespace-normalized plain text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse repeated blank lines from table/formatting artifacts
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_section(text: str, start_pattern: re.Pattern, end_pattern: re.Pattern,
                     min_length: int = 500) -> str | None:
    """
    10-K filings almost always mention "Item 1. Business" twice: once in the
    Table of Contents (short line), and once as the real section header.
    Checking only the first match often grabs the TOC line by mistake.
    Fix: try every match until one has enough real content after it.
    """
    for start_match in start_pattern.finditer(text):
        end_match = end_pattern.search(text, pos=start_match.end())
        end_pos = end_match.start() if end_match else start_match.end() + 20000
        section = text[start_match.end():end_pos].strip()
        if len(section) > min_length:
            return section
    return None


def parse_filing(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    text = html_to_text(html)

    business = extract_section(
        text, SECTION_PATTERNS["business"], SECTION_PATTERNS["risk_factors"]
    )
    risk_factors = extract_section(
        text, SECTION_PATTERNS["risk_factors"], SECTION_PATTERNS["next_section"]
    )

    return {
        "source_file": html_path.name,
        "business_section": business,
        "risk_factors_section": risk_factors,
        "business_found": business is not None,
        "risk_factors_found": risk_factors is not None,
    }


def parse_all() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "No manifest.json found. Run fetch_sec_filings.py first."
        )

    manifest = json.loads(manifest_path.read_text())
    results = []

    for entry in manifest:
        html_path = Path(entry["local_path"])
        if not html_path.exists():
            print(f"Missing file, skipping: {html_path}")
            continue

        print(f"Parsing {html_path.name}...")
        parsed = parse_filing(html_path)
        parsed["ticker"] = entry["ticker"]
        parsed["filing_date"] = entry["filing_date"]

        if not parsed["business_found"]:
            print(f"  WARNING: business section not found in {html_path.name} — "
                  f"check SECTION_PATTERNS, this filer's format may differ")
        if not parsed["risk_factors_found"]:
            print(f"  WARNING: risk factors section not found in {html_path.name}")

        out_name = html_path.stem + "_parsed.json"
        out_path = PARSED_DIR / out_name
        out_path.write_text(json.dumps(parsed, indent=2))
        results.append(parsed)

    print(f"\nParsed {len(results)} filings. Output in {PARSED_DIR}")
    found_business = sum(r["business_found"] for r in results)
    found_risk = sum(r["risk_factors_found"] for r in results)
    print(f"Business section found: {found_business}/{len(results)}")
    print(f"Risk factors found: {found_risk}/{len(results)}")


if __name__ == "__main__":
    parse_all()
