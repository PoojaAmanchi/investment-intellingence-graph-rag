"""
Builds the Neo4j knowledge graph from parsed filing sections.

Pipeline: parsed filing JSON -> LLM entity extraction -> Neo4j nodes/edges.

Requires:
- Neo4j running (docker-compose up -d)
- Ollama running locally with OLLAMA_MODEL pulled
- parse_filing_sections.py already run (produces data/sec_filings/parsed/*.json)

Usage:
    python -m ingestion.ingest_graph                 # ingest all parsed filings
    python -m ingestion.ingest_graph --ticker NVDA    # re-run just one company

--ticker is useful after fixing something that only affected certain
filings (e.g. the entity_extractor truncation fix, which specifically
under-processed NVDA's longer 2024/2025 filings) — it lets you re-ingest
just that company without re-running (and re-billing time against) the
other five.

Note: re-running a ticker assumes GraphService's upsert_* methods use
MERGE (idempotent) rather than CREATE in Neo4j — re-running should update
existing nodes/edges, not duplicate them. If you're not sure, check
graph_service.py before re-running, or wipe just that company's nodes
first.
"""
import argparse
import json
from pathlib import Path

from graph_rag.services.company_aliases import resolve_company_identity
from graph_rag.services.entity_extractor import extract_from_filing, ExtractionTimeoutError
from graph_rag.services.graph_service import GraphService

PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "sec_filings" / "parsed"


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Simple fixed-size chunking with overlap for vector indexing later."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def ingest_all(ticker_filter: str | None = None) -> None:
    gs = GraphService()
    gs.ensure_constraints()

    parsed_files = sorted(PARSED_DIR.glob("*_parsed.json"))
    if not parsed_files:
        raise FileNotFoundError(
            f"No parsed filings found in {PARSED_DIR}. Run parse_filing_sections.py first."
        )

    # Track the most recent filing_id per ticker to chain FOLLOWED_BY edges.
    # NOTE: when ticker_filter is set, this dict starts empty each run, so
    # FOLLOWED_BY chaining is only built among the filings processed in
    # *this* run. That's correct as long as all of that ticker's filings
    # are being re-ingested together (the normal case for a targeted
    # re-run) — but if you ever ingest a ticker's filings in more than one
    # partial batch, the chain across batches won't be linked automatically.
    last_filing_by_ticker: dict[str, str] = {}

    skipped_by_filter = 0
    processed_count = 0

    for path in parsed_files:
        parsed = json.loads(path.read_text())
        ticker = parsed["ticker"]

        if ticker_filter and ticker.upper() != ticker_filter.upper():
            skipped_by_filter += 1
            continue

        filing_date = parsed["filing_date"]
        filing_id = f"{ticker}_{filing_date}"

        print(f"\nIngesting {filing_id}...")
        gs.upsert_company(ticker)
        gs.upsert_filing(filing_id, ticker, filing_date)

        if ticker in last_filing_by_ticker:
            gs.link_prior_filing(ticker, filing_id, last_filing_by_ticker[ticker])
        last_filing_by_ticker[ticker] = filing_id

        # Chunk both sections for later vector indexing
        for section_name, section_text in [
            ("business", parsed.get("business_section")),
            ("risk_factors", parsed.get("risk_factors_section")),
        ]:
            if not section_text:
                continue
            for i, chunk in enumerate(chunk_text(section_text)):
                chunk_id = f"{filing_id}_{section_name}_{i}"
                gs.upsert_filing_chunk(filing_id, chunk_id, chunk, section_name)

        # Entity extraction — this is the step to spot-check by hand
        print("  running entity extraction (this calls the local LLM, may take a moment)...")
        try:
            extraction = extract_from_filing(parsed)
        except ExtractionTimeoutError as e:
            print(f"  SKIPPED (timed out): {e}")
            continue

        for comp in extraction.competitors:
            # Resolve to a canonical ticker if this competitor is one of the 6
            # tracked companies (e.g. "NVIDIA Corporation" -> {"ticker": "NVDA"}),
            # so the relationship lands on the same node as the company's own
            # canonical Company{ticker: ...} node instead of a disconnected
            # name-keyed duplicate.
            identity = resolve_company_identity(comp.competitor_name)
            gs.upsert_competitor_relationship(ticker, identity, comp.evidence_quote, filing_id)
        for sup in extraction.suppliers:
            identity = resolve_company_identity(sup.supplier_name)
            gs.upsert_supplier_relationship(ticker, identity, sup.evidence_quote, filing_id)
        for exe in extraction.executives:
            gs.upsert_executive(ticker, exe.name, exe.role, filing_id)
        for topic in extraction.topics:
            gs.upsert_topic_mention(filing_id, topic.topic, topic.evidence_quote)

        print(
            f"  extracted: {len(extraction.competitors)} competitors, "
            f"{len(extraction.suppliers)} suppliers, "
            f"{len(extraction.executives)} executives, "
            f"{len(extraction.topics)} topic mentions"
        )
        processed_count += 1

    gs.close()

    if ticker_filter:
        print(
            f"\nGraph ingestion complete for ticker={ticker_filter}: "
            f"{processed_count} filing(s) processed, {skipped_by_filter} skipped (other tickers)."
        )
    else:
        print(f"\nGraph ingestion complete: {processed_count} filing(s) processed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest parsed SEC filings into Neo4j.")
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Only ingest filings for this ticker (e.g. NVDA). Omit to ingest all parsed filings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_all(ticker_filter=args.ticker)
