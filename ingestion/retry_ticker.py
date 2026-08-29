"""
Retries entity extraction + graph ingestion for specific tickers only,
instead of redoing the entire 21-filing run. Useful when a handful of
filings timed out on a slow/loaded machine while the rest succeeded.

Usage:
    python -m ingestion.retry_ticker NVDA
    python -m ingestion.retry_ticker NVDA QCOM
"""
import json
import sys
from pathlib import Path

from graph_rag.services.entity_extractor import extract_from_filing, ExtractionTimeoutError
from graph_rag.services.graph_service import GraphService
from ingestion.ingest_graph import chunk_text

PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "sec_filings" / "parsed"


def retry_tickers(tickers: list[str]) -> None:
    tickers = {t.upper() for t in tickers}
    gs = GraphService()

    parsed_files = sorted(PARSED_DIR.glob("*_parsed.json"))
    matching = [
        p for p in parsed_files
        if json.loads(p.read_text())["ticker"].upper() in tickers
    ]

    if not matching:
        print(f"No parsed filings found for tickers: {tickers}")
        return

    print(f"Retrying {len(matching)} filing(s) for: {', '.join(sorted(tickers))}\n")

    for path in matching:
        parsed = json.loads(path.read_text())
        ticker = parsed["ticker"]
        filing_date = parsed["filing_date"]
        filing_id = f"{ticker}_{filing_date}"

        print(f"Ingesting {filing_id}...")
        gs.upsert_company(ticker)
        gs.upsert_filing(filing_id, ticker, filing_date)

        for section_name, section_text in [
            ("business", parsed.get("business_section")),
            ("risk_factors", parsed.get("risk_factors_section")),
        ]:
            if not section_text:
                continue
            for i, chunk in enumerate(chunk_text(section_text)):
                chunk_id = f"{filing_id}_{section_name}_{i}"
                gs.upsert_filing_chunk(filing_id, chunk_id, chunk, section_name)

        print("  running entity extraction (this calls the local LLM, may take a moment)...")
        try:
            extraction = extract_from_filing(parsed)
        except ExtractionTimeoutError as e:
            print(f"  SKIPPED (timed out again): {e}")
            continue

        for comp in extraction.competitors:
            gs.upsert_competitor_relationship(ticker, comp.competitor_name, comp.evidence_quote, filing_id)
        for sup in extraction.suppliers:
            gs.upsert_supplier_relationship(ticker, sup.supplier_name, sup.evidence_quote, filing_id)
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

    gs.close()
    print("\nRetry complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.retry_ticker TICKER [TICKER ...]")
        sys.exit(1)
    retry_tickers(sys.argv[1:])