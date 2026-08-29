"""
Pulls recent 10-K filings for a list of companies from SEC EDGAR.

SEC EDGAR is free and requires no API key — it only requires a compliant
User-Agent header identifying who is making the request (SEC policy, not
optional). Set SEC_USER_AGENT in your .env to "YourName your@email.com".

Rate limit: SEC asks for no more than ~10 requests/second. This script
sleeps briefly between requests to stay well under that.

Usage:
    python -m ingestion.fetch_sec_filings
"""
import json
import time
from pathlib import Path

import requests

from graph_rag.config import settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sec_filings"

# Starting universe: semiconductor sector, dense competitor/supplier relationships.
# CIK numbers are stable SEC identifiers — look up more at
# https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
COMPANIES = {
    "NVDA": "0001045810",
    "AMD": "0000002488",
    "INTC": "0000050863",
    "QCOM": "0000804328",
    "AVGO": "0001730168",
    "MU": "0000723125",
    "TXN": "0000097476",
    "ASML": "0000937966",
}

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def _headers() -> dict:
    return {"User-Agent": settings.sec_user_agent}


def get_recent_10k_filings(ticker: str, cik: str, limit: int = 3) -> list[dict]:
    """Return metadata for the most recent `limit` 10-K filings for a company."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    results = []
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            results.append({
                "ticker": ticker,
                "cik": cik,
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "primary_document": recent["primaryDocument"][i],
            })
        if len(results) >= limit:
            break
    return results


def download_filing_document(filing: dict) -> str:
    """Download the raw HTML of a filing's primary document."""
    accession_no_dashes = filing["accession_number"].replace("-", "")
    cik_int = str(int(filing["cik"]))  # SEC archive paths drop leading zeros
    url = (
        f"{SEC_ARCHIVES_BASE}/{cik_int}/{accession_no_dashes}/"
        f"{filing['primary_document']}"
    )
    resp = requests.get(url, headers=_headers(), timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_all(limit_per_company: int = 3) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for ticker, cik in COMPANIES.items():
        print(f"Fetching filing list for {ticker}...")
        filings = get_recent_10k_filings(ticker, cik, limit=limit_per_company)
        time.sleep(0.2)  # stay well under SEC's rate limit

        for filing in filings:
            out_name = f"{ticker}_{filing['filing_date']}.html"
            out_path = DATA_DIR / out_name
            if out_path.exists():
                print(f"  already have {out_name}, skipping")
                manifest.append({**filing, "local_path": str(out_path)})
                continue

            print(f"  downloading {out_name}...")
            html = download_filing_document(filing)
            out_path.write_text(html, encoding="utf-8")
            manifest.append({**filing, "local_path": str(out_path)})
            time.sleep(0.2)

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} filings recorded in {manifest_path}")


if __name__ == "__main__":
    fetch_all()
