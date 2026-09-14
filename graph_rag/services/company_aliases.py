"""Maps known name variants of the 6 tracked companies to their canonical ticker,
so competitor/supplier mentions extracted from filing text (which name companies
inconsistently — "NVIDIA", "Nvidia Corp.", "NVIDIA Corporation") resolve back to
the same canonical Company node instead of creating disconnected duplicates."""

TICKER_ALIASES: dict[str, set[str]] = {
    "AMD":  {"amd", "advanced micro devices", "advanced micro devices, inc."},
    "AVGO": {"avgo", "broadcom", "broadcom inc.", "broadcom limited", "broadcom inc"},
    "NVDA": {"nvda", "nvidia", "nvidia corporation", "nvidia corp"},
    "QCOM": {"qcom", "qualcomm", "qualcomm incorporated"},
    "MU":   {"mu", "micron", "micron technology", "micron technology, inc."},
    "TXN":  {"txn", "texas instruments", "texas instruments incorporated"},
}

_NAME_SUFFIXES = (
    " Corporation", " Corp.", " Corp", " Inc.", " Inc",
    " Co.", " Company", " Ltd.", " Ltd", " LLC",
)


def normalize_company_name(name: str) -> str:
    """Strips common corporate suffixes so untracked companies (e.g. TSMC) don't
    fragment into multiple nodes across filings due to inconsistent suffixing."""
    n = name.strip()
    for suffix in _NAME_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def resolve_company_identity(name: str) -> dict:
    """Returns {"ticker": "NVDA"} if the name matches a tracked company,
    otherwise {"name": normalized_name} for an untracked company."""
    normalized = name.strip().lower()
    for ticker, aliases in TICKER_ALIASES.items():
        if normalized in aliases:
            return {"ticker": ticker}
    return {"name": normalize_company_name(name)}
