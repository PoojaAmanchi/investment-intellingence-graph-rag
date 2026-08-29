"""
Extracts structured entities (competitors, suppliers, executives, topics)
from 10-K filing text using an LLM with enforced structured output.

This is the highest-risk step in the pipeline: unlike FOMC voting records
(which are explicit numbers), competitor/supplier relationships are
described in prose and easy to over- or under-extract. The prompt below
explicitly forbids inference — only extract relationships stated in the text.

Verify this on 2-3 filings by hand before running it across the whole
corpus. Expect to iterate on the prompt.
"""
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from graph_rag.config import settings


class CompetitorMention(BaseModel):
    competitor_name: str = Field(description="Name of the competitor company as written in the text")
    evidence_quote: str = Field(description="The exact sentence or phrase stating this, under 30 words")


class SupplierMention(BaseModel):
    supplier_name: str = Field(description="Name of the supplier or supply-chain partner")
    relationship_type: Literal["supplier", "customer", "partner"] = "supplier"
    evidence_quote: str = Field(description="The exact sentence or phrase stating this, under 30 words")


class ExecutiveMention(BaseModel):
    name: str
    role: str = Field(description="Title as stated, e.g. 'Chief Executive Officer'")


class TopicMention(BaseModel):
    topic: Literal[
        "supply_chain", "competition", "regulatory", "geopolitical",
        "litigation", "cybersecurity", "talent", "other"
    ]
    evidence_quote: str = Field(description="Short supporting phrase, under 30 words")


class FilingExtraction(BaseModel):
    competitors: list[CompetitorMention] = Field(default_factory=list)
    suppliers: list[SupplierMention] = Field(default_factory=list)
    executives: list[ExecutiveMention] = Field(default_factory=list)
    topics: list[TopicMention] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You are extracting structured facts from a company's SEC 10-K filing.

Rules:
- Only extract relationships and entities EXPLICITLY STATED in the text. Do not infer
  or guess based on general knowledge of the industry.
- If the text does not name a specific competitor, supplier, or executive, do not
  invent one.
- For every competitor or supplier mention, include the exact quote (under 30 words)
  that supports it, so the extraction can be verified against the source.
- Company names should be extracted as written in the text (don't normalize or guess
  ticker symbols).
- If a section contains no relevant mentions for a category, return an empty list for it.
- You are being shown one PART of a larger document section. Only extract what is
  present in the text below — do not assume anything about parts you cannot see.

Return ONLY structured data matching the requested schema. No commentary.
"""


def get_extractor_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


class ExtractionTimeoutError(Exception):
    """Raised when a single filing section takes too long to extract —
    surfaces silent hangs (e.g. network interception, oversized input) as a
    visible, catchable error instead of freezing the whole ingestion run."""


def _invoke_with_timeout(structured_llm, prompt: str, timeout_seconds: int = 120):
    """
    Run structured_llm.invoke(prompt) on a background thread and enforce a
    hard timeout. Does NOT use ThreadPoolExecutor as a context manager —
    `with` blocks on shutdown() waiting for the worker thread to finish,
    which defeats the timeout if that thread is genuinely stuck forever.
    """
    import concurrent.futures

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(structured_llm.invoke, prompt)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        raise ExtractionTimeoutError(
            f"LLM call did not respond within {timeout_seconds}s "
            f"(input length: {len(prompt)} chars)"
        )
    finally:
        pool.shutdown(wait=False)


# --- Chunking -----------------------------------------------------------
#
# Long filing sections used to be silently truncated at MAX_CHARS before
# extraction ran, which meant anything past that point was never seen by
# the model (this is what caused NVDA's 2024/2025 filings to under-extract
# — not a timeout, silent data loss). We now split into overlapping chunks
# and run extraction on each, merging results.

MAX_CHARS = 8000  # conservative window for a 7B local model
CHUNK_OVERLAP = 400  # small overlap so entities split across a chunk
                     # boundary aren't missed entirely


def _split_into_chunks(text: str, max_chars: int = MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of at most max_chars. Overlap is
    small and only exists to reduce (not eliminate) boundary misses — this
    is a plain character-count splitter, not sentence-aware, since filing
    text is dense and irregular enough that a simple approach is more
    predictable to reason about than sentence segmentation.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    step = max_chars - overlap
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _merge_extractions(chunk_results: list[FilingExtraction]) -> FilingExtraction:
    """
    Merge per-chunk extractions into one, deduping by the fields that
    identify a unique mention. Overlap between chunks means the same
    entity can legitimately appear in two adjacent chunk results — this
    is expected, not a bug, and is deduped here rather than avoided
    upstream (avoiding it upstream would require sentence-aware
    splitting, which we deliberately skipped for simplicity).
    """
    merged = FilingExtraction()

    seen_competitors = set()
    for chunk in chunk_results:
        for c in chunk.competitors:
            key = c.competitor_name.strip().lower()
            if key not in seen_competitors:
                seen_competitors.add(key)
                merged.competitors.append(c)

    seen_suppliers = set()
    for chunk in chunk_results:
        for s in chunk.suppliers:
            key = (s.supplier_name.strip().lower(), s.relationship_type)
            if key not in seen_suppliers:
                seen_suppliers.add(key)
                merged.suppliers.append(s)

    seen_executives = set()
    for chunk in chunk_results:
        for e in chunk.executives:
            key = e.name.strip().lower()
            if key not in seen_executives:
                seen_executives.add(key)
                merged.executives.append(e)

    seen_topics = set()
    for chunk in chunk_results:
        for t in chunk.topics:
            key = (t.topic, t.evidence_quote.strip().lower())
            if key not in seen_topics:
                seen_topics.add(key)
                merged.topics.append(t)

    return merged


def extract_from_section(text: str, section_label: str) -> FilingExtraction:
    """
    Run structured extraction over a single filing section (business or
    risk factors). Long sections are split into overlapping chunks and
    extracted separately, then merged — this replaces the previous
    behavior of hard-truncating at MAX_CHARS, which silently dropped
    content past the cutoff (the root cause of NVDA's under-extraction
    on its longer 2024/2025 filings).
    """
    llm = get_extractor_llm()
    structured_llm = llm.with_structured_output(FilingExtraction)

    chunks = _split_into_chunks(text)
    chunk_results: list[FilingExtraction] = []

    if len(chunks) > 1:
        print(f"    {section_label}: split into {len(chunks)} chunks (section is long)")

    for i, chunk_text in enumerate(chunks):
        label = section_label if len(chunks) == 1 else f"{section_label} (part {i + 1}/{len(chunks)})"
        if len(chunks) > 1:
            print(f"    extracting chunk {i + 1}/{len(chunks)}...")
        prompt = (
            f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"--- {label} section text ---\n{chunk_text}\n--- end section ---"
        )
        try:
            result = _invoke_with_timeout(structured_llm, prompt)
        except ExtractionTimeoutError:
            # A single chunk timing out shouldn't sink the whole filing —
            # skip that chunk's contribution and keep going. This is a
            # deliberate change from before: previously one timeout on a
            # (now-truncated) section meant the filing had already lost
            # data anyway. Now, worst case, we lose one chunk out of N
            # instead of everything past MAX_CHARS.
            print(f"    chunk {i + 1}/{len(chunks)} timed out after 120s — skipping this chunk, continuing")
            continue
        chunk_results.append(result)

    return _merge_extractions(chunk_results)


def extract_from_filing(parsed_filing: dict) -> FilingExtraction:
    """
    Combine extraction across a filing's business and risk factors sections.
    Merges results rather than running one giant prompt over both, since
    each section separately fits comfortably in context and is easier to
    debug when extraction quality is checked by hand.
    """
    combined = FilingExtraction()

    if parsed_filing.get("business_section"):
        biz = extract_from_section(parsed_filing["business_section"], "Business (Item 1)")
        combined.competitors += biz.competitors
        combined.suppliers += biz.suppliers
        combined.executives += biz.executives
        combined.topics += biz.topics

    if parsed_filing.get("risk_factors_section"):
        risk = extract_from_section(parsed_filing["risk_factors_section"], "Risk Factors (Item 1A)")
        combined.competitors += risk.competitors
        combined.suppliers += risk.suppliers
        combined.executives += risk.executives
        combined.topics += risk.topics

    return combined