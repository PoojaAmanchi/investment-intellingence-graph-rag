"""Shared state passed between every node in the LangGraph pipeline."""
from typing import Literal, TypedDict


class RetrievedChunk(TypedDict):
    text: str
    source: str  # "graph" or "vector" or "faiss_fallback"
    chunk_id: str
    filing_id: str
    ticker: str
    score: float


class GraphFact(TypedDict):
    text: str  # human-readable rendering of the Cypher result row
    filing_id: str | None


class AgentState(TypedDict, total=False):
    question: str

    # Router output
    question_type: Literal["factual", "relational", "temporal", "chitchat"]
    entities: list[str]  # company names/tickers extracted from the question

    # Retrieval
    graph_facts: list[GraphFact]
    vector_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]

    # Generation
    draft_answer: str
    citations: list[str]

    # Verification / retry loop
    is_grounded: bool
    confidence: float
    retry_count: int
    final_answer: str
