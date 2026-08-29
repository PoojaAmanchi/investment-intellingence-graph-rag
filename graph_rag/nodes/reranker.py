"""
Reranks merged vector chunks against the question using a local CrossEncoder.
Graph facts bypass reranking — they're already precise Cypher results, not
similarity-scored candidates, so there's nothing to rerank them against.
"""
from sentence_transformers import CrossEncoder

from graph_rag.config import settings
from graph_rag.state import AgentState, RetrievedChunk

_model_cache: dict[str, CrossEncoder] = {}


def get_reranker() -> CrossEncoder:
    if "model" not in _model_cache:
        _model_cache["model"] = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _model_cache["model"]


def rerank(state: AgentState) -> AgentState:
    chunks = state.get("vector_chunks", [])
    if not chunks:
        return {**state, "reranked_chunks": []}

    model = get_reranker()
    pairs = [(state["question"], c["text"]) for c in chunks]
    scores = model.predict(pairs)

    scored_chunks: list[RetrievedChunk] = [
        {**chunk, "score": float(score)} for chunk, score in zip(chunks, scores)
    ]
    scored_chunks.sort(key=lambda c: c["score"], reverse=True)

    top_k = settings.top_k_retrieval
    return {**state, "reranked_chunks": scored_chunks[:top_k]}
