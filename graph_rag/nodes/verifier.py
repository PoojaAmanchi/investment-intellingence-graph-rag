"""
Verifier node — checks whether the draft answer is actually grounded in
the retrieved context. If not, and retries remain, signals the graph to
loop back to the retriever. This is the same CRAG pattern from the FOMC
project: generate, grade, retry on failure, give up honestly after
max retries rather than pretending certainty.
"""
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from graph_rag.config import settings
from graph_rag.state import AgentState

VERIFY_SYSTEM_PROMPT = """You are checking whether an answer is genuinely grounded in
the provided context, or whether it contains claims not supported by that context.

SPECIAL CASE: If the answer explicitly states it cannot find enough information in the
context (e.g. "I don't have enough retrieved context", "the filings do not mention this"),
treat this as fully grounded and score confidence = 0.95 — declining to answer when the
context doesn't support one is correct behavior, not a grounding failure.

Otherwise, score confidence from 0.0 (not grounded / fabricated) to 1.0 (every claim is
directly supported by the context).
"""


class VerificationResult(BaseModel):
    confidence: float = Field(description="0.0 to 1.0 groundedness score")
    reasoning: str = Field(description="brief explanation of the score")


# Cached once at module load — avoids re-initializing the Ollama client on every call.
_verifier_llm: ChatOllama | None = None


def get_verifier_llm() -> ChatOllama:
    global _verifier_llm
    if _verifier_llm is None:
        _verifier_llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
    return _verifier_llm


def verify(state: AgentState) -> AgentState:
    llm = get_verifier_llm().with_structured_output(VerificationResult)

    context_summary = "\n".join(
        [f"- {f['text']}" for f in state.get("graph_facts", [])]
        + [f"- {c['text'][:300]}" for c in state.get("reranked_chunks", [])]
    ) or "(no context was retrieved)"

    result: VerificationResult = llm.invoke(
        f"{VERIFY_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_summary}\n\n"
        f"Answer to verify:\n{state.get('draft_answer', '')}"
    )

    is_grounded = result.confidence >= settings.confidence_threshold
    retry_count = state.get("retry_count", 0)

    final_answer = state.get("draft_answer", "")
    if not is_grounded and retry_count >= settings.max_verify_retries:
        # Give up honestly rather than looping forever or pretending certainty
        final_answer += (
            "\n\n[Note: this answer could not be fully verified against the "
            "retrieved context after multiple retrieval attempts — treat with "
            "appropriate caution.]"
        )

    return {
        **state,
        "is_grounded": is_grounded,
        "confidence": result.confidence,
        "retry_count": retry_count + (0 if is_grounded else 1),
        "final_answer": final_answer if (is_grounded or retry_count >= settings.max_verify_retries) else "",
    }


def should_retry(state: AgentState) -> str:
    """Conditional edge function used by the graph to decide the next step."""
    if state.get("is_grounded"):
        return "done"
    if state.get("retry_count", 0) >= settings.max_verify_retries:
        return "done"  # give up, final_answer already has the caveat appended
    return "retry"
