"""
Synthesizer node — writes the answer, citing which graph facts and/or
filing chunks it drew on. Every claim should be traceable back to either
a Cypher result or a specific filing chunk.
"""
from langchain_ollama import ChatOllama

from graph_rag.config import settings
from graph_rag.state import AgentState

SYNTHESIS_SYSTEM_PROMPT = """Answer the user's question using ONLY the provided context below.

Rules:
- Cite sources using bracketed numbers like [1], [2] matching the numbered context blocks.
- If the context does not contain enough information to answer, say so explicitly —
  do not guess or use outside knowledge about these companies.
- Graph facts (relationship data) are numbered separately from filing text chunks —
  cite whichever ones you actually used.
- Keep the answer focused and avoid restating the entire context.
"""


def _build_context_block(state: AgentState) -> tuple[str, list[str]]:
    blocks = []
    citations = []
    idx = 1

    for fact in state.get("graph_facts", []):
        blocks.append(f"[{idx}] (graph fact) {fact['text']}")
        citations.append(f"graph:{fact['text'][:60]}")
        idx += 1

    for chunk in state.get("reranked_chunks", []):
        blocks.append(
            f"[{idx}] (filing {chunk['ticker']}, {chunk.get('filing_id', '')}) {chunk['text'][:1000]}"
        )
        citations.append(f"filing:{chunk.get('chunk_id', '')}")
        idx += 1

    return "\n\n".join(blocks), citations


# Cached once at module load — avoids re-initializing the Ollama client on every call.
_synthesis_llm: ChatOllama | None = None


def get_synthesis_llm() -> ChatOllama:
    global _synthesis_llm
    if _synthesis_llm is None:
        _synthesis_llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
    return _synthesis_llm


def synthesize(state: AgentState) -> AgentState:
    context_text, citations = _build_context_block(state)

    if not context_text:
        return {
            **state,
            "draft_answer": (
                "I don't have enough retrieved context to answer this question. "
                "The corpus may not contain relevant information."
            ),
            "citations": [],
        }

    llm = get_synthesis_llm()
    prompt = (
        f"{SYNTHESIS_SYSTEM_PROMPT}\n\n"
        f"--- Context ---\n{context_text}\n--- End Context ---\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt)

    return {**state, "draft_answer": response.content, "citations": citations}