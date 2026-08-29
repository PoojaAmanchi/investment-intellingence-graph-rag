"""
Router node — classifies the incoming question and extracts any company
names/tickers mentioned, so the hybrid retriever knows whether to run a
Cypher traversal, a vector search, or both.

Question types:
- factual: single-filing lookup ("what does NVDA's 10-K say about X")
- relational: multi-hop company relationship ("who supplies X and competes with Y")
- temporal: how something changed across filings over time
- chitchat: not a retrieval question at all
"""
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from graph_rag.config import settings
from graph_rag.state import AgentState

ROUTER_SYSTEM_PROMPT = """Classify the user's question into exactly one category:

- "factual": a lookup about one company from one filing (e.g. "what risks does NVDA mention about talent")
- "relational": requires reasoning about relationships BETWEEN companies
  (competitors, suppliers, shared relationships) — e.g. "which of X's suppliers compete with Y"
- "temporal": asks how something changed across filings/time for one or more companies
- "chitchat": greetings, small talk, or anything not about company filings

Also extract any company names or stock tickers explicitly mentioned in the question.
If none are mentioned, return an empty list — do not guess.
"""


class RouterOutput(BaseModel):
    question_type: str = Field(description="one of: factual, relational, temporal, chitchat")
    entities: list[str] = Field(default_factory=list, description="company names/tickers mentioned")


# Cached once at module load — avoids re-initializing the Ollama client on every call.
_router_llm: ChatOllama | None = None


def get_router_llm() -> ChatOllama:
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
    return _router_llm


def route(state: AgentState) -> AgentState:
    llm = get_router_llm().with_structured_output(RouterOutput)
    result: RouterOutput = llm.invoke(
        f"{ROUTER_SYSTEM_PROMPT}\n\nQuestion: {state['question']}"
    )

    question_type = result.question_type if result.question_type in (
        "factual", "relational", "temporal", "chitchat"
    ) else "factual"  # safe default rather than crashing on an unexpected label

    return {
        **state,
        "question_type": question_type,
        "entities": result.entities,
        "retry_count": state.get("retry_count", 0),
    }