"""
Wires all nodes into the LangGraph state machine:

    router -> hybrid_retriever -> reranker -> synthesizer -> verifier
                  ^                                              |
                  |______________ retry (if not grounded) _______|

The graph is compiled once at module import and reused across requests —
rebuilding it per-request is a silent performance bug (noted as such in
the reference architecture this project is based on).
"""
from langgraph.graph import END, StateGraph

from graph_rag.nodes.hybrid_retriever import retrieve
from graph_rag.nodes.reranker import rerank
from graph_rag.nodes.router import route
from graph_rag.nodes.synthesizer import synthesize
from graph_rag.nodes.verifier import should_retry, verify
from graph_rag.state import AgentState


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", route)
    workflow.add_node("retriever", retrieve)
    workflow.add_node("reranker", rerank)
    workflow.add_node("synthesizer", synthesize)
    workflow.add_node("verifier", verify)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "synthesizer")
    workflow.add_edge("synthesizer", "verifier")

    workflow.add_conditional_edges(
        "verifier",
        should_retry,
        {
            "retry": "retriever",  # loop back — router already extracted entities, no need to re-route
            "done": END,
        },
    )

    return workflow.compile()


# Compiled once, reused across requests
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(question: str) -> dict:
    graph = get_graph()
    initial_state = {"question": question, "retry_count": 0}
    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Quick terminal test, mirrors the FOMC project's `python -m rag_agent.graph` pattern
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Which companies compete with AMD?"
    print(f"Question: {q}\n")
    result = ask(q)
    print("Final answer:")
    print(result.get("final_answer") or result.get("draft_answer"))
    print(f"\nConfidence: {result.get('confidence')}")
    print(f"Retries used: {result.get('retry_count')}")
