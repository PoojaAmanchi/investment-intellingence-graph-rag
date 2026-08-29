"""API route definitions."""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    retries_used: int
    citations: list[str]
    question_type: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    from graph_rag.graph import ask

    result = ask(request.question)
    return QueryResponse(
        answer=result.get("final_answer") or result.get("draft_answer", ""),
        confidence=result.get("confidence", 0.0),
        retries_used=result.get("retry_count", 0),
        citations=result.get("citations", []),
        question_type=result.get("question_type", "unknown"),
    )


@router.get("/graph/schema")
def graph_schema():
    from graph_rag.services.graph_service import GraphService

    gs = GraphService()
    try:
        summary = gs.get_schema_summary()
    except Exception as e:
        summary = {"error": str(e), "note": "requires APOC plugin enabled in Neo4j"}
    gs.close()
    return summary


@router.post("/eval/run")
def run_eval():
    from eval.ragas_runner import run_ragas_eval_from_file

    scores = run_ragas_eval_from_file()
    return scores.to_dict(orient="records")


@router.get("/stream/query")
def stream_query(question: str):
    """
    Server-sent events endpoint streaming pipeline progress, matching the
    pattern from the FOMC project's frontend. Since LangGraph's own
    streaming API differs from a plain LLM token stream, this emits one
    event per node completion rather than per-token — coarser, but honest
    about what's actually happening in a multi-node retry pipeline.
    """
    import json

    from graph_rag.graph import get_graph

    def event_stream():
        graph = get_graph()
        initial_state = {"question": question, "retry_count": 0}
        for step_output in graph.stream(initial_state):
            for node_name in step_output.keys():
                yield f"data: {json.dumps({'node': node_name})}\n\n"
        yield f"event: done\ndata: {json.dumps({})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
