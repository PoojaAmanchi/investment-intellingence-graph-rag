"""
Hybrid retriever — runs the graph branch and vector branch, merging results
before reranking. This is the node that makes the system "GraphRAG" rather
than plain RAG: relational questions get real Cypher traversal instead of
just similar-looking text chunks.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings

from graph_rag.config import settings
from graph_rag.services.graph_service import GraphService
from graph_rag.state import AgentState, GraphFact, RetrievedChunk

FAISS_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index"
NEO4J_VECTOR_INDEX_NAME = "filing_chunk_vector_index"


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)


def run_graph_branch(entities: list[str], question_type: str) -> list[GraphFact]:
    """
    Runs relationship-oriented Cypher queries for any companies mentioned.
    Only fires for relational/temporal question types — factual/chitchat
    questions don't need graph traversal.
    """
    if question_type not in ("relational", "temporal") or not entities:
        return []

    # No-Docker / no-Neo4j setups: skip graph reasoning entirely rather than
    # crashing. The vector branch (FAISS) still answers the question below,
    # just without relationship-traversal facts.
    try:
        gs = GraphService()
        facts: list[GraphFact] = []

        for entity in entities:
            # Try as ticker first (uppercase convention), fall back to name match
            ticker = entity.upper()

            competitors = gs.find_competitors_of(ticker)
            for row in competitors:
                facts.append({
                    "text": f"{ticker} competes with {row['competitor_name']} "
                            f"(evidence: {row.get('evidence', 'n/a')})",
                    "filing_id": None,
                })

            suppliers = gs.find_suppliers_of(ticker)
            for row in suppliers:
                facts.append({
                    "text": f"{row['supplier_name']} is a supplier to {ticker}",
                    "filing_id": None,
                })

        # If exactly two entities given, also check for shared cross-hop relationships
        if len(entities) == 2:
            a, b = entities[0].upper(), entities[1].upper()
            shared = gs.find_shared_relationships(a, b)
            for row in shared:
                facts.append({
                    "text": f"{row['company_name']} supplies {a} and competes with {b}",
                    "filing_id": None,
                })

        gs.close()
        return facts
    except Exception as e:
        print(f"Graph branch skipped (Neo4j unavailable: {e}). Falling back to vector-only retrieval.")
        return []


def run_vector_branch(question: str, top_k: int) -> list[RetrievedChunk]:
    """
    Neo4j hybrid vector search first (primary store, per architecture design).
    Falls back to local FAISS if Neo4j vector search returns nothing —
    e.g. if the index hasn't been populated, or Neo4j is unreachable.
    """
    embeddings = _get_embeddings()

    try:
        neo4j_store = Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            index_name=NEO4J_VECTOR_INDEX_NAME,
        )
        results = neo4j_store.similarity_search_with_score(question, k=top_k)
        if results:
            return [
                {
                    "text": doc.page_content,
                    "source": "vector",
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "filing_id": doc.metadata.get("filing_id", ""),
                    "ticker": doc.metadata.get("ticker", ""),
                    "score": float(score),
                }
                for doc, score in results
            ]
    except Exception as e:
        print(f"Neo4j vector search failed or returned nothing ({e}), falling back to FAISS")

    # Fallback path
    if not FAISS_INDEX_DIR.exists():
        return []
    faiss_store = FAISS.load_local(
        str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    results = faiss_store.similarity_search_with_score(question, k=top_k)
    return [
        {
            "text": doc.page_content,
            "source": "faiss_fallback",
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "filing_id": doc.metadata.get("filing_id", ""),
            "ticker": doc.metadata.get("ticker", ""),
            "score": float(score),
        }
        for doc, score in results
    ]


def retrieve(state: AgentState) -> AgentState:
    graph_facts = run_graph_branch(state.get("entities", []), state.get("question_type", "factual"))
    vector_chunks = run_vector_branch(state["question"], top_k=settings.reranker_initial_k)

    return {
        **state,
        "graph_facts": graph_facts,
        "vector_chunks": vector_chunks,
    }
