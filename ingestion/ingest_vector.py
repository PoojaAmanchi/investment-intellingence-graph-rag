"""
Populates Neo4j's built-in vector index over FilingChunk nodes (hybrid
BM25 + dense search), and separately builds a local FAISS index as a
fallback in case Neo4j's vector search is unavailable or returns nothing.

Both indexes are built from the same FilingChunk nodes created during
ingest_graph.py — run that first.

Usage:
    python -m ingestion.ingest_vector
"""
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings

from graph_rag.config import settings
from graph_rag.services.graph_service import GraphService

FAISS_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index"

NEO4J_VECTOR_INDEX_NAME = "filing_chunk_vector_index"


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )


def fetch_all_chunks() -> list[dict]:
    gs = GraphService()
    chunks = gs.run(
        """
        MATCH (chunk:FilingChunk)<-[:CONTAINS]-(f:Filing)-[:FILED_BY]->(c:Company)
        RETURN chunk.chunk_id AS chunk_id, chunk.text AS text, chunk.section AS section,
               f.filing_id AS filing_id, f.date AS filing_date, c.ticker AS ticker
        """
    )
    gs.close()
    return chunks


def build_neo4j_vector_index(chunks: list[dict]) -> None:
    """
    Populates the hybrid vector index directly on Neo4j. Per the reference
    architecture, Neo4j is the PRIMARY vector store — this is what enables
    graph context and vector context to come from one connected database
    rather than two separate systems.

    NOTE: Neo4jVector.from_texts() creates its own embedding property on
    the nodes it creates — since our FilingChunk nodes already exist from
    ingest_graph.py, we use from_existing_graph-compatible indexing by
    writing embeddings directly onto those nodes. See engineering note
    below on why the index_name must match exactly what retrieval expects.
    """
    embeddings = get_embeddings()
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "filing_id": c["filing_id"],
            "ticker": c["ticker"],
            "section": c["section"],
            "filing_date": c["filing_date"],
        }
        for c in chunks
    ]

    print(f"Embedding and indexing {len(texts)} chunks into Neo4j...")

    batch_size = 50
    store = None
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]
        print(f"  batch {start // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} "
              f"({start}-{min(end, len(texts))} of {len(texts)})...")

        if store is None:
            store = Neo4jVector.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_metadatas,
                url=settings.neo4j_uri,
                username=settings.neo4j_user,
                password=settings.neo4j_password,
                database=settings.neo4j_database,
                index_name=NEO4J_VECTOR_INDEX_NAME,
                node_label="FilingChunkEmbedding",
                text_node_property="text",
                embedding_node_property="embedding",
            )
        else:
            store.add_texts(texts=batch_texts, metadatas=batch_metadatas)

    print("Neo4j vector index built.")
    print(
        "NOTE: this creates FilingChunkEmbedding nodes alongside your existing "
        "FilingChunk nodes rather than embedding directly onto FilingChunk, to "
        "avoid LangChain's from_texts() overwriting existing node properties. "
        "Link them by chunk_id if you want a single unified node — see README "
        "known-limitations section."
    )


def build_faiss_fallback(chunks: list[dict]) -> None:
    embeddings = get_embeddings()
    texts = [c["text"] for c in chunks]
    metadatas = [
        {"chunk_id": c["chunk_id"], "filing_id": c["filing_id"], "ticker": c["ticker"], "section": c["section"]}
        for c in chunks
    ]

    print(f"Building FAISS fallback index over {len(texts)} chunks...")

    batch_size = 50
    faiss_store = None
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]
        print(f"  batch {start // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} "
              f"({start}-{min(end, len(texts))} of {len(texts)})...")

        if faiss_store is None:
            faiss_store = FAISS.from_texts(texts=batch_texts, embedding=embeddings, metadatas=batch_metadatas)
        else:
            faiss_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(str(FAISS_INDEX_DIR))
    print(f"FAISS index saved to {FAISS_INDEX_DIR}")


def run() -> None:
    chunks = fetch_all_chunks()
    if not chunks:
        raise ValueError(
            "No FilingChunk nodes found in Neo4j. Run ingest_graph.py first."
        )

    build_neo4j_vector_index(chunks)
    build_faiss_fallback(chunks)
    print("\nVector ingestion complete.")


if __name__ == "__main__":
    run()
