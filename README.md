# Investment Intelligence Graph RAG

> A hybrid Graph + Vector RAG system that answers relational questions about semiconductor companies — competitors, suppliers, executives, and risk factors — extracted from real SEC 10-K filings.

**Stack:** LangGraph · Neo4j · FAISS · Ollama · FastAPI · RAGAS 
**Cost:** $0 — fully local (Ollama + Neo4j Aura free tier) 
**Data:** 18 SEC 10-K filings across 6 companies (AMD, AVGO, NVDA, QCOM, MU, TXN), all parsed successfully

---

## Why Graph RAG?

Plain vector RAG retrieves text that *looks* similar to the query. It can't answer questions that require traversing relationships between entities:

| Question | Why vector RAG alone fails |
|----------|---------------------------|
| "Which of AMD's suppliers also compete with NVDA?" | Requires two-hop graph traversal across `SUPPLIES` and `COMPETES_WITH` — no single chunk contains this |
| "How did QCOM's risk language around supply chain change across filings?" | Requires chronologically chained filings, not similarity search |
| "Which executives lead companies that compete with TXN?" | Requires joining `LEADS` and `COMPETES_WITH` relationships |

This system answers all three by combining Neo4j graph traversal with vector search in one pipeline.

---

## Architecture

```
Router ──── classifies question type + extracts company entities
 │
 ▼
Hybrid Retriever
 ├── Graph branch ──► Neo4j Cypher traversal (competitors, suppliers, execs)
 └── Vector branch ──► Neo4j vector search ──► FAISS fallback
 │
 ▼
CrossEncoder Reranker ──── merges + reranks graph facts and vector chunks
 │
 ▼
Synthesizer ──── generates cited answer from reranked context
 │
 ▼
Verifier ──── groundedness check (0.0–1.0 confidence score)
 ├── grounded ──────────────────────────────► final answer
 ├── not grounded + retries left ──────────► loop back to Retriever
 └── retries exhausted ────────────────────► honest low-confidence answer
```

**Key design decision:** Neo4j is the single source of truth for both graph traversal *and* vector search. Graph relationships and vector embeddings live in one connected database — not two separate systems. FAISS activates only as a local fallback.

**Question routing:** The router classifies each query into one of four types before retrieval fires:
- `relational` — multi-hop company relationships → runs Cypher traversal
- `temporal` — change across filings → chains `FOLLOWED_BY` edges
- `factual` — single-company lookup → vector search only
- `chitchat` — no retrieval needed

---

## Data Model

**Nodes:** `Company` · `Filing` · `Executive` · `Topic` · `FilingChunk`

**Relationships:**

```
(Company)-[:COMPETES_WITH]-(Company)
(Company)<-[:SUPPLIES]-(Company)
(Executive)-[:LEADS]->(Company)
(Filing)-[:MENTIONS]->(Topic)
(Filing)-[:CONTAINS]->(FilingChunk)
(Filing)-[:FOLLOWED_BY]->(Filing) ← enables temporal queries
(Filing)-[:FILED_BY]->(Company)
```

---

## Results

### Ingestion

**Fetching:** 18 filings fetched from SEC EDGAR across 6 companies (AMD, AVGO, NVDA, QCOM, MU, TXN) — all 18 downloaded successfully, zero download failures.

**Parsing:** 18 filings parsed successfully across all 6 companies.

**Parsing bug found and fixed during development:** the first parsing run only produced 3/18 successfully parsed filings, all TXN. Root cause: the regex in `parse_filing_sections.py` was matching Table-of-Contents entries instead of the actual section headers, so it only worked by coincidence on filings where TOC and header text happened to align. Fixed before the full ingestion run reflected in the numbers above.

**Neo4j graph state after ingestion** (verified in Aura Query browser):

| Entity | Count |
|--------|-------|
| Company nodes | 29 |
| COMPETES_WITH relationships | 24 |
| SUPPLIES relationships | 7 |


### Pipeline — observed query behavior

The verifier's retry loop fired in real runs across multiple questions:

| Question | Confidence | Retries |
|----------|-----------|---------|
| "Which companies compete with NVDA?" | 0.8 | 1 |
| "What risks does NVDA mention?" | 0.7 | 2 |
| "What risks does QCOM mention about supply chain?" | 0.6 | 2 |
| "What products does TXN compete on?" | 0.5 | 2 |
| "What risks does AMD mention about competition?" | 0.4 | 2 |

The max-retry path returns an explicitly low-confidence answer with a caveat rather than a confident but unsupported response — confirmed working. 

### RAGAS Evaluation

Eval set built by running 6 real questions through the live pipeline (`eval/build_eval_set.py`) — no hand-typed answers or placeholder data. Scored with a local Ollama judge (qwen2.5:7b):

| Metric | Score |
|--------|-------|
| Faithfulness | **0.83** |
| Answer Relevancy | **0.65** |
| Context Recall | **0.7** |
| Context Precision | **0.75** |

---

## Project Structure

```
ingestion/
 fetch_sec_filings.py # pulls 10-Ks from SEC EDGAR (no API key needed)
 parse_filing_sections.py # extracts Item 1 / Item 1A sections
 ingest_graph.py # LLM entity extraction → Neo4j nodes + edges
 ingest_vector.py # Neo4j vector index + FAISS fallback

graph_rag/
 graph.py # LangGraph state machine, compiled once at startup
 state.py # shared AgentState TypedDict
 nodes/
 router.py # question classification + entity extraction
 hybrid_retriever.py # graph branch + vector branch
 reranker.py # CrossEncoder merge + rerank
 synthesizer.py # cited answer generation
 verifier.py # groundedness check + retry logic
 services/
 graph_service.py # Neo4j driver + Cypher reads/writes
 entity_extractor.py # structured LLM extraction with Pydantic

api/
 main.py # FastAPI app
 routes.py # /query, /stream/query (SSE), /eval/run, /graph/schema

frontend/
 index.html # plain HTML/CSS/JS with SSE progress streaming

eval/
 build_eval_set.py # generates real eval_set.json from live pipeline output
 ragas_runner.py # RAGAS metrics with local Ollama judge
 data/eval_set.json # 6 real questions with captured contexts + answers
```

---

## Setup & Run

### Prerequisites
- Python 3.11–3.13
- Docker (for Neo4j), or a free [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) instance
- [Ollama](https://ollama.com) installed and running locally

```bash
ollama pull qwen2.5:7b
ollama pull mxbai-embed-large
ollama serve
```

### Install & configure

```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Set SEC_USER_AGENT to "YourName your@email.com" (required by SEC EDGAR)
# Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD to match your Neo4j instance
```

### Run the data pipeline

```bash
docker-compose up -d # skip if using Aura

python -m ingestion.fetch_sec_filings # download 10-Ks from EDGAR
python -m ingestion.parse_filing_sections # extract Business + Risk Factors text
python -m ingestion.ingest_graph # entity extraction → Neo4j (slow — Ollama)
python -m ingestion.ingest_vector # build vector indexes
```

> Entity extraction is the slow step — a local 7B model extracting across 18 filings takes time. Run `ingest_graph.py` on one company first to sanity-check before scaling to the full corpus.

### Query + serve

```bash
# Terminal test
python -m graph_rag.graph "Which companies compete with AMD?"

# API
uvicorn api.main:app --reload

# Frontend (separate terminal)
cd frontend && npx serve . -l 3000
# open http://localhost:3000

# Build eval set + run RAGAS
python -m eval.build_eval_set
python -m eval.ragas_runner
```


---

*Built with LangGraph · Neo4j · Ollama · FastAPI · RAGAS*