"""
Neo4j connection and Cypher operations. Handles both graph writes (during
ingestion) and graph reads (during retrieval).
"""
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable

from graph_rag.config import settings


class GraphService:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self):
        self.driver.close()

    def run(self, query: str, params: dict | None = None, _retries: int = 2) -> list[dict]:
        """
        Long-running ingestion can leave this connection idle for minutes
        (while the LLM extraction step runs), long enough for AuraDB to
        close it as stale. Retry on connection errors — each retry opens a
        fresh session, forcing the driver to re-establish a live connection.
        """
        for attempt in range(_retries + 1):
            try:
                with self.driver.session() as session:
                    result = session.run(query, params or {})
                    return [record.data() for record in result]
            except (SessionExpired, ServiceUnavailable):
                if attempt == _retries:
                    raise
                print(f"  (Neo4j connection went stale, reconnecting... attempt {attempt + 2}/{_retries + 1})")

    # ---------- Schema / constraints ----------

    def ensure_constraints(self):
        constraints = [
            "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
            "CREATE CONSTRAINT filing_id IF NOT EXISTS FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
            "CREATE CONSTRAINT executive_name IF NOT EXISTS FOR (e:Executive) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        ]
        for c in constraints:
            self.run(c)

    # ---------- Writes ----------

    def upsert_company(self, ticker: str, name: str = "", sector: str = "semiconductors"):
        self.run(
            """
            MERGE (c:Company {ticker: $ticker})
            SET c.name = coalesce($name, c.name), c.sector = $sector
            """,
            {"ticker": ticker, "name": name or ticker, "sector": sector},
        )

    def upsert_filing(self, filing_id: str, ticker: str, filing_date: str, filing_type: str = "10-K"):
        self.run(
            """
            MATCH (c:Company {ticker: $ticker})
            MERGE (f:Filing {filing_id: $filing_id})
            SET f.date = $filing_date, f.type = $filing_type
            MERGE (c)<-[:FILED_BY]-(f)
            """,
            {
                "filing_id": filing_id,
                "ticker": ticker,
                "filing_date": filing_date,
                "filing_type": filing_type,
            },
        )

    def link_prior_filing(self, ticker: str, current_filing_id: str, prior_filing_id: str):
        """Chains filings chronologically per company: enables drift queries."""
        self.run(
            """
            MATCH (prior:Filing {filing_id: $prior_id})
            MATCH (current:Filing {filing_id: $current_id})
            MERGE (prior)-[:FOLLOWED_BY]->(current)
            """,
            {"prior_id": prior_filing_id, "current_id": current_filing_id},
        )

def upsert_competitor_relationship(self, ticker: str, identity: dict, evidence: str, filing_id: str):
    if "ticker" in identity:
        match_clause = "MERGE (comp:Company {ticker: $key})"
    else:
        match_clause = "MERGE (comp:Company {name: $key})"
    key = identity.get("ticker") or identity["name"]
    self.run(
        f"""
        MATCH (c:Company {{ticker: $ticker}})
        {match_clause}
        MERGE (c)-[r:COMPETES_WITH]-(comp)
        SET r.evidence = $evidence, r.filing_id = $filing_id
        """,
        {"ticker": ticker, "key": key, "evidence": evidence, "filing_id": filing_id},
    )

def upsert_supplier_relationship(self, ticker: str, identity: dict, evidence: str, filing_id: str):
    if "ticker" in identity:
        match_clause = "MERGE (s:Company {ticker: $key})"
    else:
        match_clause = "MERGE (s:Company {name: $key})"
    key = identity.get("ticker") or identity["name"]
    self.run(
        f"""
        MATCH (c:Company {{ticker: $ticker}})
        {match_clause}
        MERGE (s)-[r:SUPPLIES]->(c)
        SET r.evidence = $evidence, r.filing_id = $filing_id
        """,
        {"ticker": ticker, "key": key, "evidence": evidence, "filing_id": filing_id},
    )
    def upsert_executive(self, ticker: str, exec_name: str, role: str, filing_id: str):
        self.run(
            """
            MATCH (c:Company {ticker: $ticker})
            MERGE (e:Executive {name: $exec_name})
            SET e.role = $role
            MERGE (e)-[r:LEADS]->(c)
            SET r.as_of_filing = $filing_id
            """,
            {"ticker": ticker, "exec_name": exec_name, "role": role, "filing_id": filing_id},
        )

    def upsert_topic_mention(self, filing_id: str, topic: str, evidence: str):
        self.run(
            """
            MATCH (f:Filing {filing_id: $filing_id})
            MERGE (t:Topic {name: $topic})
            MERGE (f)-[r:MENTIONS]->(t)
            SET r.evidence = $evidence
            """,
            {"filing_id": filing_id, "topic": topic, "evidence": evidence},
        )

    def upsert_filing_chunk(self, filing_id: str, chunk_id: str, text: str, section: str):
        self.run(
            """
            MATCH (f:Filing {filing_id: $filing_id})
            MERGE (chunk:FilingChunk {chunk_id: $chunk_id})
            SET chunk.text = $text, chunk.section = $section
            MERGE (f)-[:CONTAINS]->(chunk)
            """,
            {"filing_id": filing_id, "chunk_id": chunk_id, "text": text, "section": section},
        )

    # ---------- Reads ----------

    def find_suppliers_of(self, ticker: str) -> list[dict]:
        return self.run(
            """
            MATCH (s:Company)-[:SUPPLIES]->(c:Company {ticker: $ticker})
            RETURN s.name AS supplier_name, s.ticker AS supplier_ticker
            """,
            {"ticker": ticker},
        )

    def find_competitors_of(self, ticker: str) -> list[dict]:
        return self.run(
            """
            MATCH (c:Company {ticker: $ticker})-[r:COMPETES_WITH]-(comp:Company)
            RETURN comp.name AS competitor_name, r.evidence AS evidence
            """,
            {"ticker": ticker},
        )

    def find_shared_relationships(self, ticker_a: str, ticker_b: str) -> list[dict]:
        """Companies that supply A and compete with B, or similar cross-hop patterns."""
        return self.run(
            """
            MATCH (a:Company {ticker: $ticker_a})<-[:SUPPLIES]-(shared:Company)
            MATCH (shared)-[:COMPETES_WITH]-(b:Company {ticker: $ticker_b})
            RETURN DISTINCT shared.name AS company_name
            """,
            {"ticker_a": ticker_a, "ticker_b": ticker_b},
        )

    def get_schema_summary(self) -> dict:
        node_counts = self.run(
            """
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS count', {})
            YIELD value
            RETURN label, value.count AS count
            """
        )
        return {"node_counts": node_counts}
