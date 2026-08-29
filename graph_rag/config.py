"""
Central configuration, loaded once from .env.
Every module imports `settings` from here rather than reading os.environ directly.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_password_here"
    neo4j_database: str = "neo4j"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "mxbai-embed-large"

    sec_user_agent: str = "YourName YourEmail@example.com"

    top_k_retrieval: int = 5
    reranker_initial_k: int = 15
    max_verify_retries: int = 2
    confidence_threshold: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()
