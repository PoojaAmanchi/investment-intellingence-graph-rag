"""FastAPI app entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Compile the LangGraph state machine once at startup, not per-request
    from graph_rag.graph import get_graph
    get_graph()
    print("Graph compiled and ready.")
    yield
    print("Shutting down.")


app = FastAPI(title="Investment Intelligence Graph RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
