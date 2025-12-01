#!/usr/bin/env python3
"""
Simple web chat server that talks to the existing RAG CLI logic.

This DOES NOT modify any existing files. It just imports and reuses
functions from `ingestion/scripts/rag_cli.py` to provide a web API.

Run:
    cd /Users/hamzaahmadnaeem/Downloads/company-docs-copilot
    python -m web_chat.server
"""

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


# Ensure project root is on sys.path so we can import ingestion.scripts.rag_cli
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing CLI logic
from ingestion.scripts import rag_cli as rag  # type: ignore  # noqa: E402


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


app = FastAPI(title="RAG Web Chat", version="1.0.0")

# Allow local frontends (React dev, this static page, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global, lazily-initialized components reused across requests
_index = None
_local_model = None
_openai_embedding_client = None
_llm_client = None
_llm_provider: Optional[str] = None


def ensure_initialized():
    """Initialize RAG components once, reused for all requests."""
    global _index, _local_model, _openai_embedding_client, _llm_client, _llm_provider

    if _index is not None and _llm_client is not None:
        return

    print("🔧 Initializing RAG components for web chat...")
    _index = rag.initialize_pinecone()
    _local_model, _openai_embedding_client = rag.initialize_embedding_model()
    _llm_client, _llm_provider = rag.initialize_llm_client()
    print("✅ RAG components initialized for web chat")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the simple chat page."""
    html_path = PROJECT_ROOT / "web_chat" / "index.html"
    return FileResponse(str(html_path))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Web API endpoint that mirrors the RAG CLI behaviour."""
    ensure_initialized()

    question = request.question.strip()
    if not question:
        return ChatResponse(answer="Please enter a non-empty question.", sources=[])

    # Normalize the query for better retrieval, but keep original for answering
    normalized = rag.normalize_query(
        question, _llm_client, _llm_provider or "openai"
    )
    query_for_embedding = normalized or question

    # Create embedding
    query_embedding = rag.create_embedding(
        query_for_embedding, _local_model, _openai_embedding_client
    )
    if not query_embedding:
        return ChatResponse(
            answer="Failed to create embedding for your question.", sources=[]
        )

    # Query vector DB
    retrieved_docs = rag.query_vector_db(_index, query_embedding, rag.TOP_K_RESULTS)
    if not retrieved_docs:
        # Still try to get a helpful answer but note the missing docs
        context = ""
        sources: List[str] = []
    else:
        context = rag.build_context(retrieved_docs)
        sources = [doc.get("source", "unknown") for doc in retrieved_docs]

    # Generate answer
    answer_text = rag.generate_response(
        question, context, _llm_client, _llm_provider or "openai"
    )
 
    return ChatResponse(answer=normalized+answer_text, sources=sources)

def main():
    """Entry point so this module can be run directly with -m."""
    import uvicorn

    uvicorn.run(
        "web_chat.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()


