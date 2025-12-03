#!/usr/bin/env python3
"""
Web chat server using LangChain-based RAG with conversation memory.

This server uses the LangChain RAG implementation which includes:
- Conversation memory/context
- Better handling of follow-up questions
- LangChain's ConversationalRetrievalChain

Run:
    cd /Users/hamzaahmadnaeem/Downloads/company-docs-copilot
    python -m web_chat.server_langchain
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel


# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the LangChain RAG implementation
from ingestion.scripts import rag_langchain  # type: ignore  # noqa: E402


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None  # Optional: for multi-user support


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: str
    chat_history_length: int


app = FastAPI(title="LangChain RAG Web Chat", version="1.0.0")

# Allow local frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global RAG instances per conversation (for multi-user support)
_rag_instances: Dict[str, rag_langchain.LangChainRAG] = {}
_default_rag: Optional[rag_langchain.LangChainRAG] = None


def get_or_create_rag(conversation_id: Optional[str] = None) -> rag_langchain.LangChainRAG:
    """Get or create a RAG instance for a conversation."""
    global _default_rag
    
    # If no conversation_id, use default instance
    if not conversation_id:
        if _default_rag is None:
            print("🔧 Initializing default LangChain RAG for web chat...")
            _default_rag = rag_langchain.LangChainRAG()
            _default_rag.initialize()
        return _default_rag
    
    # Use conversation-specific instance
    if conversation_id not in _rag_instances:
        print(f"🔧 Initializing LangChain RAG for conversation: {conversation_id}")
        rag = rag_langchain.LangChainRAG()
        rag.conversation_id = conversation_id
        rag.initialize()
        _rag_instances[conversation_id] = rag
    
    return _rag_instances[conversation_id]


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the simple chat page."""
    html_path = PROJECT_ROOT / "web_chat" / "index.html"
    return FileResponse(str(html_path))


async def stream_chat_response(rag: rag_langchain.LangChainRAG, question: str) -> AsyncGenerator[str, None]:
    """Stream the chat response from RAG system."""
    try:
        # Use the streaming query method
        if hasattr(rag, 'query_stream'):
            sources = []
            
            async for chunk in rag.query_stream(question):
                chunk_type = chunk.get("type")
                
                if chunk_type == "token":
                    token = chunk.get("content", "")
                    # Send token immediately
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                elif chunk_type == "sources":
                    sources = chunk.get("sources", [])
                elif chunk_type == "done":
                    # Send sources and done signal
                    if sources:
                        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                elif chunk_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': chunk.get('message', 'Unknown error')})}\n\n"
        else:
            # Fallback: simulate streaming by chunking the response
            result = rag.query(question)
            answer = result.get("answer", "I couldn't generate an answer.")
            sources = result.get("sources", [])
            
            # Stream the answer character by character for smooth effect
            for char in answer:
                await asyncio.sleep(0.01)  # Small delay for streaming effect
                yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
            
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"❌ Error in stream_chat_response: {error_msg}")
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Web API endpoint using LangChain RAG with conversation memory (non-streaming)."""
    question = request.question.strip()
    if not question:
        return ChatResponse(
            answer="Please enter a non-empty question.",
            sources=[],
            conversation_id=request.conversation_id or "default",
            chat_history_length=0
        )
    
    # Get or create RAG instance for this conversation
    rag = get_or_create_rag(request.conversation_id)
    
    # Query the RAG system (includes conversation memory)
    result = rag.query(question)
    
    return ChatResponse(
        answer=result.get("answer", "I couldn't generate an answer."),
        sources=result.get("sources", []),
        conversation_id=rag.conversation_id,
        chat_history_length=len(result.get("chat_history", []))
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming endpoint for chat responses."""
    question = request.question.strip()
    if not question:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Please enter a non-empty question.'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    # Get or create RAG instance for this conversation
    rag = get_or_create_rag(request.conversation_id)
    
    return StreamingResponse(
        stream_chat_response(rag, question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )


@app.post("/api/chat/clear")
async def clear_conversation(conversation_id: Optional[str] = None) -> Dict[str, str]:
    """Clear conversation memory for a specific conversation."""
    if conversation_id and conversation_id in _rag_instances:
        _rag_instances[conversation_id].clear_memory()
        return {"status": "cleared", "conversation_id": conversation_id}
    elif not conversation_id and _default_rag:
        _default_rag.clear_memory()
        return {"status": "cleared", "conversation_id": "default"}
    else:
        return {"status": "not_found", "message": "Conversation not found"}


@app.get("/api/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "langchain-rag-web-chat"}


def main():
    """Entry point so this module can be run directly with -m."""
    import uvicorn

    uvicorn.run(
        "web_chat.server_langchain:app",
        host="0.0.0.0",
        port=8001,  # Different port from regular server
        reload=False,
    )


if __name__ == "__main__":
    main()

