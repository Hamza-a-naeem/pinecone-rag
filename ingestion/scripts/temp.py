#!/usr/bin/env python3
"""
Command-line interface for RAG (Retrieval Augmented Generation) Q&A.

This script allows users to ask questions and get answers based on the
company documents stored in the vector database.

Usage:
    python rag_cli.py
    python rag_cli.py --question "What is the company policy on vacation?"
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from dotenv import load_dotenv

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import Pinecone - try new SDK first, fall back to old
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_NEW_SDK = True
except (ImportError, AttributeError):
    try:
        import pinecone
        PINECONE_NEW_SDK = False
    except ImportError:
        print("❌ Error: Pinecone package not found. Please install it:")
        print("   pip install pinecone-client")
        sys.exit(1)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "backend" / ".env")

# Configuration from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.getenv("pinecone")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Embedding configuration
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
if USE_LOCAL_EMBEDDINGS:
    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model and env_model.startswith("text-embedding"):
        EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    else:
        EMBEDDING_MODEL = env_model or "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# Retrieval configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Generation parameters
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))


def initialize_pinecone():
    """Initialize and return Pinecone index."""
    try:
        if PINECONE_NEW_SDK:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
        else:
            pinecone.init(api_key=PINECONE_API_KEY, environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"))
            index = pinecone.Index(PINECONE_INDEX_NAME)
        
        print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
        return index
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        sys.exit(1)


def initialize_embedding_model():
    """Initialize embedding model (local or OpenAI)."""
    if USE_LOCAL_EMBEDDINGS:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("❌ Error: sentence-transformers not available. Install it with: pip install sentence-transformers")
            sys.exit(1)
        print(f"📦 Loading local embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Embedding model loaded")
        return model, None
    else:
        if not OPENAI_AVAILABLE:
            print("❌ Error: OpenAI package not available. Install it with: pip install openai")
            sys.exit(1)
        if not OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY not set")
            sys.exit(1)
        print(f"📦 Using OpenAI embedding model: {EMBEDDING_MODEL}")
        client = OpenAI(api_key=OPENAI_API_KEY)
        return None, client


def initialize_llm_client():
    """Initialize LLM client based on provider."""
    if LLM_PROVIDER == "openai":
        if not OPENAI_AVAILABLE:
            print("❌ Error: OpenAI package not available. Install it with: pip install openai")
            sys.exit(1)
        if not OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY not set")
            sys.exit(1)
        print(f"🤖 Using OpenAI LLM: {LLM_MODEL}")
        return OpenAI(api_key=OPENAI_API_KEY), "openai"
    
    elif LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            print("❌ Error: Anthropic package not available. Install it with: pip install anthropic")
            sys.exit(1)
        if not ANTHROPIC_API_KEY:
            print("❌ Error: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        print(f"🤖 Using Anthropic LLM: {LLM_MODEL}")
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY), "anthropic"
    
    elif LLM_PROVIDER == "ollama":
        print(f"🤖 Using Ollama LLM: {LLM_MODEL}")
        print(f"   Base URL: {OLLAMA_BASE_URL}")
        return OLLAMA_BASE_URL, "ollama"
    
    else:
        print(f"❌ Error: Unknown LLM provider: {LLM_PROVIDER}")
        print("   Supported providers: openai, anthropic, ollama")
        sys.exit(1)


def create_embedding(query: str, local_model=None, openai_client=None) -> List[float]:
    """Create embedding for a query."""
    if USE_LOCAL_EMBEDDINGS and local_model:
        try:
            embedding = local_model.encode([query], show_progress_bar=False)[0]
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            return []
    elif not USE_LOCAL_EMBEDDINGS and openai_client:
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            return []
    else:
        print("❌ Error: No embedding method available")
        return []


def normalize_query(
    query: str,
    llm_client,
    provider: str,
    max_output_tokens: int = 128,
) -> str:
    """
    Normalize a user query using the configured LLM.

    Goals:
      1. Fix grammar, spelling, slang.
      2. Extract the actual intent.
      3. Expand meaning if the query is too short or vague.
      4. Keep the original meaning intact (do not change the user's intent).

    The normalized query is meant to be used for retrieval (embeddings / vector
    search), while the original query can still be used for answering.
    """
    original = (query or "").strip()
    if not original or llm_client is None:
        return original

    # Short-circuit very long inputs to avoid wasting tokens
    if len(original) > 1000:
        return original

    prompt = (
        "You normalize user search queries for a Retrieval-Augmented Generation (RAG) system.\n\n"
        "Rewrite the query in clear, grammatically correct English, suitable for semantic search over company documentation.\n"
        "Requirements:\n"
        "1. Fix grammar, spelling, and slang.\n"
        "2. Make the user's intent explicit.\n"
        "3. Expand vague or very short queries into a more descriptive search query.\n"
        "4. Do NOT change the underlying meaning or intent.\n"
        "5. Output ONLY the normalized query text, no explanations.\n\n"
        f"Original query:\n\"\"\"{original}\"\"\"\n\n"
        "Normalized query:"
    )

    try:
        if provider == "openai":
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You normalize search queries for a company documentation assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_output_tokens,
                temperature=0.0,
            )
            normalized = response.choices[0].message.content.strip()
            return normalized or original

        if provider == "anthropic":
            message = llm_client.messages.create(
                model=LLM_MODEL,
                max_tokens=max_output_tokens,
                temperature=0.0,
                system="You normalize search queries for a company documentation assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            normalized = message.content[0].text.strip()
            return normalized or original

        if provider == "ollama":
            try:
                import httpx
            except ImportError:
                return original

            response = httpx.post(
                f"{llm_client}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_output_tokens,
                    },
                },
                timeout=60.0,
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
            return text or original

        # Unknown provider: fall back to original
        return original

    except Exception:
        # On any failure, just return the original query so behaviour degrades gracefully
        return original

def query_vector_db(index, query_embedding: List[float], top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
    """Query vector database and return relevant documents."""
    try:
        if PINECONE_NEW_SDK:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            matches = results.matches
        else:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            matches = results.get("matches", [])
        
        # Filter by similarity threshold
        filtered_matches = []
        for match in matches:
            score = match.score if PINECONE_NEW_SDK else match.get("score", 0.0)
            if score >= SIMILARITY_THRESHOLD:
                metadata = match.metadata if PINECONE_NEW_SDK else match.get("metadata", {})
                filtered_matches.append({
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", "unknown"),
                    "score": score
                })
        
        return filtered_matches
    except Exception as e:
        print(f"❌ Error querying vector database: {e}")
        return []


def build_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Build context string from retrieved documents."""
    if not retrieved_docs:
        return ""
    
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        context_parts.append(f"[Document {i} - Source: {source}]\n{text}\n")
    
    return "\n".join(context_parts)


def generate_response(
    question: str,
    context: str,
    llm_client,
    provider: str
) -> str:
    """Generate response using LLM with RAG context."""
    
    if not context:
        # If no context, still try to answer but mention it
        prompt = f"""You are a helpful assistant. A user asked a question, but no relevant documents were found in the knowledge base.

Question: {question}

Please provide a helpful response, but clearly state that you couldn't find relevant information in the company documents."""
    else:
        # Build the prompt with context
        prompt = f"""You are a helpful assistant that answers questions based on the provided company documents.

Use the following context from the company documents to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context from documents:
{context}

Question: {question}

Please provide a clear, concise answer based on the context above. If you reference specific information, mention which document it came from."""

    try:
        if provider == "openai":
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on company documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        
        elif provider == "anthropic":
            message = llm_client.messages.create(
                model=LLM_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system="You are a helpful assistant that answers questions based on company documents.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text.strip()
        
        elif provider == "ollama":
            try:
                import httpx
            except ImportError:
                return "❌ Error: httpx package required for Ollama. Install it with: pip install httpx"
            
            try:
                response = httpx.post(
                    f"{llm_client}/api/generate",
                    json={
                        "model": LLM_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": TEMPERATURE,
                            "num_predict": MAX_TOKENS
                        }
                    },
                    timeout=120.0
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except httpx.RequestError as e:
                return f"❌ Error connecting to Ollama: {e}"
            except Exception as e:
                return f"❌ Error generating response with Ollama: {e}"
        
        else:
            return "❌ Error: Unknown LLM provider"
    
    except Exception as e:
        return f"❌ Error generating response: {e}"


def format_response(response: str, sources: List[str]):
    """Format and display the response with sources."""
    print("\n" + "=" * 70)
    print("💬 Response:")
    print("=" * 70)
    print(response)
    
    if sources:
        print("\n" + "-" * 70)
        print("📚 Sources:")
        for i, source in enumerate(set(sources), 1):
            print(f"   {i}. {source}")
    print("=" * 70 + "\n")


def interactive_mode(index, local_model, openai_embedding_client, llm_client, llm_provider):
    """Run in interactive mode (REPL)."""
    print("\n" + "=" * 70)
    print("🚀 RAG CLI - Interactive Mode")
    print("=" * 70)
    print("Enter your questions (type 'exit', 'quit', or 'q' to exit)")
    print("=" * 70 + "\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Create embedding
            print("🔍 Searching documents...")
            query_embedding = create_embedding(question, local_model, openai_embedding_client)
            
            if not query_embedding:
                print("❌ Failed to create embedding")
                continue
            
            # Query vector database
            retrieved_docs = query_vector_db(index, query_embedding, TOP_K_RESULTS)
            
            if not retrieved_docs:
                print("⚠️  No relevant documents found. Try rephrasing your question.")
                continue
            
            print(f"📄 Found {len(retrieved_docs)} relevant document(s)")
            
            # Build context
            context = build_context(retrieved_docs)
            sources = [doc.get("source", "unknown") for doc in retrieved_docs]
            
            # Generate response
            print("🤖 Generating response...")
            response = generate_response(question, context, llm_client, llm_provider)
            
            # Display response
            format_response(response, sources)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def single_question_mode(question: str, index, local_model, openai_embedding_client, llm_client, llm_provider):
    """Process a single question and exit."""
    print(f"\n❓ Question: {question}\n")
    
    # Create embedding
    print("🔍 Searching documents...")
    query_embedding = create_embedding(question, local_model, openai_embedding_client)
    
    if not query_embedding:
        print("❌ Failed to create embedding")
        return
    
    # print(f"🔍 Query embedding: {query_embedding}")
    # Query vector database
    retrieved_docs = query_vector_db(index, query_embedding, TOP_K_RESULTS)
    
    if not retrieved_docs:
        print("⚠️  No relevant documents found.")
        return
    
    print(f"📄 Found {len(retrieved_docs)} relevant document(s)")
    
    # Build context
    context = build_context(retrieved_docs)
    sources = [doc.get("source", "unknown") for doc in retrieved_docs]
    
    # Generate response
    print("🤖 Generating response...")
    response = generate_response(question, context, llm_client, llm_provider)
    
    # Display response
    format_response(response, sources)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG CLI - Ask questions about company documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_cli.py
  python rag_cli.py --question "What is the vacation policy?"
  python rag_cli.py -q "How do I submit expenses?"
        """
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Question to ask (if not provided, runs in interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    print("🔧 Initializing components...")
    index = initialize_pinecone()
    local_model, openai_embedding_client = initialize_embedding_model()
    llm_client, llm_provider = initialize_llm_client()
    print("✅ All components initialized\n")
    
    # Run in appropriate mode
    if args.question:
        single_question_mode(args.question, index, local_model, openai_embedding_client, llm_client, llm_provider)
    else:
        interactive_mode(index, local_model, openai_embedding_client, llm_client, llm_provider)


if __name__ == "__main__":
    main()

