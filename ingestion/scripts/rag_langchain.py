#!/usr/bin/env python3
"""
LangChain-based RAG implementation with conversation memory.

This file provides a complete RAG solution using LangChain that:
1. Connects to existing Pinecone vector store
2. Maintains conversation history/context
3. Supports multiple LLM providers (OpenAI, Anthropic, Ollama)
4. Supports local or OpenAI embeddings
5. Can be used as CLI or imported by web server

Usage:
    python rag_langchain.py
    python rag_langchain.py --question "What is the vacation policy?"
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "backend" / ".env")

# Try to import LangChain components
try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.llms import OpenAI as LangChainOpenAI
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Pinecone as LangChainPinecone
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some LangChain components not available: {e}")
    print("   Trying alternative imports...")
    try:
        # Try newer LangChain structure (0.1.x)
        from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory
        from langchain.llms.openai import OpenAI as LangChainOpenAI
        from langchain.chat_models.openai import ChatOpenAI
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.vectorstores import Pinecone as LangChainPinecone
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        print("❌ Error: LangChain not properly installed")
        print("   Install with: pip install langchain langchain-community langchain-openai")

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
    from langchain.embeddings import HuggingFaceEmbeddings
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import Pinecone
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

# Memory configuration
MEMORY_TYPE = os.getenv("MEMORY_TYPE", "buffer").lower()  # "buffer" or "summary"
MAX_MEMORY_TOKENS = int(os.getenv("MAX_MEMORY_TOKENS", "2000"))


class LangChainRAG:
    """LangChain-based RAG system with conversation memory."""
    
    def __init__(self):
        """Initialize the RAG system with all components."""
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        self.chain = None
        self.memory = None
        self.conversation_id = str(uuid4())
        
    def initialize_embeddings(self):
        """Initialize embedding model (local or OpenAI)."""
        if USE_LOCAL_EMBEDDINGS:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                print("❌ Error: sentence-transformers not available")
                sys.exit(1)
            print(f"📦 Loading local embedding model: {EMBEDDING_MODEL}")
            try:
                # Try using HuggingFaceEmbeddings wrapper
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
            except Exception:
                # Fallback: create a custom wrapper
                local_model = SentenceTransformer(EMBEDDING_MODEL)
                self.embeddings = CustomEmbeddings(local_model)
            print("✅ Embedding model loaded")
        else:
            if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
                print("❌ Error: OpenAI not available or API key missing")
                sys.exit(1)
            print(f"📦 Using OpenAI embedding model: {EMBEDDING_MODEL}")
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            print("✅ OpenAI embeddings initialized")
    
    def initialize_vectorstore(self):
        """Initialize Pinecone vector store connection."""
        try:
            print(f"🔌 Connecting to Pinecone index: {PINECONE_INDEX_NAME}...")
            
            # Initialize Pinecone client
            if PINECONE_NEW_SDK:
                pc = Pinecone(api_key=PINECONE_API_KEY)
            else:
                pinecone.init(api_key=PINECONE_API_KEY, environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"))
            
            # Create LangChain Pinecone vectorstore
            # Try different methods based on LangChain version
            try:
                # Newer LangChain API
                self.vectorstore = LangChainPinecone.from_existing_index(
                    index_name=PINECONE_INDEX_NAME,
                    embedding=self.embeddings
                )
            except Exception:
                try:
                    # Alternative: pass index directly
                    if PINECONE_NEW_SDK:
                        index = pc.Index(PINECONE_INDEX_NAME)
                    else:
                        index = pinecone.Index(PINECONE_INDEX_NAME)
                    
                    self.vectorstore = LangChainPinecone(
                        index=index,
                        embedding=self.embeddings,
                        text_key="text"
                    )
                except Exception as e2:
                    # Fallback: create a custom retriever
                    print(f"⚠️  Warning: Could not use LangChain Pinecone wrapper: {e2}")
                    print("   Using custom Pinecone retriever...")
                    if PINECONE_NEW_SDK:
                        index = pc.Index(PINECONE_INDEX_NAME)
                    else:
                        index = pinecone.Index(PINECONE_INDEX_NAME)
                    self.vectorstore = CustomPineconeRetriever(index, self.embeddings)
            
            print("✅ Vector store connected")
        except Exception as e:
            print(f"❌ Error connecting to vector store: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def initialize_llm(self):
        """Initialize LLM based on provider."""
        if LLM_PROVIDER == "openai":
            if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
                print("❌ Error: OpenAI not available or API key missing")
                sys.exit(1)
            print(f"🤖 Using OpenAI LLM: {LLM_MODEL}")
            # Use ChatOpenAI for better conversation handling
            try:
                self.llm = ChatOpenAI(
                    model_name=LLM_MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    openai_api_key=OPENAI_API_KEY
                )
            except Exception:
                # Fallback to older API
                self.llm = LangChainOpenAI(
                    model_name=LLM_MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    openai_api_key=OPENAI_API_KEY
                )
        
        elif LLM_PROVIDER == "anthropic":
            if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
                print("❌ Error: Anthropic not available or API key missing")
                sys.exit(1)
            print(f"🤖 Using Anthropic LLM: {LLM_MODEL}")
            try:
                self.llm = ChatAnthropic(
                    model=LLM_MODEL,
                    temperature=TEMPERATURE,
                    max_tokens_to_sample=MAX_TOKENS,
                    anthropic_api_key=ANTHROPIC_API_KEY
                )
            except Exception as e:
                print(f"⚠️  Error initializing Anthropic: {e}")
                print("   Falling back to OpenAI if available...")
                if OPENAI_AVAILABLE and OPENAI_API_KEY:
                    self.llm = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        openai_api_key=OPENAI_API_KEY
                    )
                else:
                    sys.exit(1)
        
        elif LLM_PROVIDER == "ollama":
            print(f"🤖 Using Ollama LLM: {LLM_MODEL}")
            try:
                from langchain.llms import Ollama
                self.llm = Ollama(
                    model=LLM_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=TEMPERATURE
                )
            except ImportError:
                print("⚠️  Ollama not available in LangChain, using custom wrapper...")
                self.llm = OllamaLLMWrapper(LLM_MODEL, OLLAMA_BASE_URL)
        
        else:
            print(f"❌ Error: Unknown LLM provider: {LLM_PROVIDER}")
            sys.exit(1)
        
        print("✅ LLM initialized")
    
    def initialize_memory(self):
        """Initialize conversation memory."""
        if MEMORY_TYPE == "summary":
            print("📝 Using summary memory (condenses long conversations)")
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=MAX_MEMORY_TOKENS
            )
        else:
            print("📝 Using buffer memory (stores full conversation)")
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        print("✅ Memory initialized")
    
    def initialize_chain(self):
        """Initialize the ConversationalRetrievalChain."""
        print("🔗 Building RAG chain...")
        
        # Custom prompt template for better context handling
        template = """Use the following pieces of context from company documents to answer the question.
If you don't know the answer based on the context, say so clearly.
Use the conversation history to understand follow-up questions and maintain context.

Context from documents:
{context}

Conversation history:
{chat_history}

Question: {question}

Answer based on the context and conversation history:"""
        
        try:
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "chat_history", "question"]
            )
        except Exception:
            # Fallback if PromptTemplate structure is different
            prompt = None
        
        # Create retriever
        # Note: Pinecone retriever doesn't support score_threshold in search_kwargs
        # We'll filter by score after retrieval if needed
        try:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": TOP_K_RESULTS
                    # Removed score_threshold - not supported by Pinecone retriever
                }
            )
        except Exception as e:
            print(f"⚠️  Warning: Error creating retriever: {e}")
            # If vectorstore doesn't have as_retriever, use it directly
            if hasattr(self.vectorstore, "as_retriever"):
                retriever = self.vectorstore
            else:
                # Use custom retriever
                retriever = self.vectorstore
        
        # Create the chain
        try:
            # Try with custom prompt first
            chain_kwargs = {
                "llm": self.llm,
                "retriever": retriever,
                "memory": self.memory,
                "return_source_documents": True,
                "verbose": False
            }
            
            # Add prompt if available and supported
            if prompt:
                try:
                    chain_kwargs["combine_docs_chain_kwargs"] = {"prompt": prompt}
                except Exception:
                    # Some versions don't support this parameter
                    pass
            
            self.chain = ConversationalRetrievalChain.from_llm(**chain_kwargs)
        except Exception as e:
            print(f"⚠️  Error creating chain: {e}")
            print("   Retrying with minimal configuration...")
            try:
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=False
                )
            except Exception as e2:
                print(f"❌ Failed to create chain: {e2}")
                import traceback
                traceback.print_exc()
                raise
        
        print("✅ RAG chain initialized")
    
    def initialize(self):
        """Initialize all components."""
        print("🔧 Initializing LangChain RAG system...")
        self.initialize_embeddings()
        self.initialize_vectorstore()
        self.initialize_llm()
        self.initialize_memory()
        self.initialize_chain()
        print("✅ All components initialized\n")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Returns:
            Dictionary with 'answer', 'sources', and 'chat_history'
        """
        if not self.chain:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        try:
            # Run the chain - use invoke() instead of __call__() to avoid deprecation warning
            if hasattr(self.chain, "invoke"):
                result = self.chain.invoke({"question": question})
            else:
                # Fallback for older LangChain versions
                result = self.chain({"question": question})
            
            # Extract answer
            answer = result.get("answer", "I couldn't generate an answer.")
            
            # Extract source documents
            source_docs = result.get("source_documents", [])
            sources = []
            for doc in source_docs:
                # Try to get source from metadata
                metadata = getattr(doc, "metadata", {})
                if isinstance(metadata, dict):
                    source = metadata.get("source", "unknown")
                    if source not in sources:
                        sources.append(source)
            
            # Get conversation history
            chat_history = []
            if self.memory:
                try:
                    messages = self.memory.chat_memory.messages if hasattr(self.memory, "chat_memory") else []
                    for msg in messages[-10:]:  # Last 10 messages
                        if hasattr(msg, "content"):
                            role = "user" if hasattr(msg, "type") and msg.type == "human" else "assistant"
                            chat_history.append({"role": role, "content": msg.content})
                except Exception:
                    pass
            
            return {
                "answer": answer,
                "sources": sources,
                "chat_history": chat_history
            }
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"❌ Error details: {error_msg}")
            traceback.print_exc()
            return {
                "answer": f"Error processing query: {error_msg}",
                "sources": [],
                "chat_history": []
            }
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            self.conversation_id = str(uuid4())
            print("🧹 Conversation memory cleared")


class CustomEmbeddings:
    """Custom embeddings wrapper for sentence-transformers."""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        return embedding.tolist()


class OllamaLLMWrapper:
    """Simple wrapper for Ollama when LangChain doesn't support it."""
    
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
    
    def __call__(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            import httpx
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120.0
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error calling Ollama: {e}"


class CustomPineconeRetriever:
    """Custom retriever for Pinecone when LangChain wrapper doesn't work."""
    
    def __init__(self, index, embeddings):
        self.index = index
        self.embeddings = embeddings
        self.search_kwargs = {"k": TOP_K_RESULTS}
    
    def as_retriever(self, search_kwargs=None):
        """Return self as retriever."""
        if search_kwargs:
            self.search_kwargs.update(search_kwargs)
        return self
    
    def get_relevant_documents(self, query: str) -> List[Any]:
        """Retrieve relevant documents."""
        try:
            # Create embedding for query
            if hasattr(self.embeddings, "embed_query"):
                query_embedding = self.embeddings.embed_query(query)
            else:
                query_embedding = self.embeddings.embed_documents([query])[0]
            
            # Query Pinecone
            k = self.search_kwargs.get("k", TOP_K_RESULTS)
            # Note: score_threshold filtering happens after retrieval
            threshold = SIMILARITY_THRESHOLD
            
            if PINECONE_NEW_SDK:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=k,
                    include_metadata=True
                )
                matches = results.matches
            else:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=k,
                    include_metadata=True
                )
                matches = results.get("matches", [])
            
            # Filter and format documents
            docs = []
            for match in matches:
                score = match.score if PINECONE_NEW_SDK else match.get("score", 0.0)
                # Only filter by threshold if we have matches, otherwise return all
                if not matches or score >= threshold:
                    metadata = match.metadata if PINECONE_NEW_SDK else match.get("metadata", {})
                    # Create a document-like object compatible with LangChain
                    try:
                        from langchain.schema import Document
                        doc = Document(
                            page_content=metadata.get("text", ""),
                            metadata=metadata
                        )
                    except ImportError:
                        try:
                            from langchain_core.documents import Document
                            doc = Document(
                                page_content=metadata.get("text", ""),
                                metadata=metadata
                            )
                        except ImportError:
                            # Fallback if Document class not available
                            doc = type('Document', (), {
                                'page_content': metadata.get("text", ""),
                                'metadata': metadata
                            })()
                    docs.append(doc)
            
            return docs
        except Exception as e:
            print(f"⚠️  Error retrieving documents: {e}")
            import traceback
            traceback.print_exc()
            return []


def interactive_mode(rag: LangChainRAG):
    """Run in interactive mode (REPL)."""
    print("\n" + "=" * 70)
    print("🚀 LangChain RAG - Interactive Mode")
    print("=" * 70)
    print("Enter your questions (type 'exit', 'quit', 'q', or 'clear' to clear memory)")
    print("=" * 70 + "\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'clear':
                rag.clear_memory()
                print("🧹 Memory cleared. You can start a new conversation.\n")
                continue
            
            # Query the RAG system
            print("🔍 Processing...")
            result = rag.query(question)
            
            # Display response
            print("\n" + "=" * 70)
            print("💬 Answer:")
            print("=" * 70)
            print(result["answer"])
            
            if result["sources"]:
                print("\n" + "-" * 70)
                print("📚 Sources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"   {i}. {source}")
            
            if result["chat_history"]:
                print("\n" + "-" * 70)
                print(f"💭 Conversation history: {len(result['chat_history'])} messages")
            
            print("=" * 70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def single_question_mode(question: str, rag: LangChainRAG):
    """Process a single question and exit."""
    print(f"\n❓ Question: {question}\n")
    
    print("🔍 Processing...")
    result = rag.query(question)
    
    print("\n" + "=" * 70)
    print("💬 Answer:")
    print("=" * 70)
    print(result["answer"])
    
    if result["sources"]:
        print("\n" + "-" * 70)
        print("📚 Sources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"   {i}. {source}")
    
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    if not LANGCHAIN_AVAILABLE:
        print("❌ Error: LangChain is not available")
        print("   Install with: pip install langchain langchain-community")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="LangChain RAG - Ask questions about company documents with conversation memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_langchain.py
  python rag_langchain.py --question "What is the vacation policy?"
  python rag_langchain.py -q "How do I submit expenses?"
        """
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Question to ask (if not provided, runs in interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = LangChainRAG()
    rag.initialize()
    
    # Run in appropriate mode
    if args.question:
        single_question_mode(args.question, rag)
    else:
        interactive_mode(rag)


if __name__ == "__main__":
    main()

