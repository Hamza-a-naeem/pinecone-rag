#!/usr/bin/env python3
"""
Script to ingest company documents into Pinecone vector database.

This script:
1. Reads markdown document files
2. Chunks the text appropriately
3. Creates embeddings using OpenAI
4. Uploads to Pinecone index
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import uuid

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from dotenv import load_dotenv
import time

# Try to import OpenAI (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
    # Fall back to old SDK (pinecone-client v2.x)
    try:
        import pinecone
        PINECONE_NEW_SDK = False
    except ImportError:
        print("❌ Error: Pinecone package not found. Please install it:")
        print("   pip install pinecone-client")
        sys.exit(1)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "backend" / ".env")

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.getenv("pinecone")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding configuration - use local by default to avoid API costs
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"

# Set default model and dimension based on embedding type
# Only use env var if explicitly set, otherwise use defaults based on embedding type
if USE_LOCAL_EMBEDDINGS:
    # Local embeddings using sentence-transformers
    # Check if EMBEDDING_MODEL is explicitly set, otherwise use local default
    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model and env_model.startswith("text-embedding"):  # OpenAI model name
        # User set OpenAI model but wants local embeddings - use local default
        EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        print(f"⚠️  Note: EMBEDDING_MODEL was set to OpenAI model '{env_model}', but using local model '{EMBEDDING_MODEL}' instead")
    else:
        EMBEDDING_MODEL = env_model or "all-MiniLM-L6-v2"  # Local model by default
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # 384 for all-MiniLM-L6-v2
else:
    # OpenAI embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))  # 1536 for OpenAI small

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Document files to ingest
DOCUMENT_FILES = [
    "employee_handbook.md",
    "company_policies.md",
    "product_features.md",
    "api_documentation.md",
    "troubleshooting_guide.md",
]


def read_markdown_file(file_path: Path) -> str:
    """Read content from a markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence boundary (period, newline, or paragraph break)
        if end < len(text):
            # Look for sentence endings within the last 200 chars of the chunk
            search_start = max(start, end - 200)
            for i in range(end, search_start, -1):
                if text[i] in ['.', '\n', '!', '?']:
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks


def create_embeddings(texts: List[str], client=None, local_model=None) -> List[List[float]]:
    """
    Create embeddings for a list of texts using either OpenAI or local model.
    
    Args:
        texts: List of text strings to embed
        client: OpenAI client instance (if using OpenAI)
        local_model: SentenceTransformer model (if using local)
    
    Returns:
        List of embedding vectors
    """
    if USE_LOCAL_EMBEDDINGS and local_model:
        try:
            # Local embeddings using sentence-transformers
            embeddings = local_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error creating local embeddings: {e}")
            return []
    elif not USE_LOCAL_EMBEDDINGS and client:
        try:
            # OpenAI embeddings
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error creating OpenAI embeddings: {e}")
            return []
    else:
        print("Error: No embedding method available")
        return []


def generate_id(text: str, source: str, chunk_index: int) -> str:
    """Generate a unique ID for a chunk."""
    content = f"{source}:{chunk_index}:{text[:50]}"
    return hashlib.md5(content.encode()).hexdigest()


def process_document(file_path: Path, pc, openai_client=None, local_model=None) -> Dict[str, Any]:
    """
    Process a single document: read, chunk, embed, and upload to Pinecone.
    
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n📄 Processing: {file_path.name}")
    
    # Read document
    text = read_markdown_file(file_path)
    if not text:
        print(f"⚠️  Skipping {file_path.name} - file is empty or could not be read")
        return {"file": file_path.name, "chunks": 0, "status": "skipped"}
    
    # Chunk the text
    chunks = chunk_text(text)
    print(f"   Created {len(chunks)} chunks")
    
    if not chunks:
        print(f"⚠️  No chunks created for {file_path.name}")
        return {"file": file_path.name, "chunks": 0, "status": "no_chunks"}
    
    # Create embeddings in batches
    batch_size = 100  # OpenAI allows up to 2048 inputs per request, but we'll batch for safety
    vectors_to_upsert = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        print(f"   Creating embeddings for batch {i//batch_size + 1} ({len(batch_chunks)} chunks)...")
        
        embeddings = create_embeddings(batch_chunks, openai_client, local_model)
        
        if not embeddings or len(embeddings) != len(batch_chunks):
            print(f"⚠️  Error: Expected {len(batch_chunks)} embeddings, got {len(embeddings)}")
            continue
        
        # Prepare vectors for Pinecone
        for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
            chunk_id = generate_id(chunk, file_path.stem, i + j)
            
            vector = {
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "source": file_path.name,
                    "source_path": str(file_path),
                    "chunk_index": i + j,
                    "text": chunk,
                    "chunk_size": len(chunk)
                }
            }
            vectors_to_upsert.append(vector)
    
    # Upload to Pinecone
    if vectors_to_upsert:
        print(f"   Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
        try:
            if PINECONE_NEW_SDK:
                index = pc.Index(PINECONE_INDEX_NAME)
                
                # Upsert in batches of 100 (Pinecone limit)
                for i in range(0, len(vectors_to_upsert), 100):
                    batch = vectors_to_upsert[i:i + 100]
                    index.upsert(vectors=batch)
                    print(f"   Uploaded batch {i//100 + 1} ({len(batch)} vectors)")
            else:
                # Old SDK format
                index = pinecone.Index(PINECONE_INDEX_NAME)
                
                # Convert to old format: list of tuples (id, values) or (id, values, metadata)
                for i in range(0, len(vectors_to_upsert), 100):
                    batch = vectors_to_upsert[i:i + 100]
                    # Old SDK expects list of tuples: (id, vector, metadata_dict)
                    old_format_batch = [
                        (v["id"], v["values"], v.get("metadata", {})) for v in batch
                    ]
                    index.upsert(vectors=old_format_batch)
                    print(f"   Uploaded batch {i//100 + 1} ({len(batch)} vectors)")
            
            print(f"✅ Successfully uploaded {len(vectors_to_upsert)} vectors from {file_path.name}")
            return {
                "file": file_path.name,
                "chunks": len(vectors_to_upsert),
                "status": "success"
            }
        except Exception as e:
            print(f"❌ Error uploading to Pinecone: {e}")
            return {
                "file": file_path.name,
                "chunks": len(vectors_to_upsert),
                "status": "error",
                "error": str(e)
            }
    
    return {"file": file_path.name, "chunks": 0, "status": "no_vectors"}


def ensure_index_exists(pc) -> bool:
    """
    Ensure the Pinecone index exists, create it if it doesn't.
    Checks dimension compatibility and recreates if needed.
    
    Returns:
        True if index exists or was created successfully
    """
    try:
        if PINECONE_NEW_SDK:
            # New SDK (pinecone package v3+)
            # List existing indexes
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            
            if PINECONE_INDEX_NAME in existing_indexes:
                # Check if dimension matches
                try:
                    index_info = pc.describe_index(PINECONE_INDEX_NAME)
                    existing_dim = index_info.dimension
                    if existing_dim != EMBEDDING_DIMENSION:
                        print(f"⚠️  Warning: Index '{PINECONE_INDEX_NAME}' exists with dimension {existing_dim}")
                        print(f"   But current embedding model uses dimension {EMBEDDING_DIMENSION}")
                        print(f"   Deleting old index and creating new one with correct dimension...")
                        pc.delete_index(PINECONE_INDEX_NAME)
                        # Wait for deletion
                        while PINECONE_INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
                            time.sleep(1)
                        time.sleep(2)
                    else:
                        print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists (dimension: {EMBEDDING_DIMENSION})")
                        return True
                except Exception as e:
                    print(f"⚠️  Could not check index dimension: {e}")
                    print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists")
                    return True
            
            # Create index if it doesn't exist
            print(f"📦 Creating index '{PINECONE_INDEX_NAME}'...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            print("   Waiting for index to be ready...")
            while PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
                time.sleep(1)
            
            # Additional wait to ensure index is fully ready
            time.sleep(5)
            print(f"✅ Index '{PINECONE_INDEX_NAME}' created successfully")
            return True
        else:
            # Old SDK (pinecone-client v2.x)
            existing_indexes = pinecone.list_indexes()
            
            if PINECONE_INDEX_NAME in existing_indexes:
                print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists")
                return True
            
            # Create index if it doesn't exist
            print(f"📦 Creating index '{PINECONE_INDEX_NAME}'...")
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine"
            )
            
            # Wait for index to be ready
            print("   Waiting for index to be ready...")
            while PINECONE_INDEX_NAME not in pinecone.list_indexes():
                time.sleep(1)
            
            time.sleep(5)
            print(f"✅ Index '{PINECONE_INDEX_NAME}' created successfully")
            return True
        
    except Exception as e:
        print(f"❌ Error ensuring index exists: {e}")
        return False


def main():
    """Main function to orchestrate the ingestion process."""
    print("=" * 60)
    print("🚀 Company Docs Copilot - Document Ingestion Script")
    print("=" * 60)
    
    # Validate configuration
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("   Please set PINECONE_API_KEY in your .env file")
        sys.exit(1)
    
    # Initialize clients
    print("\n🔌 Initializing clients...")
    if PINECONE_NEW_SDK:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    else:
        # Old SDK initialization
        pinecone.init(api_key=PINECONE_API_KEY)
        pc = pinecone  # For compatibility with function signatures
    
    # Initialize embedding model
    openai_client = None
    local_model = None
    
    if USE_LOCAL_EMBEDDINGS:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("❌ Error: sentence-transformers not installed")
            print("   Install it with: pip install sentence-transformers")
            sys.exit(1)
        print(f"📦 Loading local embedding model: {EMBEDDING_MODEL}...")
        try:
            local_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"✅ Local embedding model loaded (dimension: {EMBEDDING_DIMENSION})")
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            sys.exit(1)
    else:
        if not OPENAI_AVAILABLE:
            print("❌ Error: OpenAI package not installed")
            print("   Install it with: pip install openai")
            sys.exit(1)
        if not OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("   Please set OPENAI_API_KEY in your .env file")
            sys.exit(1)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"✅ Using OpenAI embeddings (model: {EMBEDDING_MODEL}, dimension: {EMBEDDING_DIMENSION})")
    
    # Ensure index exists
    if not ensure_index_exists(pc):
        print("❌ Failed to ensure index exists. Exiting.")
        sys.exit(1)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    documents_dir = project_root
    
    # Process each document
    results = []
    for doc_file in DOCUMENT_FILES:
        file_path = documents_dir / doc_file
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            results.append({"file": doc_file, "chunks": 0, "status": "not_found"})
            continue
        
        result = process_document(file_path, pc, openai_client, local_model)
        results.append(result)
        
        # Small delay between documents to avoid rate limits
        time.sleep(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Ingestion Summary")
    print("=" * 60)
    
    total_chunks = 0
    successful = 0
    failed = 0
    
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['file']}: {result['chunks']} chunks - {result['status']}")
        
        if result["status"] == "success":
            successful += 1
            total_chunks += result["chunks"]
        else:
            failed += 1
    
    print("\n" + "-" * 60)
    print(f"Total documents processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed/Skipped: {failed}")
    print(f"Total chunks uploaded: {total_chunks}")
    print("=" * 60)
    
    if successful > 0:
        print("\n✅ Ingestion completed successfully!")
    else:
        print("\n❌ No documents were successfully ingested.")


if __name__ == "__main__":
    main()

