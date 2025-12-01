#!/usr/bin/env python3
"""
Script to ingest company documents into Pinecone using distiluse-base-multilingual-cased-v2 model.

This script uses the multilingual embedding model distiluse-base-multilingual-cased-v2 (512 dimensions)
which supports 50+ languages and is optimized for multilingual semantic search.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from dotenv import load_dotenv
import time

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ Error: sentence-transformers not installed")
    sys.exit(1)

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

# Configuration - Using distiluse-base-multilingual-cased-v2
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.getenv("pinecone")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME_MULTILINGUAL", "company-docs-multilingual")  # Different index name
EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"
EMBEDDING_DIMENSION = 512  # 512 for distiluse-base-multilingual-cased-v2
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
    """Split text into chunks with overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence boundary
        if end < len(text):
            search_start = max(start, end - 200)
            for i in range(end, search_start, -1):
                if text[i] in ['.', '\n', '!', '?']:
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks


def generate_id(text: str, source: str, chunk_index: int) -> str:
    """Generate a unique ID for a chunk."""
    content = f"{source}:{chunk_index}:{text[:50]}"
    return hashlib.md5(content.encode()).hexdigest()


def process_document(file_path: Path, pc, local_model: SentenceTransformer) -> Dict[str, Any]:
    """Process a single document: read, chunk, embed, and upload to Pinecone."""
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
    
    # Create embeddings
    print(f"   Creating embeddings for {len(chunks)} chunks...")
    try:
        embeddings = local_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        embeddings_list = embeddings.tolist()
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        return {"file": file_path.name, "chunks": 0, "status": "embedding_error"}
    
    # Prepare vectors for Pinecone
    vectors_to_upsert = []
    for j, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
        chunk_id = generate_id(chunk, file_path.stem, j)
        
        vector = {
            "id": chunk_id,
            "values": embedding,
            "metadata": {
                "source": file_path.name,
                "source_path": str(file_path),
                "chunk_index": j,
                "text": chunk,
                "chunk_size": len(chunk),
                "model": EMBEDDING_MODEL  # Track which model was used
            }
        }
        vectors_to_upsert.append(vector)
    
    # Upload to Pinecone
    if vectors_to_upsert:
        print(f"   Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
        try:
            if PINECONE_NEW_SDK:
                index = pc.Index(PINECONE_INDEX_NAME)
                
                # Upsert in batches of 100
                for i in range(0, len(vectors_to_upsert), 100):
                    batch = vectors_to_upsert[i:i + 100]
                    index.upsert(vectors=batch)
                    print(f"   Uploaded batch {i//100 + 1} ({len(batch)} vectors)")
            else:
                index = pinecone.Index(PINECONE_INDEX_NAME)
                
                for i in range(0, len(vectors_to_upsert), 100):
                    batch = vectors_to_upsert[i:i + 100]
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
    """Ensure the Pinecone index exists, create it if it doesn't."""
    try:
        if PINECONE_NEW_SDK:
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            
            if PINECONE_INDEX_NAME in existing_indexes:
                try:
                    index_info = pc.describe_index(PINECONE_INDEX_NAME)
                    existing_dim = index_info.dimension
                    if existing_dim != EMBEDDING_DIMENSION:
                        print(f"⚠️  Warning: Index '{PINECONE_INDEX_NAME}' exists with dimension {existing_dim}")
                        print(f"   But current embedding model uses dimension {EMBEDDING_DIMENSION}")
                        print(f"   Deleting old index and creating new one with correct dimension...")
                        pc.delete_index(PINECONE_INDEX_NAME)
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
            
            print("   Waiting for index to be ready...")
            while PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
                time.sleep(1)
            
            time.sleep(5)
            print(f"✅ Index '{PINECONE_INDEX_NAME}' created successfully")
            return True
        else:
            existing_indexes = pinecone.list_indexes()
            
            if PINECONE_INDEX_NAME in existing_indexes:
                print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists")
                return True
            
            print(f"📦 Creating index '{PINECONE_INDEX_NAME}'...")
            pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine"
            )
            
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
    print("🚀 Company Docs Copilot - Multilingual Embedding Ingestion")
    print("=" * 60)
    print(f"Model: {EMBEDDING_MODEL} (Dimension: {EMBEDDING_DIMENSION})")
    print(f"Index: {PINECONE_INDEX_NAME}")
    print("🌍 Supports 50+ languages")
    
    # Validate configuration
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Initialize clients
    print("\n🔌 Initializing clients...")
    if PINECONE_NEW_SDK:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    else:
        pinecone.init(api_key=PINECONE_API_KEY)
        pc = pinecone
    
    # Load embedding model
    print(f"📦 Loading embedding model: {EMBEDDING_MODEL}...")
    try:
        local_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Model loaded successfully (dimension: {EMBEDDING_DIMENSION})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
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
        
        result = process_document(file_path, pc, local_model)
        results.append(result)
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

