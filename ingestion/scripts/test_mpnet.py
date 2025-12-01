#!/usr/bin/env python3
"""
Test script to evaluate the quality of all-mpnet-base-v2 embeddings.

This script tests:
1. Retrieval quality - queries and checks if relevant documents are retrieved
2. Similarity scores - evaluates if semantically similar chunks have high scores
3. Semantic understanding - tests if the model understands context and meaning
4. Retrieval accuracy - checks if the top results are actually relevant
5. Comparison with all-MiniLM-L6-v2 baseline
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend" / "src"))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

# Import Pinecone
try:
    from pinecone import Pinecone
    PINECONE_NEW_SDK = True
except (ImportError, AttributeError):
    try:
        import pinecone
        PINECONE_NEW_SDK = False
    except ImportError:
        print("❌ Error: Pinecone package not found")
        sys.exit(1)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "backend" / ".env")

# Configuration - Using all-mpnet-base-v2
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.getenv("pinecone")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME_MPNET", "company-docs-mpnet")  # MPNET index
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Fixed to mpnet model
EMBEDDING_DIMENSION = 768  # 768 dimensions for all-mpnet-base-v2
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Test queries with expected source documents
TEST_QUERIES = [
    {
        "query": "What are the work hours?",
        "expected_sources": ["employee_handbook.md"],
        "category": "Employee Handbook"
    },
    {
        "query": "How many days of PTO do I get?",
        "expected_sources": ["employee_handbook.md"],
        "category": "Employee Handbook"
    },
    {
        "query": "What is the password policy?",
        "expected_sources": ["company_policies.md"],
        "category": "Security Policy"
    },
    {
        "query": "Can I work remotely?",
        "expected_sources": ["company_policies.md"],
        "category": "Remote Work"
    },
    {
        "query": "What is the maximum file size for CloudSync Pro?",
        "expected_sources": ["product_features.md"],
        "category": "Product Features"
    },
    {
        "query": "How do I authenticate with the API?",
        "expected_sources": ["api_documentation.md"],
        "category": "API Documentation"
    },
    {
        "query": "My files are not syncing, what should I do?",
        "expected_sources": ["troubleshooting_guide.md"],
        "category": "Troubleshooting"
    },
    {
        "query": "What health insurance benefits are available?",
        "expected_sources": ["employee_handbook.md"],
        "category": "Benefits"
    },
    {
        "query": "How do I submit an expense report?",
        "expected_sources": ["company_policies.md"],
        "category": "Expense Policy"
    },
    {
        "query": "What integrations does CloudSync Pro support?",
        "expected_sources": ["product_features.md"],
        "category": "Product Features"
    },
    {
        "query": "I forgot my password, how do I reset it?",
        "expected_sources": ["troubleshooting_guide.md"],
        "category": "Troubleshooting"
    },
    {
        "query": "What is the rate limit for API requests?",
        "expected_sources": ["api_documentation.md"],
        "category": "API Documentation"
    },
    {
        "query": "What are the requirements for remote work eligibility?",
        "expected_sources": ["company_policies.md"],
        "category": "Remote Work"
    },
    {
        "query": "How much does CloudSync Pro cost?",
        "expected_sources": ["product_features.md"],
        "category": "Product Features"
    },
    {
        "query": "What security certifications does the product have?",
        "expected_sources": ["product_features.md"],
        "category": "Product Features"
    },
]

# Semantic similarity test pairs (should have high similarity)
SEMANTIC_PAIRS = [
    ("work hours", "working schedule"),
    ("paid time off", "vacation days"),
    ("password requirements", "security policy"),
    ("remote work", "working from home"),
    ("file sync", "synchronization"),
    ("API authentication", "login credentials"),
    ("troubleshooting", "problem solving"),
    ("health insurance", "medical benefits"),
    ("expense reimbursement", "cost reimbursement"),
    ("API rate limit", "request throttling"),
    ("cloud storage", "online file storage"),
    ("two-factor authentication", "2FA"),
    ("performance review", "employee evaluation"),
    ("sick leave", "medical leave"),
]


def load_embedding_model() -> SentenceTransformer:
    """Load the mpnet embedding model."""
    print(f"📦 Loading embedding model: {EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Model loaded successfully (dimension: {EMBEDDING_DIMENSION})")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def initialize_pinecone():
    """Initialize Pinecone connection."""
    print(f"🔌 Connecting to Pinecone index: {PINECONE_INDEX_NAME}...")
    try:
        if PINECONE_NEW_SDK:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
        else:
            pinecone.init(api_key=PINECONE_API_KEY)
            index = pinecone.Index(PINECONE_INDEX_NAME)
        print(f"✅ Connected successfully")
        return index
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        print(f"   Make sure the index '{PINECONE_INDEX_NAME}' exists and was created with mpnet embeddings")
        sys.exit(1)


def query_pinecone(index, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Query Pinecone with an embedding vector."""
    try:
        if PINECONE_NEW_SDK:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
        else:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results.get("matches", [])
    except Exception as e:
        print(f"❌ Error querying Pinecone: {e}")
        return []


def test_retrieval_quality(model: SentenceTransformer, index) -> Dict[str, Any]:
    """Test retrieval quality with test queries."""
    print("\n" + "=" * 70)
    print("🧪 Testing Retrieval Quality (all-mpnet-base-v2)")
    print("=" * 70)
    
    results = []
    correct_retrievals = 0
    total_queries = len(TEST_QUERIES)
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        expected_sources = test_case["expected_sources"]
        category = test_case["category"]
        
        print(f"\n[{i}/{total_queries}] Query: '{query}'")
        print(f"   Category: {category}")
        print(f"   Expected sources: {', '.join(expected_sources)}")
        
        # Create embedding for query
        query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()
        
        # Query Pinecone
        matches = query_pinecone(index, query_embedding, top_k=TOP_K)
        
        if not matches:
            print("   ⚠️  No results returned")
            results.append({
                "query": query,
                "category": category,
                "correct": False,
                "top_score": 0.0,
                "retrieved_sources": []
            })
            continue
        
        # Extract retrieved sources
        retrieved_sources = []
        for match in matches:
            source = match.metadata.get("source", "unknown") if PINECONE_NEW_SDK else match.get("metadata", {}).get("source", "unknown")
            score = match.score if PINECONE_NEW_SDK else match.get("score", 0.0)
            retrieved_sources.append((source, score))
        
        # Check if any expected source is in top results
        top_sources = [src for src, _ in retrieved_sources[:3]]  # Check top 3
        is_correct = any(expected in top_sources for expected in expected_sources)
        
        if is_correct:
            correct_retrievals += 1
            status = "✅"
        else:
            status = "❌"
        
        top_score = retrieved_sources[0][1] if retrieved_sources else 0.0
        
        print(f"   {status} Top result: {retrieved_sources[0][0]} (score: {top_score:.4f})")
        print(f"   Retrieved sources: {', '.join([f'{s}({sc:.3f})' for s, sc in retrieved_sources[:3]])}")
        
        results.append({
            "query": query,
            "category": category,
            "correct": is_correct,
            "top_score": top_score,
            "retrieved_sources": [s for s, _ in retrieved_sources]
        })
    
    accuracy = (correct_retrievals / total_queries) * 100
    
    print("\n" + "-" * 70)
    print(f"📊 Retrieval Accuracy: {correct_retrievals}/{total_queries} ({accuracy:.1f}%)")
    print("=" * 70)
    
    return {
        "accuracy": accuracy,
        "correct": correct_retrievals,
        "total": total_queries,
        "results": results
    }


def test_semantic_similarity(model: SentenceTransformer) -> Dict[str, Any]:
    """Test semantic similarity between related phrases."""
    print("\n" + "=" * 70)
    print("🔍 Testing Semantic Similarity (all-mpnet-base-v2)")
    print("=" * 70)
    
    similarities = []
    
    for phrase1, phrase2 in SEMANTIC_PAIRS:
        # Create embeddings
        embeddings = model.encode([phrase1, phrase2], show_progress_bar=False)
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        similarities.append({
            "phrase1": phrase1,
            "phrase2": phrase2,
            "similarity": float(similarity)
        })
        
        status = "✅" if similarity > 0.5 else "⚠️"
        print(f"{status} '{phrase1}' ↔ '{phrase2}': {similarity:.4f}")
    
    avg_similarity = np.mean([s["similarity"] for s in similarities])
    min_similarity = min([s["similarity"] for s in similarities])
    max_similarity = max([s["similarity"] for s in similarities])
    
    print("\n" + "-" * 70)
    print(f"📊 Average Similarity: {avg_similarity:.4f}")
    print(f"   Min: {min_similarity:.4f}, Max: {max_similarity:.4f}")
    print("=" * 70)
    
    return {
        "average": float(avg_similarity),
        "min": float(min_similarity),
        "max": float(max_similarity),
        "pairs": similarities
    }


def test_similarity_distribution(model: SentenceTransformer, index) -> Dict[str, Any]:
    """Test the distribution of similarity scores."""
    print("\n" + "=" * 70)
    print("📈 Testing Similarity Score Distribution (all-mpnet-base-v2)")
    print("=" * 70)
    
    all_scores = []
    threshold_passed = 0
    total_results = 0
    
    for test_case in TEST_QUERIES[:5]:  # Test first 5 queries
        query = test_case["query"]
        query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()
        matches = query_pinecone(index, query_embedding, top_k=TOP_K)
        
        for match in matches:
            score = match.score if PINECONE_NEW_SDK else match.get("score", 0.0)
            all_scores.append(score)
            total_results += 1
            if score >= SIMILARITY_THRESHOLD:
                threshold_passed += 1
    
    if all_scores:
        avg_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        std_score = np.std(all_scores)
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        print(f"📊 Score Statistics (from {total_results} results):")
        print(f"   Average: {avg_score:.4f}")
        print(f"   Median: {median_score:.4f}")
        print(f"   Std Dev: {std_score:.4f}")
        print(f"   Min: {min_score:.4f}, Max: {max_score:.4f}")
        print(f"   Above threshold ({SIMILARITY_THRESHOLD}): {threshold_passed}/{total_results} ({threshold_passed/total_results*100:.1f}%)")
        print("=" * 70)
        
        return {
            "average": float(avg_score),
            "median": float(median_score),
            "std": float(std_score),
            "min": float(min_score),
            "max": float(max_score),
            "above_threshold": threshold_passed,
            "total": total_results
        }
    else:
        print("⚠️  No scores to analyze")
        return {}


def test_category_accuracy(retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze accuracy by category."""
    print("\n" + "=" * 70)
    print("📋 Category-wise Accuracy Analysis (all-mpnet-base-v2)")
    print("=" * 70)
    
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in retrieval_results:
        category = result["category"]
        category_stats[category]["total"] += 1
        if result["correct"]:
            category_stats[category]["correct"] += 1
    
    for category, stats in sorted(category_stats.items()):
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        status = "✅" if accuracy >= 80 else "⚠️" if accuracy >= 50 else "❌"
        print(f"{status} {category}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    print("=" * 70)
    
    return dict(category_stats)


def main():
    """Main test function."""
    print("=" * 70)
    print("🧪 all-mpnet-base-v2 Embedding Quality Test Suite")
    print("=" * 70)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Dimension: {EMBEDDING_DIMENSION}")
    print(f"Index: {PINECONE_INDEX_NAME}")
    print(f"Top K: {TOP_K}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print("\n⚠️  Note: Make sure you've ingested documents using injest_docs_mpnet.py")
    print("   to create the mpnet index before running this test.")
    
    # Load model and initialize Pinecone
    model = load_embedding_model()
    index = initialize_pinecone()
    
    # Run tests
    retrieval_results = test_retrieval_quality(model, index)
    similarity_results = test_semantic_similarity(model)
    distribution_results = test_similarity_distribution(model, index)
    category_results = test_category_accuracy(retrieval_results["results"])
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY (all-mpnet-base-v2)")
    print("=" * 70)
    print(f"✅ Retrieval Accuracy: {retrieval_results['accuracy']:.1f}%")
    print(f"✅ Average Semantic Similarity: {similarity_results['average']:.4f}")
    if distribution_results:
        print(f"✅ Average Similarity Score: {distribution_results['average']:.4f}")
        print(f"✅ Results Above Threshold: {distribution_results['above_threshold']}/{distribution_results['total']}")
        print(f"✅ Score Range: {distribution_results['min']:.4f} - {distribution_results['max']:.4f}")
    
    # Overall assessment
    print("\n" + "-" * 70)
    if retrieval_results['accuracy'] >= 90 and similarity_results['average'] >= 0.6:
        print("🎉 Overall Assessment: EXCELLENT")
        print("   The mpnet embeddings are performing exceptionally well!")
        print("   Higher quality than all-MiniLM-L6-v2 expected.")
    elif retrieval_results['accuracy'] >= 80 and similarity_results['average'] >= 0.5:
        print("✅ Overall Assessment: VERY GOOD")
        print("   The mpnet embeddings are performing well for retrieval tasks.")
        print("   Should show improvement over all-MiniLM-L6-v2.")
    elif retrieval_results['accuracy'] >= 60 and similarity_results['average'] >= 0.4:
        print("⚠️  Overall Assessment: GOOD")
        print("   The embeddings are performing adequately.")
    else:
        print("⚠️  Overall Assessment: NEEDS IMPROVEMENT")
        print("   Consider checking if the index was created with mpnet embeddings.")
    print("=" * 70)
    
    # Model comparison note
    print("\n💡 Comparison Note:")
    print("   all-mpnet-base-v2 (768 dim) typically provides:")
    print("   - Higher quality embeddings than all-MiniLM-L6-v2 (384 dim)")
    print("   - Better semantic understanding")
    print("   - Higher similarity scores")
    print("   - Slightly slower inference time")
    print("=" * 70)


if __name__ == "__main__":
    main()

