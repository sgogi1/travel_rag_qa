"""
Hybrid retriever combining BM25 (Whoosh) and Vector search (Qdrant).
Uses Reciprocal Rank Fusion (RRF) to combine results.
"""

import os
from typing import List, Dict, Any, Optional
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from improved_retriever import ImprovedRetriever
from vector_retriever import VectorRetriever


class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search."""
    
    def __init__(self, bm25_index_path: str, 
                 qdrant_collection: str = "travel_documents",
                 rrf_k: int = 60):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_index_path: Path to Whoosh BM25 index
            qdrant_collection: Name of Qdrant collection
            rrf_k: RRF constant (higher = more weight to top results)
        """
        self.bm25_retriever = ImprovedRetriever(bm25_index_path)
        self.vector_retriever = VectorRetriever(collection_name=qdrant_collection)
        self.rrf_k = rrf_k
    
    def reciprocal_rank_fusion(self, bm25_results: List[Dict], 
                                vector_results: List[Dict]) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
        
        Returns:
            Combined and re-ranked results
        """
        # Create score maps
        bm25_scores = {}
        vector_scores = {}
        
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.get("doc_id", "")
            if doc_id:
                bm25_scores[doc_id] = 1.0 / (self.rrf_k + rank)
        
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.get("doc_id", "")
            if doc_id:
                vector_scores[doc_id] = 1.0 / (self.rrf_k + rank)
        
        # Combine scores
        combined_scores = {}
        all_docs = {}
        
        # Add BM25 results
        for doc in bm25_results:
            doc_id = doc.get("doc_id", "")
            if doc_id:
                combined_scores[doc_id] = bm25_scores.get(doc_id, 0)
                all_docs[doc_id] = doc.copy()
        
        # Add vector results
        for doc in vector_results:
            doc_id = doc.get("doc_id", "")
            if doc_id:
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + vector_scores.get(doc_id, 0)
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc.copy()
        
        # Sort by combined score
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final results
        final_results = []
        for doc_id, rrf_score in sorted_docs:
            doc = all_docs[doc_id].copy()
            doc["rrf_score"] = rrf_score
            doc["bm25_score"] = bm25_scores.get(doc_id, 0)
            doc["vector_score"] = vector_scores.get(doc_id, 0)
            final_results.append(doc)
        
        return final_results
    
    def search(self, query: str, limit: int = 10, 
               use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Natural language query
            limit: Maximum number of results
            use_hybrid: If True, combine both methods; if False, use only BM25
        
        Returns:
            Dict with combined results
        """
        # Get BM25 results
        bm25_result = self.bm25_retriever.search(query, limit=limit)
        bm25_results = bm25_result.get("results", [])
        
        if not use_hybrid:
            return {
                "query": query,
                "method": "bm25_only",
                "results": bm25_results,
                "num_results": len(bm25_results),
                "bm25_rewritten_query": bm25_result.get("rewritten_query")
            }
        
        # Get vector results
        vector_result = self.vector_retriever.search(query, limit=limit)
        vector_results = vector_result.get("results", [])
        
        # Combine using RRF
        combined_results = self.reciprocal_rank_fusion(bm25_results, vector_results)
        
        return {
            "query": query,
            "method": "hybrid",
            "results": combined_results[:limit],
            "num_results": len(combined_results),
            "bm25_count": len(bm25_results),
            "vector_count": len(vector_results),
            "bm25_rewritten_query": bm25_result.get("rewritten_query")
        }
    
    def close(self):
        """Close retrievers."""
        self.bm25_retriever.close()


if __name__ == "__main__":
    # Test hybrid retriever
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes", "improved")
    
    if os.path.exists(index_path):
        retriever = HybridRetriever(index_path)
        
        test_query = "snorkeling in tropical waters"
        result = retriever.search(test_query, limit=5)
        
        print(f"Query: {test_query}")
        print(f"Method: {result['method']}")
        print(f"Results: {result['num_results']}")
        print(f"BM25: {result.get('bm25_count', 0)}, Vector: {result.get('vector_count', 0)}")
        
        for i, doc in enumerate(result['results'][:3], 1):
            print(f"{i}. {doc['name']} ({doc['doc_type']})")
            print(f"   RRF Score: {doc.get('rrf_score', 0):.4f}")
            print(f"   BM25: {doc.get('bm25_score', 0):.4f}, Vector: {doc.get('vector_score', 0):.4f}")
        
        retriever.close()
    else:
        print(f"Index not found at {index_path}")

