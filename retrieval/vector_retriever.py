"""
Vector-based retriever using Qdrant for semantic search.
"""

import os
from typing import List, Dict, Any, Optional
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from embedding_generator import EmbeddingGenerator
from qdrant_store import QdrantStore


class VectorRetriever:
    """Vector-based retriever using Qdrant for semantic similarity search."""
    
    def __init__(self, collection_name: str = "travel_documents"):
        """
        Initialize vector retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
        """
        self.embedding_generator = EmbeddingGenerator()
        self.qdrant_store = QdrantStore(collection_name=collection_name)
    
    def search(self, query: str, limit: int = 10, 
               doc_type: Optional[str] = None,
               country: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Natural language query
            limit: Maximum number of results
            doc_type: Optional filter by document type
            country: Optional filter by country
        
        Returns:
            Dict with results and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Build filters
        filter_dict = {}
        if doc_type:
            filter_dict["doc_type"] = doc_type
        if country:
            filter_dict["country"] = country
        
        # Search in Qdrant
        results = self.qdrant_store.search(
            query_embedding=query_embedding,
            limit=limit,
            filter_dict=filter_dict if filter_dict else None
        )
        
        return {
            "query": query,
            "method": "vector",
            "results": results,
            "num_results": len(results)
        }
    
    def search_with_activities(self, query: str, activities: List[str], 
                               limit: int = 10) -> Dict[str, Any]:
        """
        Search with activity-based filtering (post-filtering).
        
        Args:
            query: Natural language query
            activities: List of activity filters
            limit: Maximum number of results
        
        Returns:
            Dict with filtered results
        """
        # Perform vector search
        result = self.search(query, limit=limit * 2)  # Get more results for filtering
        
        # Filter by activities
        if activities:
            filtered_results = []
            for doc in result["results"]:
                doc_activities = doc.get("activities", [])
                # Check if any requested activity matches
                if any(act.lower() in str(doc_activities).lower() for act in activities):
                    filtered_results.append(doc)
                if len(filtered_results) >= limit:
                    break
            result["results"] = filtered_results[:limit]
            result["num_results"] = len(filtered_results)
        
        return result


if __name__ == "__main__":
    # Test vector retriever
    retriever = VectorRetriever()
    
    test_query = "snorkeling in tropical waters"
    result = retriever.search(test_query, limit=5)
    
    print(f"Query: {test_query}")
    print(f"Results: {result['num_results']}")
    for i, doc in enumerate(result['results'][:3], 1):
        print(f"{i}. {doc['name']} ({doc['doc_type']}) - Score: {doc['score']:.4f}")

