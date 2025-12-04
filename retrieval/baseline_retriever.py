"""
Baseline retrieval system using naive BM25 text search.
Demonstrates limitations of simple text matching.
"""

import os
import json
from typing import List, Dict, Any, Optional
from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.query import And, Or, Term


class BaselineRetriever:
    """Naive BM25-based retriever (baseline approach)."""
    
    def __init__(self, index_path: str):
        """
        Initialize retriever with index path.
        
        Args:
            index_path: Path to the Whoosh index directory
        """
        if not os.path.exists(index_path):
            raise ValueError(f"Index not found at {index_path}")
        
        self.index_path = index_path
        self.ix = index.open_dir(index_path)
        self.searcher = self.ix.searcher()
        self.query_parser = QueryParser("content", schema=self.ix.schema)
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform naive text search.
        
        Args:
            query: Natural language query
            limit: Maximum number of results
        
        Returns:
            List of retrieved documents with scores
        """
        # Parse query for BM25 search
        parsed_query = self.query_parser.parse(query)
        
        # Search
        results = self.searcher.search(parsed_query, limit=limit)
        
        # Format results
        formatted_results = []
        for result in results:
            doc = json.loads(result["raw_data"])
            formatted_results.append({
                "doc_id": result["doc_id"],
                "doc_type": result["doc_type"],
                "name": result["name"],
                "country": result.get("country", ""),
                "region": result.get("region", ""),
                "score": result.score,
                "document": doc
            })
        
        return formatted_results
    
    def close(self):
        """Close the searcher."""
        if self.searcher:
            self.searcher.close()


if __name__ == "__main__":
    # Test baseline retriever
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes", "baseline")
    
    if os.path.exists(index_path):
        retriever = BaselineRetriever(index_path)
        
        test_queries = [
            "snorkeling",
            "wine tasting",
            "Lisbon",
            "city tours in Paris"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            results = retriever.search(query, limit=5)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['name']} ({result['doc_type']}) - Score: {result['score']:.3f}")
                print(f"   Region: {result.get('region', 'N/A')}")
        
        retriever.close()
    else:
        print(f"Index not found at {index_path}. Please build the index first.")

