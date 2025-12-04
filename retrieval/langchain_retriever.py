"""
LangChain-based retrievers for travel documents.
Provides vector search, hybrid search, and structured filtering.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    # Fallback if EnsembleRetriever not available
    EnsembleRetriever = None
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query_rewriter import QueryRewriter
from activity_matcher import ActivityMatcher


class LangChainVectorRetriever:
    """LangChain-based vector retriever with structured filtering."""
    
    def __init__(self, qdrant_path: str = "./qdrant_db", collection_name: str = "travel_documents"):
        """
        Initialize LangChain vector retriever.
        
        Args:
            qdrant_path: Path to Qdrant database
            collection_name: Name of Qdrant collection
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Qdrant(
            embedding=self.embeddings,
            path=qdrant_path,
            collection_name=collection_name,
        )
        self.query_rewriter = QueryRewriter()
        self.activity_matcher = ActivityMatcher()
    
    def search(
        self,
        query: str,
        limit: int = 10,
        city: Optional[str] = None,
        country: Optional[str] = None,
        activities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform vector search with optional structured filters.
        
        Args:
            query: Search query
            limit: Maximum number of results
            city: Optional city filter
            country: Optional country filter
            activities: Optional activities filter
        
        Returns:
            Dict with search results and metadata
        """
        # Rewrite query if needed
        rewritten = self.query_rewriter.rewrite_query(query)
        if not city:
            city = rewritten.get("city")
        if not country:
            country = rewritten.get("country")
        if not activities:
            activities = rewritten.get("activities", [])
        
        # Expand activities for matching
        expanded_activities = set()
        if activities:
            for activity in activities:
                expanded_activities.update(self.activity_matcher.expand_activity(activity))
        
        # Perform vector search
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=limit * 2,  # Get more results for filtering
        )
        
        # Apply structured filters
        filtered_results = []
        for doc, score in results:
            metadata = doc.metadata
            
            # Filter by city
            if city:
                city_match = (
                    metadata.get("name", "").lower() == city.lower() or
                    metadata.get("region", "").lower().find(city.lower()) >= 0
                )
                if not city_match:
                    continue
            
            # Filter by country
            if country:
                country_match = metadata.get("country", "").lower() == country.lower()
                if not country_match:
                    continue
            
            # Filter by activities
            doc_activities = metadata.get("activities", [])
            if not isinstance(doc_activities, list):
                doc_activities = str(doc_activities).split(",") if doc_activities else []
            
            if expanded_activities:
                # Check if any expanded activity matches
                activity_match = False
                for doc_activity in doc_activities:
                    doc_activity_lower = str(doc_activity).lower().strip()
                    if doc_activity_lower in expanded_activities:
                        activity_match = True
                        break
                
                if not activity_match:
                    continue
            
            # Convert to result format
            result_dict = {
                "doc_id": metadata.get("doc_id"),
                "doc_type": metadata.get("doc_type"),
                "name": metadata.get("name"),
                "country": metadata.get("country"),
                "region": metadata.get("region"),
                "activities": doc_activities,
                "description": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": float(score),
                "raw_data": metadata.get("raw_data")
            }
            filtered_results.append(result_dict)
        
        # Limit results
        filtered_results = filtered_results[:limit]
        
        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "results": filtered_results,
            "num_results": len(filtered_results),
            "method": "langchain_vector"
        }


class LangChainBM25Retriever(BaseRetriever):
    """
    Custom BM25 retriever using Whoosh (since LangChain doesn't have native Whoosh support).
    This wraps the existing Whoosh-based retriever in a LangChain-compatible interface.
    """
    
    def __init__(self, whoosh_index_path: str, **kwargs):
        """Initialize BM25 retriever with Whoosh index."""
        from retrieval.improved_retriever import ImprovedRetriever
        super().__init__(**kwargs)
        # Store as private attribute to avoid Pydantic validation
        self._whoosh_retriever = ImprovedRetriever(whoosh_index_path)
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant documents using BM25."""
        results = self._whoosh_retriever.search(query, limit=10)
        
        # Convert to LangChain Documents
        langchain_docs = []
        for result in results.get("results", []):
            doc = Document(
                page_content=result.get("description", ""),
                metadata={
                    "doc_id": result.get("doc_id"),
                    "doc_type": result.get("doc_type"),
                    "name": result.get("name"),
                    "country": result.get("country"),
                    "region": result.get("region"),
                    "activities": result.get("activities", []),
                }
            )
            langchain_docs.append(doc)
        
        return langchain_docs
    
    def close(self):
        """Close the underlying retriever."""
        self._whoosh_retriever.close()


class LangChainHybridRetriever:
    """Hybrid retriever combining BM25 and Vector search using LangChain."""
    
    def __init__(
        self,
        qdrant_path: str = "./qdrant_db",
        collection_name: str = "travel_documents",
        whoosh_index_path: Optional[str] = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            qdrant_path: Path to Qdrant database
            collection_name: Name of Qdrant collection
            whoosh_index_path: Path to Whoosh index (for BM25)
        """
        # Vector retriever
        self.vector_retriever = LangChainVectorRetriever(qdrant_path, collection_name)
        
        # BM25 retriever (if Whoosh index available)
        if whoosh_index_path and os.path.exists(whoosh_index_path):
            self.bm25_retriever = LangChainBM25Retriever(whoosh_index_path)
            # Create ensemble retriever if available
            if EnsembleRetriever:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, self.vector_retriever.vector_store.as_retriever()],
                    weights=[0.5, 0.5],  # Equal weight for BM25 and vector
                )
            else:
                self.ensemble_retriever = None
        else:
            self.bm25_retriever = None
            self.ensemble_retriever = None
    
    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            Dict with combined search results
        """
        if self.ensemble_retriever:
            # Use ensemble retriever
            docs = self.ensemble_retriever.get_relevant_documents(query)
            
            # Convert to result format
            results = []
            for doc in docs[:limit]:
                metadata = doc.metadata
                results.append({
                    "doc_id": metadata.get("doc_id"),
                    "doc_type": metadata.get("doc_type"),
                    "name": metadata.get("name"),
                    "country": metadata.get("country"),
                    "region": metadata.get("region"),
                    "activities": metadata.get("activities", []),
                    "description": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                })
            
            return {
                "original_query": query,
                "rewritten_query": None,  # Can add query rewriting if needed
                "results": results,
                "num_results": len(results),
                "method": "langchain_hybrid"
            }
        else:
            # Fall back to vector search only
            return self.vector_retriever.search(query, limit=limit)


if __name__ == "__main__":
    # Test LangChain retrievers
    print("Testing LangChain Vector Retriever...")
    
    retriever = LangChainVectorRetriever()
    result = retriever.search("snorkeling in tropical waters", limit=5)
    
    print(f"\nQuery: {result['original_query']}")
    print(f"Results: {result['num_results']}")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"  {i}. {r['name']} ({r['doc_type']}) - Score: {r['score']:.4f}")

