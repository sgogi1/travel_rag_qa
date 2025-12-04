"""
Qdrant vector store integration for storing and searching document embeddings.
"""

import os
import json
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models


class QdrantStore:
    """Manages Qdrant vector database for document embeddings."""
    
    def __init__(self, collection_name: str = "travel_documents", 
                 embedding_dim: int = 1536,
                 url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize Qdrant store.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_dim: Dimension of embedding vectors
            url: Qdrant server URL (None for local)
            api_key: Qdrant API key (for cloud)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize Qdrant client
        if url:
            # Cloud Qdrant
            self.client = QdrantClient(
                url=url,
                api_key=api_key
            )
        else:
            # Local Qdrant - use file-based with proper path
            qdrant_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qdrant_db")
            os.makedirs(qdrant_path, exist_ok=True)
            try:
                self.client = QdrantClient(path=qdrant_path, prefer_grpc=False)
            except Exception as e:
                # Fallback to in-memory if file lock issues
                print(f"Warning: Could not use file-based Qdrant ({e}), using in-memory")
                self.client = QdrantClient(location=":memory:")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Add documents with embeddings to Qdrant.
        
        Args:
            documents: List of document dictionaries
            embeddings: List of embedding vectors
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "doc_id": doc.get("doc_id", str(i)),
                    "doc_type": doc.get("doc_type", "unknown"),
                    "name": doc.get("name", ""),
                    "country": doc.get("country", ""),
                    "region": doc.get("region", ""),
                    "activities": doc.get("activities", []),
                    "description": doc.get("description", ""),
                    "raw_data": json.dumps(doc.get("raw_data", {}))
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Added {len(points)} documents to Qdrant")
        except Exception as e:
            print(f"Error adding documents to Qdrant: {e}")
            raise
    
    def search(self, query_embedding: List[float], limit: int = 10, 
               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_dict: Optional filters (e.g., {"doc_type": "destination"})
        
        Returns:
            List of similar documents with scores
        """
        # Build filter if provided
        query_filter = None
        if filter_dict:
            must_conditions = []
            for key, value in filter_dict.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)
        
        try:
            # Use query_points method (Qdrant client API)
            # Can pass vector directly or use Query object
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,  # Pass vector directly
                query_filter=query_filter,
                limit=limit
            )
            
            formatted_results = []
            for point in results.points:
                formatted_results.append({
                    "doc_id": point.payload.get("doc_id", ""),
                    "doc_type": point.payload.get("doc_type", ""),
                    "name": point.payload.get("name", ""),
                    "country": point.payload.get("country", ""),
                    "region": point.payload.get("region", ""),
                    "activities": point.payload.get("activities", []),
                    "description": point.payload.get("description", ""),
                    "score": point.score,  # Cosine similarity score
                    "document": json.loads(point.payload.get("raw_data", "{}"))
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []
    
    def delete_collection(self):
        """Delete the collection (use with caution)."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.points_count,
                "indexed": info.indexed_vectors_count
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}


if __name__ == "__main__":
    # Test Qdrant store
    store = QdrantStore()
    print(f"Qdrant store initialized with collection: {store.collection_name}")
    print(f"Embedding dimension: {store.embedding_dim}")

