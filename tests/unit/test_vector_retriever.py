"""Unit tests for VectorRetriever."""

import os
import pytest
from unittest.mock import patch, MagicMock
from retrieval.vector_retriever import VectorRetriever


class TestVectorRetriever:
    """Test suite for VectorRetriever."""
    
    @pytest.fixture
    def retriever(self):
        """Create VectorRetriever instance."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            ret = VectorRetriever(collection_name="test_collection")
            # Mock the embedding generator and qdrant store
            ret.embedding_generator = MagicMock()
            ret.qdrant_store = MagicMock()
            return ret
    
    def test_init(self, retriever):
        """Test VectorRetriever initialization."""
        assert retriever is not None
        assert hasattr(retriever, 'embedding_generator')
        assert hasattr(retriever, 'qdrant_store')
    
    def test_search(self, retriever):
        """Test vector search."""
        # Mock embedding
        retriever.embedding_generator.generate_embedding.return_value = [0.1] * 1536
        
        # Mock Qdrant search
        retriever.qdrant_store.search.return_value = [
            {"doc_id": "test_1", "name": "Test", "score": 0.9}
        ]
        
        result = retriever.search("test query", limit=10)
        
        assert "query" in result
        assert "results" in result
        assert "num_results" in result
    
    def test_search_with_filters(self, retriever):
        """Test vector search with filters."""
        retriever.embedding_generator.generate_embedding.return_value = [0.1] * 1536
        retriever.qdrant_store.search.return_value = [
            {"doc_id": "test_1", "name": "Test", "score": 0.9}
        ]
        
        result = retriever.search("test", limit=10, doc_type="destination", country="France")
        
        assert "results" in result

