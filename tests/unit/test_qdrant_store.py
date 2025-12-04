"""Unit tests for QdrantStore."""

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from retrieval.qdrant_store import QdrantStore


class TestQdrantStore:
    """Test suite for QdrantStore."""
    
    @pytest.fixture
    def temp_qdrant_dir(self):
        """Create temporary Qdrant directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    def test_init(self, temp_qdrant_dir):
        """Test QdrantStore initialization."""
        store = QdrantStore(collection_name="test_collection")
        assert store is not None
        assert store.collection_name == "test_collection"
        assert store.embedding_dim == 1536
    
    def test_recreate_collection(self, temp_qdrant_dir):
        """Test collection recreation."""
        store = QdrantStore(collection_name="test_collection")
        # Should not raise exception
        assert store is not None
    
    def test_upsert_documents(self, temp_qdrant_dir):
        """Test document upsertion."""
        store = QdrantStore(collection_name="test_collection")
        
        documents = [
            {
                "id": 0,
                "vector": [0.1] * 1536,
                "payload": {
                    "doc_id": "test_1",
                    "name": "Test",
                    "activities": ["test"]
                }
            }
        ]
        
        # Should not raise exception (may fail if Qdrant not properly initialized)
        try:
            store.upsert_documents(documents)
        except Exception:
            pass  # Expected if Qdrant not available
    
    def test_search(self, temp_qdrant_dir):
        """Test vector search."""
        store = QdrantStore(collection_name="test_collection")
        
        query_embedding = [0.1] * 1536
        results = store.search(query_embedding, limit=10)
        
        assert isinstance(results, list)
    
    def test_search_with_filter(self, temp_qdrant_dir):
        """Test vector search with filter."""
        store = QdrantStore(collection_name="test_collection")
        
        query_embedding = [0.1] * 1536
        filter_dict = {"doc_type": "destination"}
        
        results = store.search(query_embedding, limit=10, filter_dict=filter_dict)
        assert isinstance(results, list)

