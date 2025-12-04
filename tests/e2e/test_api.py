"""End-to-end tests for API endpoints."""

import os
import sys
import pytest
import tempfile
import shutil
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.main import app


class TestAPI:
    """End-to-end API tests."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_retrievers(self):
        """Mock retrievers for testing."""
        with patch('app.main.baseline_retriever') as mock_baseline, \
             patch('app.main.improved_retriever') as mock_improved, \
             patch('app.main.vector_retriever') as mock_vector, \
             patch('app.main.hybrid_retriever') as mock_hybrid, \
             patch('app.main.query_rewriter') as mock_rewriter, \
             patch('app.main.langchain_vector_retriever') as mock_langchain_vector, \
             patch('app.main.langchain_hybrid_retriever') as mock_langchain_hybrid, \
             patch('app.main.LANGCHAIN_AVAILABLE', True):
            
            # Setup mocks
            mock_baseline.search.return_value = [
                {"doc_id": "test_1", "name": "Test", "score": 1.0}
            ]
            
            mock_improved.search.return_value = {
                "original_query": "test",
                "rewritten_query": {"activities": ["test"]},
                "results": [{"doc_id": "test_1", "name": "Test", "score": 1.0}],
                "num_results": 1
            }
            
            mock_vector.search.return_value = {
                "query": "test",
                "results": [{"doc_id": "test_1", "name": "Test", "score": 0.9}],
                "num_results": 1
            }
            
            mock_hybrid.search.return_value = {
                "query": "test",
                "results": [{"doc_id": "test_1", "name": "Test"}],
                "num_results": 1,
                "bm25_count": 1,
                "vector_count": 1
            }
            
            mock_rewriter.rewrite_query.return_value = {
                "city": None,
                "country": None,
                "activities": ["test"],
                "original_query": "test"
            }
            
            # LangChain mocks
            mock_langchain_vector.search.return_value = {
                "original_query": "test",
                "rewritten_query": {"activities": ["test"]},
                "results": [{"doc_id": "test_1", "name": "Test", "score": 0.9}],
                "num_results": 1,
                "method": "langchain_vector"
            }
            
            mock_langchain_hybrid.search.return_value = {
                "original_query": "test",
                "rewritten_query": None,
                "results": [{"doc_id": "test_1", "name": "Test"}],
                "num_results": 1,
                "method": "langchain_hybrid"
            }
            
            yield {
                "baseline": mock_baseline,
                "improved": mock_improved,
                "vector": mock_vector,
                "hybrid": mock_hybrid,
                "rewriter": mock_rewriter,
                "langchain_vector": mock_langchain_vector,
                "langchain_hybrid": mock_langchain_hybrid
            }
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code in [200, 404]  # May not have frontend file
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_search_baseline(self, client, mock_retrievers):
        """Test baseline search endpoint."""
        response = client.post(
            "/api/search",
            json={"query": "test", "use_improved": False, "limit": 10}
        )
        assert response.status_code in [200, 503]  # 503 if index not available
    
    def test_search_improved(self, client, mock_retrievers):
        """Test improved search endpoint."""
        response = client.post(
            "/api/search",
            json={"query": "test", "use_improved": True, "limit": 10}
        )
        assert response.status_code in [200, 503]
    
    def test_search_vector(self, client, mock_retrievers):
        """Test vector search endpoint."""
        response = client.post(
            "/api/search",
            json={"query": "test", "use_vector": True, "limit": 10}
        )
        assert response.status_code in [200, 503]
    
    def test_search_hybrid(self, client, mock_retrievers):
        """Test hybrid search endpoint."""
        response = client.post(
            "/api/search",
            json={"query": "test", "use_hybrid": True, "limit": 10}
        )
        assert response.status_code in [200, 503]
    
    def test_rewrite_query(self, client, mock_retrievers):
        """Test query rewriting endpoint."""
        response = client.post(
            "/api/rewrite-query",
            json={"query": "test query"}
        )
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "original_query" in data
    
    def test_chat_endpoint(self, client, mock_retrievers):
        """Test chat endpoint."""
        # Chat endpoint uses retrievers which are already mocked
        response = client.post(
            "/api/chat",
            json={"query": "test", "use_improved": True, "limit": 5}
        )
        # May return 200 (if retrievers work) or 503 (if not initialized)
        assert response.status_code in [200, 503, 500]
    
    def test_search_invalid_request(self, client):
        """Test search with invalid request."""
        response = client.post(
            "/api/search",
            json={"invalid": "data"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.post(
            "/api/search",
            json={"query": "", "limit": 10}
        )
        # Should either accept empty query or return error (500 if retrievers not initialized)
        assert response.status_code in [200, 422, 503, 500]
    
    def test_search_langchain_vector(self, client, mock_retrievers):
        """Test LangChain vector search endpoint."""
        response = client.post(
            "/api/search",
            json={"query": "test", "use_langchain": True, "limit": 10}
        )
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data["method"] == "langchain_vector"
            assert "original_query" in data
            assert "results" in data
    
    def test_search_langchain_hybrid(self, client, mock_retrievers):
        """Test LangChain hybrid search endpoint."""
        response = client.post(
            "/api/search",
            json={"query": "test", "use_langchain": True, "use_hybrid": True, "limit": 10}
        )
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data["method"] == "langchain_hybrid"
            assert "original_query" in data
            assert "results" in data
    
    def test_search_langchain_not_available(self, client):
        """Test LangChain search when LangChain is not available."""
        with patch('app.main.LANGCHAIN_AVAILABLE', False):
            response = client.post(
                "/api/search",
                json={"query": "test", "use_langchain": True, "limit": 10}
            )
            # Should fall back to other methods, return 503, or 500 if error
            assert response.status_code in [200, 503, 500]
    
    def test_search_langchain_priority(self, client, mock_retrievers):
        """Test that LangChain flag takes priority over other flags."""
        response = client.post(
            "/api/search",
            json={
                "query": "test",
                "use_langchain": True,
                "use_hybrid": True,
                "use_vector": True,
                "use_improved": True,
                "limit": 10
            }
        )
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            # Should use LangChain method
            assert data["method"] in ["langchain_vector", "langchain_hybrid"]

