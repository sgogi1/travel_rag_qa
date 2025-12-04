"""Unit tests for HybridRetriever."""

import os
import pytest
import tempfile
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
from retrieval.hybrid_retriever import HybridRetriever
from unittest.mock import patch, MagicMock


class TestHybridRetriever:
    """Test suite for HybridRetriever."""
    
    @pytest.fixture
    def test_index(self):
        """Create test index."""
        temp_dir = tempfile.mkdtemp()
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            doc_type=STORED,
            name=STORED,
            content=TEXT,
            activities=KEYWORD(stored=True, lowercase=True, commas=True),
            raw_data=STORED
        )
        
        ix = index.create_in(temp_dir, schema)
        writer = ix.writer()
        writer.add_document(
            doc_id="test_1",
            doc_type="destination",
            name="Paris",
            content="Paris has museums.",
            activities="museums",
            raw_data='{}'
        )
        writer.commit()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('retrieval.embedding_generator.OpenAI')
    def test_init(self, mock_openai, test_index):
        """Test HybridRetriever initialization."""
        retriever = HybridRetriever(
            bm25_index_path=test_index,
            qdrant_collection="test_collection"
        )
        assert retriever is not None
        assert retriever.bm25_retriever is not None
        assert retriever.vector_retriever is not None
        retriever.close()
    
    @patch('retrieval.embedding_generator.OpenAI')
    def test_reciprocal_rank_fusion(self, mock_openai, test_index):
        """Test RRF ranking."""
        retriever = HybridRetriever(
            bm25_index_path=test_index,
            qdrant_collection="test_collection"
        )
        
        bm25_results = [
            {"doc_id": "doc1", "name": "Doc1", "score": 1.0},
            {"doc_id": "doc2", "name": "Doc2", "score": 0.8}
        ]
        
        vector_results = [
            {"doc_id": "doc2", "name": "Doc2", "score": 0.9},
            {"doc_id": "doc3", "name": "Doc3", "score": 0.7}
        ]
        
        combined = retriever.reciprocal_rank_fusion(bm25_results, vector_results)
        
        assert len(combined) == 3  # doc1, doc2, doc3
        assert all("rrf_score" in doc for doc in combined)
        # doc2 should have highest score (appears in both)
        doc2 = next(doc for doc in combined if doc["doc_id"] == "doc2")
        assert doc2["rrf_score"] > 0
        
        retriever.close()
    
    @patch('retrieval.embedding_generator.OpenAI')
    @patch('retrieval.query_rewriter.OpenAI')
    def test_search(self, mock_query_openai, mock_embedding_openai, test_index):
        """Test hybrid search."""
        retriever = HybridRetriever(
            bm25_index_path=test_index,
            qdrant_collection="test_collection"
        )
        
        # Mock BM25 retriever search
        retriever.bm25_retriever.search = MagicMock(return_value={
            "original_query": "museums",
            "rewritten_query": {"activities": ["museums"]},
            "results": [{"doc_id": "test_1", "name": "Paris", "score": 1.0}],
            "num_results": 1
        })
        
        # Mock vector retriever search
        retriever.vector_retriever.search = MagicMock(return_value={
            "query": "museums",
            "results": [],
            "num_results": 0
        })
        
        result = retriever.search("museums", limit=10)
        
        assert "query" in result or "original_query" in result
        assert "results" in result
        assert "num_results" in result
        
        retriever.close()

