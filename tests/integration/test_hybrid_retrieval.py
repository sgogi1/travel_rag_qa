"""Integration tests for hybrid retrieval."""

import os
import pytest
import tempfile
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
from retrieval.hybrid_retriever import HybridRetriever
from unittest.mock import patch, MagicMock


class TestHybridRetrieval:
    """Integration tests for hybrid retrieval."""
    
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
    def test_hybrid_retriever_init(self, mock_openai, test_index):
        """Test HybridRetriever initialization."""
        retriever = HybridRetriever(
            bm25_index_path=test_index,
            qdrant_collection="test_collection"
        )
        assert retriever is not None
        retriever.close()
    
    @patch('retrieval.embedding_generator.OpenAI')
    @patch('retrieval.query_rewriter.OpenAI')
    def test_hybrid_search(self, mock_query_openai, mock_embedding_openai, test_index):
        """Test hybrid search."""
        # Mock query rewriter
        mock_query_response = MagicMock()
        mock_query_response.choices = [MagicMock()]
        mock_query_response.choices[0].message.content = '{"city": null, "country": null, "activities": ["museums"], "original_query": "museums"}'
        
        mock_query_client = MagicMock()
        mock_query_client.chat.completions.create.return_value = mock_query_response
        mock_query_openai.return_value = mock_query_client
        
        # Mock embedding
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [0.1] * 1536
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [mock_embedding_data]
        
        mock_embedding_client = MagicMock()
        mock_embedding_client.embeddings.create.return_value = mock_embedding_response
        mock_embedding_openai.return_value = mock_embedding_client
        
        retriever = HybridRetriever(
            bm25_index_path=test_index,
            qdrant_collection="test_collection"
        )
        
        result = retriever.search("museums", limit=10)
        
        assert "query" in result or "original_query" in result
        assert "results" in result
        assert "num_results" in result
        
        retriever.close()

