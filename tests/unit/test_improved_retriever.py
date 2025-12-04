"""Unit tests for ImprovedRetriever."""

import os
import pytest
import tempfile
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
from retrieval.improved_retriever import ImprovedRetriever
from unittest.mock import patch, MagicMock


class TestImprovedRetriever:
    """Test suite for ImprovedRetriever."""
    
    @pytest.fixture
    def test_index(self):
        """Create test index."""
        temp_dir = tempfile.mkdtemp()
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            doc_type=STORED,
            name=STORED,
            country=STORED,
            region=STORED,
            content=TEXT,
            activities=KEYWORD(stored=True, lowercase=True, commas=True),
            extracted_activities=STORED,
            raw_data=STORED
        )
        
        ix = index.create_in(temp_dir, schema)
        writer = ix.writer()
        
        writer.add_document(
            doc_id="test_1",
            doc_type="destination",
            name="Paris",
            country="France",
            region="Paris, France",
            content="Paris has museums and art galleries.",
            activities="museums,art galleries",
            extracted_activities='["museums", "art galleries"]',
            raw_data='{"name": "Paris"}'
        )
        
        writer.commit()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('retrieval.query_rewriter.OpenAI')
    def test_init(self, mock_openai, test_index):
        """Test ImprovedRetriever initialization."""
        retriever = ImprovedRetriever(test_index)
        assert retriever is not None
        assert retriever.index_path == test_index
        retriever.close()
    
    @patch('retrieval.query_rewriter.OpenAI')
    def test_search(self, mock_openai, test_index):
        """Test search functionality."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"city": null, "country": null, "activities": ["museums"], "original_query": "museums"}'
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        retriever = ImprovedRetriever(test_index)
        result = retriever.search("museums", limit=10)
        
        assert "results" in result
        assert "rewritten_query" in result
        assert isinstance(result["results"], list)
        
        retriever.close()
    
    def test_search_with_filters(self, test_index):
        """Test search with structured filters."""
        retriever = ImprovedRetriever(test_index)
        results = retriever.search_with_filters(
            query="museums",
            city="Paris",
            activities=["museums"],
            limit=10
        )
        
        assert isinstance(results, list)
        retriever.close()
    
    def test_search_with_filters_no_results(self, test_index):
        """Test search with filters that yield no results."""
        retriever = ImprovedRetriever(test_index)
        results = retriever.search_with_filters(
            query="nonexistent",
            city="NonexistentCity",
            activities=["nonexistent"],
            limit=10
        )
        
        assert isinstance(results, list)
        retriever.close()
    
    def test_close(self, test_index):
        """Test retriever closing."""
        retriever = ImprovedRetriever(test_index)
        retriever.close()
        # Should not raise exception

