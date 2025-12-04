"""Integration tests for retrieval systems."""

import os
import pytest
import tempfile
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
from retrieval.baseline_retriever import BaselineRetriever
from retrieval.improved_retriever import ImprovedRetriever
from retrieval.activity_matcher import ActivityMatcher
from unittest.mock import patch, MagicMock


class TestRetrieval:
    """Integration tests for retrieval."""
    
    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary index directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def baseline_index(self, temp_index_dir):
        """Create baseline index for testing."""
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            doc_type=STORED,
            name=STORED,
            country=STORED,
            region=STORED,
            content=TEXT,
            raw_data=STORED
        )
        
        ix = index.create_in(temp_index_dir, schema)
        writer = ix.writer()
        
        writer.add_document(
            doc_id="dest_0",
            doc_type="destination",
            name="Paris",
            country="France",
            region="Paris, France",
            content="Paris is a beautiful city with museums and art galleries.",
            raw_data='{"name": "Paris", "country": "France"}'
        )
        
        writer.add_document(
            doc_id="guide_0",
            doc_type="guide",
            name="Jean-Pierre",
            country="France",
            region="Paris, France",
            content="Expert guide for city tours in Paris.",
            raw_data='{"name": "Jean-Pierre", "country": "France"}'
        )
        
        writer.commit()
        return temp_index_dir
    
    @pytest.fixture
    def improved_index(self, temp_index_dir):
        """Create improved index for testing."""
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
        
        ix = index.create_in(temp_index_dir, schema)
        writer = ix.writer()
        
        writer.add_document(
            doc_id="dest_0",
            doc_type="destination",
            name="Paris",
            country="France",
            region="Paris, France",
            content="Paris is a beautiful city with museums and art galleries.",
            activities="museums,art galleries,city tours",
            extracted_activities='["museums", "art galleries"]',
            raw_data='{"name": "Paris", "country": "France"}'
        )
        
        writer.commit()
        return temp_index_dir
    
    def test_baseline_retriever_search(self, baseline_index):
        """Test baseline retriever search."""
        retriever = BaselineRetriever(baseline_index)
        results = retriever.search("museums", limit=10)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all("doc_id" in r for r in results)
        assert all("name" in r for r in results)
        
        retriever.close()
    
    @patch('retrieval.query_rewriter.OpenAI')
    def test_improved_retriever_search(self, mock_openai, improved_index):
        """Test improved retriever search."""
        # Mock query rewriter
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"city": null, "country": null, "activities": ["museums"], "original_query": "museums"}'
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        retriever = ImprovedRetriever(improved_index)
        result = retriever.search("museums", limit=10)
        
        assert "results" in result
        assert "rewritten_query" in result
        assert isinstance(result["results"], list)
        
        retriever.close()
    
    @patch('retrieval.query_rewriter.OpenAI')
    def test_improved_retriever_with_filters(self, mock_openai, improved_index):
        """Test improved retriever with structured filters."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"city": "Paris", "country": null, "activities": ["museums"], "original_query": "museums in Paris"}'
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        retriever = ImprovedRetriever(improved_index)
        result = retriever.search_with_filters(
            query="museums",
            city="Paris",
            activities=["museums"],
            limit=10
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        retriever.close()
    
    def test_activity_matcher_integration(self):
        """Test activity matcher integration."""
        matcher = ActivityMatcher()
        
        # Test with real data
        query_activities = ["tour"]
        data_activities = ["tours", "city tours", "photography tours"]
        
        assert matcher.match_activities(query_activities, data_activities) is True
        
        matches = matcher.find_matching_activities(query_activities, data_activities)
        assert len(matches) > 0

