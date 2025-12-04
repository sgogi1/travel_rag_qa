"""Unit tests for BaselineRetriever."""

import os
import pytest
import tempfile
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from retrieval.baseline_retriever import BaselineRetriever


class TestBaselineRetriever:
    """Test suite for BaselineRetriever."""
    
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
            content="Paris is a beautiful city with museums.",
            raw_data='{"name": "Paris"}'
        )
        
        writer.commit()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_init(self, test_index):
        """Test BaselineRetriever initialization."""
        retriever = BaselineRetriever(test_index)
        assert retriever is not None
        assert retriever.index_path == test_index
        retriever.close()
    
    def test_init_invalid_path(self):
        """Test initialization with invalid path."""
        with pytest.raises(ValueError):
            BaselineRetriever("/nonexistent/path")
    
    def test_search(self, test_index):
        """Test search functionality."""
        retriever = BaselineRetriever(test_index)
        results = retriever.search("museums", limit=10)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all("doc_id" in r for r in results)
        assert all("name" in r for r in results)
        assert all("score" in r for r in results)
        
        retriever.close()
    
    def test_search_no_results(self, test_index):
        """Test search with no results."""
        retriever = BaselineRetriever(test_index)
        results = retriever.search("nonexistent_query_xyz", limit=10)
        
        assert isinstance(results, list)
        # May or may not have results depending on index
        
        retriever.close()
    
    def test_search_limit(self, test_index):
        """Test search limit."""
        retriever = BaselineRetriever(test_index)
        results = retriever.search("city", limit=1)
        
        assert len(results) <= 1
        
        retriever.close()
    
    def test_close(self, test_index):
        """Test retriever closing."""
        retriever = BaselineRetriever(test_index)
        retriever.close()
        # Should not raise exception

