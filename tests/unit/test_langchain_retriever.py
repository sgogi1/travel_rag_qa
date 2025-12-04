"""Unit tests for LangChain retrievers."""

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, Mock
from langchain_core.documents import Document

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from retrieval.langchain_retriever import (
    LangChainVectorRetriever,
    LangChainBM25Retriever,
    LangChainHybridRetriever
)


class TestLangChainVectorRetriever:
    """Unit tests for LangChainVectorRetriever."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock Qdrant vector store."""
        mock_store = MagicMock()
        mock_doc1 = Document(
            page_content="Paris is the capital of France",
            metadata={
                "doc_id": "dest_1",
                "doc_type": "destination",
                "name": "Paris",
                "country": "France",
                "region": "Île-de-France",
                "activities": ["museums", "dining"]
            }
        )
        mock_doc2 = Document(
            page_content="Tokyo is a vibrant city",
            metadata={
                "doc_id": "dest_2",
                "doc_type": "destination",
                "name": "Tokyo",
                "country": "Japan",
                "region": "Kanto",
                "activities": ["temple visits", "shopping"]
            }
        )
        mock_store.similarity_search_with_score.return_value = [
            (mock_doc1, 0.9),
            (mock_doc2, 0.8)
        ]
        return mock_store
    
    @pytest.fixture
    def retriever(self, temp_dir, mock_vector_store):
        """Create LangChainVectorRetriever instance."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        with patch('retrieval.langchain_retriever.Qdrant') as mock_qdrant, \
             patch('retrieval.langchain_retriever.OpenAIEmbeddings'), \
             patch('retrieval.langchain_retriever.QueryRewriter') as mock_rewriter, \
             patch('retrieval.langchain_retriever.ActivityMatcher') as mock_matcher:
            
            mock_qdrant.return_value = mock_vector_store
            mock_rewriter_instance = MagicMock()
            mock_rewriter_instance.rewrite_query.return_value = {
                "city": None,
                "country": None,
                "activities": []
            }
            mock_rewriter.return_value = mock_rewriter_instance
            
            mock_matcher_instance = MagicMock()
            mock_matcher_instance.expand_activity.return_value = ["museums"]
            mock_matcher.return_value = mock_matcher_instance
            
            retriever = LangChainVectorRetriever(qdrant_path=qdrant_path)
            retriever.vector_store = mock_vector_store
            retriever.query_rewriter = mock_rewriter_instance
            retriever.activity_matcher = mock_matcher_instance
            return retriever
    
    def test_init(self, temp_dir):
        """Test LangChainVectorRetriever initialization."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        with patch('retrieval.langchain_retriever.Qdrant') as mock_qdrant, \
             patch('retrieval.langchain_retriever.OpenAIEmbeddings'), \
             patch('retrieval.langchain_retriever.QueryRewriter'), \
             patch('retrieval.langchain_retriever.ActivityMatcher'):
            
            mock_qdrant.return_value = MagicMock()
            retriever = LangChainVectorRetriever(qdrant_path=qdrant_path)
            assert retriever.vector_store is not None
            assert retriever.query_rewriter is not None
            assert retriever.activity_matcher is not None
    
    def test_search_basic(self, retriever):
        """Test basic search without filters."""
        result = retriever.search("Paris", limit=5)
        
        assert result["original_query"] == "Paris"
        assert result["method"] == "langchain_vector"
        assert "results" in result
        assert "rewritten_query" in result
        assert retriever.vector_store.similarity_search_with_score.called
    
    def test_search_with_city_filter(self, retriever):
        """Test search with city filter."""
        result = retriever.search("Paris", limit=5, city="Paris")
        
        assert result["num_results"] >= 0
        # Should filter by city
        if result["num_results"] > 0:
            assert result["results"][0]["name"] == "Paris"
    
    def test_search_with_country_filter(self, retriever):
        """Test search with country filter."""
        result = retriever.search("capital cities", limit=5, country="France")
        
        assert result["num_results"] >= 0
        # Should filter by country
        if result["num_results"] > 0:
            assert result["results"][0]["country"] == "France"
    
    def test_search_with_activities_filter(self, retriever):
        """Test search with activities filter."""
        retriever.activity_matcher.expand_activity.return_value = ["museums", "museum"]
        result = retriever.search("cultural activities", limit=5, activities=["museums"])
        
        assert result["num_results"] >= 0
        # Should filter by activities
        if result["num_results"] > 0:
            doc_activities = result["results"][0]["activities"]
            assert any("museum" in str(act).lower() for act in doc_activities)
    
    def test_search_rewrites_query(self, retriever):
        """Test that query rewriting is applied."""
        retriever.query_rewriter.rewrite_query.return_value = {
            "city": "Paris",
            "country": "France",
            "activities": ["museums"]
        }
        
        result = retriever.search("Paris museums", limit=5)
        
        assert retriever.query_rewriter.rewrite_query.called
        assert result["rewritten_query"]["city"] == "Paris"
        assert result["rewritten_query"]["country"] == "France"


class TestLangChainBM25Retriever:
    """Unit tests for LangChainBM25Retriever."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_whoosh_retriever(self):
        """Mock Whoosh retriever."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = {
            "results": [
                {
                    "doc_id": "dest_1",
                    "doc_type": "destination",
                    "name": "Paris",
                    "country": "France",
                    "region": "Île-de-France",
                    "activities": ["museums"],
                    "description": "Paris is the capital of France"
                }
            ]
        }
        return mock_retriever
    
    @pytest.fixture
    def retriever(self, temp_dir, mock_whoosh_retriever):
        """Create LangChainBM25Retriever instance."""
        whoosh_index_path = os.path.join(temp_dir, "whoosh_index")
        with patch('retrieval.improved_retriever.ImprovedRetriever') as mock_improved:
            mock_improved.return_value = mock_whoosh_retriever
            retriever = LangChainBM25Retriever(whoosh_index_path)
            return retriever
    
    def test_init(self, temp_dir):
        """Test LangChainBM25Retriever initialization."""
        whoosh_index_path = os.path.join(temp_dir, "whoosh_index")
        with patch('retrieval.improved_retriever.ImprovedRetriever') as mock_improved:
            mock_retriever = MagicMock()
            mock_improved.return_value = mock_retriever
            retriever = LangChainBM25Retriever(whoosh_index_path)
            assert retriever._whoosh_retriever is not None
    
    def test_get_relevant_documents(self, retriever):
        """Test retrieving relevant documents."""
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        
        run_manager = MagicMock(spec=CallbackManagerForRetrieverRun)
        docs = retriever._get_relevant_documents("Paris", run_manager=run_manager)
        
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].metadata["name"] == "Paris"
        assert "Paris is the capital of France" in docs[0].page_content
    
    def test_close(self, retriever):
        """Test closing the retriever."""
        retriever.close()
        assert retriever._whoosh_retriever.close.called


class TestLangChainHybridRetriever:
    """Unit tests for LangChainHybridRetriever."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_vector_retriever(self):
        """Mock vector retriever."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = {
            "original_query": "test",
            "rewritten_query": None,
            "results": [
                {
                    "doc_id": "dest_1",
                    "doc_type": "destination",
                    "name": "Paris",
                    "country": "France",
                    "description": "Paris is beautiful"
                }
            ],
            "num_results": 1,
            "method": "langchain_vector"
        }
        return mock_retriever
    
    @pytest.fixture
    def mock_ensemble_retriever(self):
        """Mock ensemble retriever."""
        mock_ensemble = MagicMock()
        mock_doc = Document(
            page_content="Paris is the capital",
            metadata={
                "doc_id": "dest_1",
                "doc_type": "destination",
                "name": "Paris",
                "country": "France"
            }
        )
        mock_ensemble.get_relevant_documents.return_value = [mock_doc]
        return mock_ensemble
    
    @pytest.fixture
    def retriever(self, temp_dir, mock_vector_retriever, mock_ensemble_retriever):
        """Create LangChainHybridRetriever instance."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        whoosh_index_path = os.path.join(temp_dir, "whoosh_index")
        os.makedirs(whoosh_index_path, exist_ok=True)
        
        with patch('retrieval.langchain_retriever.LangChainVectorRetriever') as mock_vector_class, \
             patch('retrieval.langchain_retriever.LangChainBM25Retriever') as mock_bm25_class, \
             patch('retrieval.langchain_retriever.EnsembleRetriever') as mock_ensemble_class:
            
            mock_vector_class.return_value = mock_vector_retriever
            mock_bm25_class.return_value = MagicMock()
            mock_ensemble_class.return_value = mock_ensemble_retriever
            
            retriever = LangChainHybridRetriever(
                qdrant_path=qdrant_path,
                whoosh_index_path=whoosh_index_path
            )
            retriever.vector_retriever = mock_vector_retriever
            retriever.ensemble_retriever = mock_ensemble_retriever
            return retriever
    
    def test_init_with_whoosh(self, temp_dir, mock_vector_retriever):
        """Test initialization with Whoosh index."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        whoosh_index_path = os.path.join(temp_dir, "whoosh_index")
        os.makedirs(whoosh_index_path, exist_ok=True)
        
        with patch('retrieval.langchain_retriever.LangChainVectorRetriever') as mock_vector_class, \
             patch('retrieval.langchain_retriever.LangChainBM25Retriever') as mock_bm25_class, \
             patch('retrieval.langchain_retriever.EnsembleRetriever') as mock_ensemble_class:
            
            mock_vector_class.return_value = mock_vector_retriever
            mock_bm25_class.return_value = MagicMock()
            mock_ensemble_class.return_value = MagicMock()
            
            retriever = LangChainHybridRetriever(
                qdrant_path=qdrant_path,
                whoosh_index_path=whoosh_index_path
            )
            assert retriever.vector_retriever is not None
            assert retriever.bm25_retriever is not None
            assert retriever.ensemble_retriever is not None
    
    def test_init_without_whoosh(self, temp_dir, mock_vector_retriever):
        """Test initialization without Whoosh index."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        
        with patch('retrieval.langchain_retriever.LangChainVectorRetriever') as mock_vector_class:
            mock_vector_class.return_value = mock_vector_retriever
            
            retriever = LangChainHybridRetriever(
                qdrant_path=qdrant_path,
                whoosh_index_path=None
            )
            assert retriever.vector_retriever is not None
            assert retriever.bm25_retriever is None
            assert retriever.ensemble_retriever is None
    
    def test_search_with_ensemble(self, retriever):
        """Test search using ensemble retriever."""
        result = retriever.search("Paris", limit=5)
        
        assert result["original_query"] == "Paris"  # Should use the query parameter
        assert result["method"] == "langchain_hybrid"
        assert "results" in result
        assert retriever.ensemble_retriever.get_relevant_documents.called
    
    def test_search_fallback_to_vector(self, temp_dir, mock_vector_retriever):
        """Test search falls back to vector when ensemble not available."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        
        with patch('retrieval.langchain_retriever.LangChainVectorRetriever') as mock_vector_class:
            mock_vector_class.return_value = mock_vector_retriever
            
            retriever = LangChainHybridRetriever(
                qdrant_path=qdrant_path,
                whoosh_index_path=None
            )
            retriever.vector_retriever = mock_vector_retriever
            
            result = retriever.search("Paris", limit=5)
            
            assert result["method"] == "langchain_vector"
            assert mock_vector_retriever.search.called

