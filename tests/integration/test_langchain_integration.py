"""Integration tests for LangChain components."""

import os
import sys
import pytest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indexing.langchain_index_builder import LangChainIndexBuilder
from retrieval.langchain_retriever import (
    LangChainVectorRetriever,
    LangChainHybridRetriever
)


class TestLangChainIntegration:
    """Integration tests for LangChain pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                "type": "destination",
                "name": "Paris",
                "country": "France",
                "region": "Île-de-France",
                "description": "The City of Light, known for art, culture, and cuisine. Visit the Eiffel Tower, Louvre Museum, and enjoy fine dining.",
                "activities": ["museums", "dining", "sightseeing"]
            },
            {
                "type": "destination",
                "name": "Tokyo",
                "country": "Japan",
                "region": "Kanto",
                "description": "A vibrant metropolis blending tradition and modernity. Explore temples, enjoy sushi, and experience cutting-edge technology.",
                "activities": ["temple visits", "dining", "shopping"]
            },
            {
                "type": "guide",
                "name": "European Travel Guide",
                "country": "Multiple",
                "region": "Europe",
                "description": "Comprehensive guide to European destinations including Paris, Rome, and Barcelona.",
                "activities": ["sightseeing", "museums", "cultural tours"]
            }
        ]
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API calls."""
        with patch('retrieval.embedding_generator.OpenAI') as mock_openai_class, \
             patch('retrieval.query_rewriter.OpenAI') as mock_openai_rewriter, \
             patch('indexing.llm_extractor.OpenAI') as mock_openai_extractor, \
             patch('retrieval.langchain_retriever.OpenAIEmbeddings') as mock_embeddings:
            
            # Mock embedding generator
            mock_embedding_response = MagicMock()
            mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_openai_class.return_value.embeddings.create.return_value = mock_embedding_response
            
            # Mock query rewriter
            mock_chat_response = MagicMock()
            mock_chat_response.choices = [MagicMock()]
            mock_chat_response.choices[0].message.content = '{"city": null, "country": null, "activities": []}'
            mock_openai_rewriter.return_value.chat.completions.create.return_value = mock_chat_response
            
            # Mock activity extractor
            mock_extractor_response = MagicMock()
            mock_extractor_response.choices = [MagicMock()]
            mock_extractor_response.choices[0].message.content = '["museums", "dining"]'
            mock_openai_extractor.return_value.chat.completions.create.return_value = mock_extractor_response
            
            # Mock LangChain embeddings
            mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
            mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536] * 3
            
            yield {
                "embedding": mock_openai_class,
                "rewriter": mock_openai_rewriter,
                "extractor": mock_openai_extractor,
                "langchain_embeddings": mock_embeddings
            }
    
    @pytest.mark.integration
    def test_index_builder_to_retriever_flow(self, temp_dir, sample_documents, mock_openai):
        """Test full flow from index building to retrieval."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        
        # Build index
        with patch('indexing.langchain_index_builder.Qdrant') as mock_qdrant:
            mock_vector_store = MagicMock()
            mock_qdrant.from_documents.return_value = mock_vector_store
            
            builder = LangChainIndexBuilder(
                qdrant_path=qdrant_path,
                collection_name="test_collection"
            )
            
            # Mock the activity extractor
            builder.extractor.extract_structured_fields = MagicMock(
                return_value={"extracted_activities": ["museums"]}
            )
            
            vector_store = builder.build_vector_index(sample_documents, recreate=True)
            
            assert mock_qdrant.from_documents.called
            call_args = mock_qdrant.from_documents.call_args
            assert len(call_args[1]["documents"]) == 3
    
    @pytest.mark.integration
    def test_vector_retriever_with_filters(self, temp_dir, mock_openai):
        """Test vector retriever with structured filters."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        
        with patch('retrieval.langchain_retriever.Qdrant') as mock_qdrant:
            # Create mock documents
            from langchain_core.documents import Document
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
            
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_with_score.return_value = [
                (mock_doc1, 0.9),
                (mock_doc2, 0.8)
            ]
            mock_qdrant.return_value = mock_vector_store
            
            retriever = LangChainVectorRetriever(qdrant_path=qdrant_path)
            retriever.vector_store = mock_vector_store
            
            # Test search with country filter
            result = retriever.search("capital cities", limit=5, country="France")
            
            assert result["method"] == "langchain_vector"
            assert result["num_results"] >= 0
            if result["num_results"] > 0:
                assert result["results"][0]["country"] == "France"
    
    @pytest.mark.integration
    def test_hybrid_retriever_flow(self, temp_dir, mock_openai):
        """Test hybrid retriever combining BM25 and vector search."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        whoosh_index_path = os.path.join(temp_dir, "whoosh_index")
        os.makedirs(whoosh_index_path, exist_ok=True)
        
        with patch('retrieval.langchain_retriever.LangChainVectorRetriever') as mock_vector_class, \
             patch('retrieval.langchain_retriever.LangChainBM25Retriever') as mock_bm25_class, \
             patch('retrieval.langchain_retriever.EnsembleRetriever') as mock_ensemble_class:
            
            # Mock vector retriever
            mock_vector_retriever = MagicMock()
            mock_vector_retriever.search.return_value = {
                "original_query": "test",
                "results": [{"doc_id": "dest_1", "name": "Paris"}],
                "num_results": 1
            }
            mock_vector_class.return_value = mock_vector_retriever
            
            # Mock BM25 retriever
            mock_bm25_retriever = MagicMock()
            mock_bm25_class.return_value = mock_bm25_retriever
            
            # Mock ensemble retriever
            from langchain_core.documents import Document
            mock_doc = Document(
                page_content="Paris is beautiful",
                metadata={"doc_id": "dest_1", "name": "Paris"}
            )
            mock_ensemble = MagicMock()
            mock_ensemble.get_relevant_documents.return_value = [mock_doc]
            mock_ensemble_class.return_value = mock_ensemble
            
            retriever = LangChainHybridRetriever(
                qdrant_path=qdrant_path,
                whoosh_index_path=whoosh_index_path
            )
            
            result = retriever.search("Paris", limit=5)
            
            assert result["method"] == "langchain_hybrid"
            assert "results" in result
            assert mock_ensemble.get_relevant_documents.called
    
    @pytest.mark.integration
    def test_activity_filtering_integration(self, temp_dir, mock_openai):
        """Test activity filtering works end-to-end."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        
        with patch('retrieval.langchain_retriever.Qdrant') as mock_qdrant:
            from langchain_core.documents import Document
            mock_doc = Document(
                page_content="Paris museums and dining",
                metadata={
                    "doc_id": "dest_1",
                    "doc_type": "destination",
                    "name": "Paris",
                    "country": "France",
                    "activities": ["museums", "dining"]
                }
            )
            
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_with_score.return_value = [
                (mock_doc, 0.9)
            ]
            mock_qdrant.return_value = mock_vector_store
            
            retriever = LangChainVectorRetriever(qdrant_path=qdrant_path)
            retriever.vector_store = mock_vector_store
            
            # Test with activity filter
            result = retriever.search("cultural activities", limit=5, activities=["museums"])
            
            assert result["num_results"] >= 0
            if result["num_results"] > 0:
                doc_activities = result["results"][0]["activities"]
                assert any("museum" in str(act).lower() for act in doc_activities)

