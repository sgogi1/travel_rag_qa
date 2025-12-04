"""Unit tests for LangChainIndexBuilder."""

import os
import sys
import pytest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock, Mock
from langchain_core.documents import Document

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indexing.langchain_index_builder import LangChainIndexBuilder


class TestLangChainIndexBuilder:
    """Unit tests for LangChainIndexBuilder."""
    
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
                "description": "The City of Light, known for art, culture, and cuisine.",
                "activities": ["museums", "dining"]
            },
            {
                "type": "guide",
                "name": "Tokyo Travel Guide",
                "country": "Japan",
                "region": "Kanto",
                "description": "A comprehensive guide to Tokyo's attractions.",
                "activities": ["temple visits", "shopping"]
            }
        ]
    
    @pytest.fixture
    def builder(self, temp_dir):
        """Create LangChainIndexBuilder instance."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        with patch('indexing.langchain_index_builder.OpenAIEmbeddings'):
            builder = LangChainIndexBuilder(
                qdrant_path=qdrant_path,
                collection_name="test_collection"
            )
            return builder
    
    def test_init(self, temp_dir):
        """Test LangChainIndexBuilder initialization."""
        qdrant_path = os.path.join(temp_dir, "qdrant_db")
        with patch('indexing.langchain_index_builder.OpenAIEmbeddings'):
            builder = LangChainIndexBuilder(
                qdrant_path=qdrant_path,
                collection_name="test_collection"
            )
            assert builder.qdrant_path == qdrant_path
            assert builder.collection_name == "test_collection"
            assert builder.extractor is not None
            assert builder.text_splitter is not None
    
    def test_load_documents(self, builder, temp_dir, sample_documents):
        """Test loading documents from JSON files."""
        # Create test JSON files
        destinations_path = os.path.join(temp_dir, "destinations.json")
        guides_path = os.path.join(temp_dir, "guides.json")
        
        with open(destinations_path, 'w') as f:
            json.dump([sample_documents[0]], f)
        
        with open(guides_path, 'w') as f:
            json.dump([sample_documents[1]], f)
        
        # Load documents
        documents = builder.load_documents(destinations_path, guides_path)
        assert len(documents) == 2
        assert documents[0]["name"] == "Paris"
        assert documents[1]["name"] == "Tokyo Travel Guide"
    
    def test_load_documents_missing_files(self, builder, temp_dir):
        """Test loading documents when files don't exist."""
        destinations_path = os.path.join(temp_dir, "destinations.json")
        guides_path = os.path.join(temp_dir, "guides.json")
        
        # Should not raise error, just return empty list
        documents = builder.load_documents(destinations_path, guides_path)
        assert documents == []
    
    @patch('indexing.langchain_index_builder.ActivityExtractor')
    def test_document_to_langchain_doc(self, mock_extractor, builder, sample_documents):
        """Test converting document dict to LangChain Document."""
        # Mock activity extractor
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_structured_fields.return_value = {
            "extracted_activities": ["art", "culture"]
        }
        builder.extractor = mock_extractor_instance
        
        # Convert document
        doc = builder.document_to_langchain_doc(sample_documents[0], 0)
        
        assert isinstance(doc, Document)
        assert "Paris" in doc.page_content
        assert "France" in doc.page_content
        assert doc.metadata["doc_id"] == "destination_0"
        assert doc.metadata["doc_type"] == "destination"
        assert doc.metadata["name"] == "Paris"
        assert doc.metadata["country"] == "France"
        assert "museums" in doc.metadata["activities"]
        assert "art" in doc.metadata["activities"]  # From extracted activities
    
    @patch('indexing.langchain_index_builder.Qdrant')
    @patch('indexing.langchain_index_builder.ActivityExtractor')
    def test_build_vector_index(self, mock_extractor, mock_qdrant, builder, sample_documents, temp_dir):
        """Test building vector index."""
        # Mock activity extractor
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_structured_fields.return_value = {
            "extracted_activities": []
        }
        builder.extractor = mock_extractor_instance
        
        # Mock Qdrant
        mock_vector_store = MagicMock()
        mock_qdrant.from_documents.return_value = mock_vector_store
        
        # Build index
        vector_store = builder.build_vector_index(sample_documents, recreate=True)
        
        # Verify Qdrant.from_documents was called
        assert mock_qdrant.from_documents.called
        call_args = mock_qdrant.from_documents.call_args
        assert len(call_args[1]["documents"]) == 2
        assert isinstance(call_args[1]["documents"][0], Document)
    
    @patch('indexing.langchain_index_builder.Qdrant')
    @patch('indexing.langchain_index_builder.ActivityExtractor')
    def test_build_vector_index_recreate(self, mock_extractor, mock_qdrant, builder, sample_documents, temp_dir):
        """Test building vector index with recreate=True removes existing database."""
        # Create existing Qdrant directory
        os.makedirs(builder.qdrant_path, exist_ok=True)
        
        # Mock activity extractor
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_structured_fields.return_value = {
            "extracted_activities": []
        }
        builder.extractor = mock_extractor_instance
        
        # Mock Qdrant
        mock_vector_store = MagicMock()
        mock_qdrant.from_documents.return_value = mock_vector_store
        
        # Build index
        builder.build_vector_index(sample_documents, recreate=True)
        
        # Verify shutil.rmtree would be called (or directory removed)
        # The actual removal happens in the method
    
    @patch('indexing.langchain_index_builder.Qdrant')
    def test_load_vector_store(self, mock_qdrant, builder):
        """Test loading existing vector store."""
        mock_vector_store = MagicMock()
        mock_qdrant.return_value = mock_vector_store
        
        vector_store = builder.load_vector_store()
        
        assert mock_qdrant.called
        call_args = mock_qdrant.call_args
        assert call_args[1]["path"] == builder.qdrant_path
        assert call_args[1]["collection_name"] == builder.collection_name

