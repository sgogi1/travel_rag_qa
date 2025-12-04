"""Integration tests for indexing pipeline."""

import os
import json
import pytest
import tempfile
import shutil
from indexing.index_builder import IndexBuilder
from indexing.llm_extractor import ActivityExtractor
from unittest.mock import patch, MagicMock


class TestIndexing:
    """Integration tests for indexing."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """Create sample data files."""
        destinations = [
            {
                "name": "Paris",
                "country": "France",
                "description": "City with museums and art galleries.",
                "activities": ["museums", "art galleries"]
            }
        ]
        guides = [
            {
                "name": "Test Guide",
                "country": "France",
                "description": "Expert guide in Paris.",
                "activities": ["city tours"]
            }
        ]
        
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        with open(os.path.join(data_dir, "destinations.json"), 'w') as f:
            json.dump(destinations, f)
        with open(os.path.join(data_dir, "guides.json"), 'w') as f:
            json.dump(guides, f)
        
        return os.path.join(data_dir, "destinations.json"), os.path.join(data_dir, "guides.json")
    
    @patch('indexing.llm_extractor.OpenAI')
    def test_build_baseline_index(self, mock_openai, temp_dir, sample_data):
        """Test baseline index building."""
        builder = IndexBuilder(index_dir=os.path.join(temp_dir, "indexes"))
        destinations_path, guides_path = sample_data
        
        documents = builder.load_documents(destinations_path, guides_path)
        index_path = builder.build_baseline_index(documents)
        
        assert os.path.exists(index_path)
        assert os.path.isdir(index_path)
    
    @patch('indexing.llm_extractor.OpenAI')
    @patch('retrieval.embedding_generator.OpenAI')
    def test_build_improved_index(self, mock_embedding_openai, mock_extractor_openai, temp_dir, sample_data):
        """Test improved index building."""
        # Mock LLM extractor
        mock_extractor_response = MagicMock()
        mock_extractor_response.choices = [MagicMock()]
        mock_extractor_response.choices[0].message.content = '["museums", "art galleries"]'
        
        mock_extractor_client = MagicMock()
        mock_extractor_client.chat.completions.create.return_value = mock_extractor_response
        mock_extractor_openai.return_value = mock_extractor_client
        
        # Mock embedding generator
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [0.1] * 1536
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [mock_embedding_data]
        
        mock_embedding_client = MagicMock()
        mock_embedding_client.embeddings.create.return_value = mock_embedding_response
        mock_embedding_openai.return_value = mock_embedding_client
        
        builder = IndexBuilder(
            index_dir=os.path.join(temp_dir, "indexes"),
            build_vector_index=False  # Skip vector for faster tests
        )
        destinations_path, guides_path = sample_data
        
        documents = builder.load_documents(destinations_path, guides_path)
        index_path = builder.build_improved_index(documents)
        
        assert os.path.exists(index_path)
        assert os.path.isdir(index_path)
    
    def test_load_documents(self, sample_data):
        """Test document loading."""
        builder = IndexBuilder()
        destinations_path, guides_path = sample_data
        
        documents = builder.load_documents(destinations_path, guides_path)
        assert len(documents) == 2  # 1 destination + 1 guide
        assert all("type" in doc for doc in documents)
        assert any(doc["type"] == "destination" for doc in documents)
        assert any(doc["type"] == "guide" for doc in documents)

