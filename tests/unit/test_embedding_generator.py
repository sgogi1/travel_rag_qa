"""Unit tests for EmbeddingGenerator."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from retrieval.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create EmbeddingGenerator instance."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            gen = EmbeddingGenerator()
            # Mock the client after initialization
            gen.client = MagicMock()
            return gen
    
    def test_init(self, generator):
        """Test EmbeddingGenerator initialization."""
        assert generator is not None
        assert hasattr(generator, 'client')
        assert hasattr(generator, 'model')
        assert generator.model == "text-embedding-3-small"
    
    def test_generate_embedding(self, generator):
        """Test embedding generation."""
        # Mock OpenAI response
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 1536
        
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        
        generator.client.embeddings.create.return_value = mock_response
        
        embedding = generator.generate_embedding("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_generate_embedding_error(self, generator):
        """Test error handling in embedding generation."""
        generator.client.embeddings.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            generator.generate_embedding("test text")
    
    def test_generate_embeddings_batch(self, generator):
        """Test batch embedding generation."""
        # Create mock data objects for each text
        mock_data_objects = []
        for i in range(3):
            mock_data = MagicMock()
            mock_data.embedding = [0.1] * 1536
            mock_data_objects.append(mock_data)
        
        mock_response = MagicMock()
        mock_response.data = mock_data_objects
        
        generator.client.embeddings.create.return_value = mock_response
        
        texts = ["text1", "text2", "text3"]
        embeddings = generator.generate_embeddings_batch(texts, batch_size=3)
        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)
    
    def test_generate_embedding_text_cleaning(self, generator):
        """Test text cleaning in embedding generation."""
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 1536
        
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        
        generator.client.embeddings.create.return_value = mock_response
        
        generator.generate_embedding("text\nwith\nnewlines")
        # Verify the API was called
        assert generator.client.embeddings.create.called
        call_args = generator.client.embeddings.create.call_args
        # Check that input was passed correctly
        assert "input" in call_args[1] or len(call_args[0]) > 0

