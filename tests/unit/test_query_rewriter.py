"""Unit tests for QueryRewriter."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from retrieval.query_rewriter import QueryRewriter


class TestQueryRewriter:
    """Test suite for QueryRewriter."""
    
    @pytest.fixture
    def rewriter(self):
        """Create QueryRewriter instance."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            rew = QueryRewriter()
            # Mock the client after initialization
            rew.client = MagicMock()
            return rew
    
    def test_init(self, rewriter):
        """Test QueryRewriter initialization."""
        assert rewriter is not None
        assert hasattr(rewriter, 'client')
        assert hasattr(rewriter, 'model')
    
    def test_rewrite_query_simple(self, rewriter):
        """Test simple query rewriting."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"city": null, "country": null, "activities": ["snorkeling"], "original_query": "snorkeling"}'
        
        rewriter.client.chat.completions.create.return_value = mock_response
        
        result = rewriter.rewrite_query("snorkeling")
        assert "activities" in result
        assert result["original_query"] == "snorkeling"
    
    def test_rewrite_query_with_location(self, rewriter):
        """Test query rewriting with location."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"city": "Bali", "country": "Indonesia", "activities": ["snorkeling"], "original_query": "snorkeling in Bali"}'
        
        rewriter.client.chat.completions.create.return_value = mock_response
        
        result = rewriter.rewrite_query("snorkeling in Bali")
        assert result.get("city") == "Bali" or result.get("country") == "Indonesia"
    
    def test_rewrite_query_with_category(self, rewriter):
        """Test query rewriting with category."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"city": null, "country": null, "activities": ["hiking", "snorkeling", "diving"], "original_query": "outdoor activities"}'
        
        rewriter.client.chat.completions.create.return_value = mock_response
        
        result = rewriter.rewrite_query("outdoor activities")
        assert "activities" in result
        assert len(result.get("activities", [])) > 0
    
    def test_rewrite_query_error_handling(self, rewriter):
        """Test error handling in query rewriting."""
        rewriter.client.chat.completions.create.side_effect = Exception("API Error")
        
        result = rewriter.rewrite_query("test query")
        # Should return default structure on error
        assert "original_query" in result
        assert result.get("activities", []) == []

