"""Unit tests for LLM extractor."""

import os
import pytest
from unittest.mock import patch, MagicMock
from indexing.llm_extractor import ActivityExtractor


class TestActivityExtractor:
    """Test suite for ActivityExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create ActivityExtractor instance."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            ext = ActivityExtractor()
            # Mock the client after initialization
            ext.client = MagicMock()
            return ext
    
    def test_init(self, extractor):
        """Test ActivityExtractor initialization."""
        assert extractor is not None
        assert hasattr(extractor, 'client')
        assert hasattr(extractor, 'model')
    
    def test_extract_activities(self, extractor):
        """Test activity extraction."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["snorkeling", "diving", "beaches"]'
        
        extractor.client.chat.completions.create.return_value = mock_response
        
        activities = extractor.extract_activities(
            "Tropical destination with snorkeling, diving, and beautiful beaches.",
            "destination"
        )
        
        assert isinstance(activities, list)
        assert len(activities) > 0
        assert "snorkeling" in activities or "diving" in activities
    
    def test_extract_structured_fields(self, extractor):
        """Test structured field extraction."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["museums", "art galleries", "city tours"]'
        
        extractor.client.chat.completions.create.return_value = mock_response
        
        doc = {
            "name": "Paris",
            "country": "France",
            "description": "City with museums and art galleries."
        }
        
        result = extractor.extract_structured_fields(doc)
        assert "extracted_activities" in result
        assert isinstance(result["extracted_activities"], list)
    
    def test_extract_activities_error_handling(self, extractor):
        """Test error handling in activity extraction."""
        extractor.client.chat.completions.create.side_effect = Exception("API Error")
        
        activities = extractor.extract_activities("test description", "destination")
        assert activities == []

