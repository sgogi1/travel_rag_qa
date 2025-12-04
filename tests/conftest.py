"""Pytest configuration and shared fixtures."""

import os
import sys
import json
import tempfile
import shutil
from typing import List, Dict, Any
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for test indexes."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_qdrant_dir():
    """Create a temporary directory for Qdrant database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_destinations() -> List[Dict[str, Any]]:
    """Sample destination data for testing."""
    return [
        {
            "name": "Paris",
            "country": "France",
            "description": "Beautiful city with museums, art galleries, and culinary tours.",
            "activities": ["museums", "art galleries", "culinary tours", "city tours"]
        },
        {
            "name": "Bali",
            "country": "Indonesia",
            "description": "Tropical paradise with beaches, snorkeling, and yoga retreats.",
            "activities": ["beaches", "snorkeling", "yoga", "spa treatments"]
        },
        {
            "name": "Iceland",
            "country": "Iceland",
            "description": "Adventure destination with hiking, glacier tours, and Northern Lights.",
            "activities": ["hiking", "glacier tours", "Northern Lights viewing", "photography tours"]
        }
    ]


@pytest.fixture
def sample_guides() -> List[Dict[str, Any]]:
    """Sample guide data for testing."""
    return [
        {
            "name": "Jean-Pierre Dubois",
            "country": "France",
            "region": "Paris, France",
            "description": "Expert guide specializing in city tours and historical sites in Paris.",
            "activities": ["city tours", "historical tours", "museums"]
        },
        {
            "name": "Sarah Johnson",
            "country": "Iceland",
            "region": "Iceland",
            "description": "Adventure guide for hiking and outdoor activities in Iceland.",
            "activities": ["hiking", "outdoor activities", "adventure tours"]
        }
    ]


@pytest.fixture
def sample_documents(sample_destinations, sample_guides) -> List[Dict[str, Any]]:
    """Combined sample documents for testing."""
    documents = []
    for dest in sample_destinations:
        doc = dest.copy()
        doc["type"] = "destination"
        doc["region"] = f"{dest['name']}, {dest['country']}"
        documents.append(doc)
    
    for guide in sample_guides:
        doc = guide.copy()
        doc["type"] = "guide"
        documents.append(doc)
    
    return documents


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing."""
    class MockResponse:
        def __init__(self, content: str):
            self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})()})()]
    
    return MockResponse


@pytest.fixture
def mock_embedding_response():
    """Mock OpenAI embedding response."""
    class MockEmbeddingData:
        def __init__(self):
            self.embedding = [0.1] * 1536
    
    class MockEmbeddingResponse:
        def __init__(self):
            self.data = [MockEmbeddingData()]
    
    return MockEmbeddingResponse

