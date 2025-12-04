"""Unit tests for ActivityMatcher."""

import pytest
from retrieval.activity_matcher import ActivityMatcher


class TestActivityMatcher:
    """Test suite for ActivityMatcher."""
    
    def test_init(self):
        """Test ActivityMatcher initialization."""
        matcher = ActivityMatcher()
        assert matcher is not None
        assert hasattr(matcher, 'SYNONYMS')
        assert hasattr(matcher, 'CATEGORIES')
    
    def test_normalize_activity(self):
        """Test activity normalization."""
        matcher = ActivityMatcher()
        assert matcher._normalize("City Tour") == "city tour"
        assert matcher._normalize("  SNORKELING  ") == "snorkeling"
        assert matcher._normalize("wine   tasting") == "wine tasting"
    
    def test_expand_activity_synonyms(self):
        """Test activity expansion with synonyms."""
        matcher = ActivityMatcher()
        expanded = matcher.expand_activity("tour")
        assert "tour" in expanded
        assert "tours" in expanded
        assert len(expanded) > 1
    
    def test_expand_activity_categories(self):
        """Test category expansion."""
        matcher = ActivityMatcher()
        expanded = matcher.expand_activity("outdoor activities")
        assert len(expanded) > 10  # Should expand to many activities
        assert "hiking" in expanded
        assert "snorkeling" in expanded
    
    def test_expand_activity_plural_singular(self):
        """Test plural/singular handling."""
        matcher = ActivityMatcher()
        expanded_singular = matcher.expand_activity("tour")
        expanded_plural = matcher.expand_activity("tours")
        assert "tour" in expanded_singular
        assert "tours" in expanded_singular
        assert "tour" in expanded_plural
        assert "tours" in expanded_plural
    
    def test_match_activities_exact(self):
        """Test exact activity matching."""
        matcher = ActivityMatcher()
        query_activities = ["snorkeling"]
        data_activities = ["snorkeling", "diving"]
        assert matcher.match_activities(query_activities, data_activities) is True
    
    def test_match_activities_synonym(self):
        """Test synonym matching."""
        matcher = ActivityMatcher()
        query_activities = ["tour"]
        data_activities = ["tours"]
        assert matcher.match_activities(query_activities, data_activities) is True
    
    def test_match_activities_no_match(self):
        """Test no match scenario."""
        matcher = ActivityMatcher()
        query_activities = ["snorkeling"]
        data_activities = ["hiking", "camping"]
        assert matcher.match_activities(query_activities, data_activities) is False
    
    def test_find_matching_activities(self):
        """Test finding matching activities."""
        matcher = ActivityMatcher()
        query_activities = ["tour", "snorkeling"]
        data_activities = ["tours", "snorkeling", "hiking", "diving"]
        matches = matcher.find_matching_activities(query_activities, data_activities)
        assert "tours" in matches
        assert "snorkeling" in matches
        assert len(matches) >= 2
    
    def test_get_category_activities(self):
        """Test getting category activities."""
        matcher = ActivityMatcher()
        activities = matcher.get_category_activities("outdoor")
        assert isinstance(activities, list)
        assert len(activities) > 0
        assert "hiking" in activities
    
    def test_get_category_activities_invalid(self):
        """Test invalid category."""
        matcher = ActivityMatcher()
        activities = matcher.get_category_activities("nonexistent_category")
        assert activities == []
    
    def test_fuzzy_match_exact(self):
        """Test fuzzy matching with exact match."""
        matcher = ActivityMatcher()
        assert matcher._fuzzy_match("snorkeling", "snorkeling") is True
    
    def test_fuzzy_match_contains(self):
        """Test fuzzy matching with substring."""
        matcher = ActivityMatcher()
        assert matcher._fuzzy_match("tour", "city tour") is True
    
    def test_fuzzy_match_plural(self):
        """Test fuzzy matching with plural variations."""
        matcher = ActivityMatcher()
        assert matcher._fuzzy_match("tour", "tours") is True
        assert matcher._fuzzy_match("tours", "tour") is True
    
    def test_simple_similarity(self):
        """Test similarity calculation."""
        matcher = ActivityMatcher()
        similarity = matcher._simple_similarity("snorkeling", "snorkel")
        assert 0 <= similarity <= 1
    
    def test_is_plural_variant(self):
        """Test plural variant detection."""
        matcher = ActivityMatcher()
        assert matcher._is_plural_variant("tour", "tours") is True
        assert matcher._is_plural_variant("tours", "tour") is True
        assert matcher._is_plural_variant("hiking", "hike") is True

