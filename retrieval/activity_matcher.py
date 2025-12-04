"""
Activity matching with fuzzy matching and synonym expansion.
Handles plural/singular variations and similar activity names.
"""

from typing import List, Set
import re


class ActivityMatcher:
    """Matches activities with fuzzy matching and synonym support."""
    
    # Activity synonyms and variations
    SYNONYMS = {
        "tour": ["tours", "tour", "guided tour", "guided tours"],
        "tours": ["tour", "tours", "guided tour", "guided tours"],
        "city tour": ["city tours", "city tour", "urban tour", "urban tours"],
        "city tours": ["city tour", "city tours", "urban tour", "urban tours"],
        "photography tour": ["photography tours", "photo tour", "photo tours", "photography tour"],
        "photography tours": ["photography tour", "photography tours", "photo tour", "photo tours"],
        "historical tour": ["historical tours", "history tour", "history tours", "historical tour"],
        "historical tours": ["historical tour", "historical tours", "history tour", "history tours"],
        "culinary tour": ["culinary tours", "food tour", "food tours", "culinary tour"],
        "culinary tours": ["culinary tour", "culinary tours", "food tour", "food tours"],
        "wine tasting": ["wine tastings", "wine tasting", "tasting", "tastings"],
        "snorkeling": ["snorkeling", "snorkel", "snorkelling", "snorkel diving"],
        "diving": ["diving", "scuba diving", "dive"],
        "hiking": ["hiking", "hike", "trekking", "trek"],
        "beach": ["beaches", "beach", "beach activities"],
        "beaches": ["beach", "beaches", "beach activities"],
        "museum": ["museums", "museum", "gallery", "galleries"],
        "museums": ["museum", "museums", "gallery", "galleries"],
    }
    
    # Activity categories - maps category terms to specific activities
    CATEGORIES = {
        "outdoor": ["hiking", "trekking", "camping", "rock climbing", "cycling", "kayaking", 
                   "rafting", "paragliding", "skydiving", "bungee jumping", "zip-lining",
                   "snorkeling", "diving", "surfing", "beach", "beaches", "fishing",
                   "bird watching", "wildlife viewing", "glacier tours", "volcano tours",
                   "cave exploration", "adventure tours"],
        "outdoor activities": ["hiking", "trekking", "camping", "rock climbing", "cycling",
                             "kayaking", "rafting", "paragliding", "snorkeling", "diving",
                             "surfing", "beach", "beaches", "fishing", "adventure tours"],
        "adventure": ["hiking", "trekking", "rock climbing", "kayaking", "rafting",
                     "paragliding", "skydiving", "bungee jumping", "zip-lining",
                     "snorkeling", "diving", "surfing", "glacier tours", "volcano tours",
                     "cave exploration", "adventure tours"],
        "adventure activities": ["hiking", "trekking", "rock climbing", "kayaking", "rafting",
                                "paragliding", "skydiving", "bungee jumping", "snorkeling",
                                "diving", "surfing", "adventure tours"],
        "wellness": ["yoga", "spa treatments", "meditation", "wellness retreats", "hot springs",
                    "massage", "relaxation", "mindfulness"],
        "wellness retreats": ["yoga", "spa treatments", "meditation", "wellness retreats",
                             "hot springs", "massage", "relaxation"],
        "cultural": ["museums", "art galleries", "temple visits", "historical tours",
                    "cultural experiences", "traditional ceremonies", "tea ceremonies",
                    "cultural heritage", "local traditions"],
        "cultural activities": ["museums", "art galleries", "temple visits", "historical tours",
                              "cultural experiences", "traditional ceremonies", "tea ceremonies"],
        "entertainment": ["nightlife", "bars", "clubs", "concerts", "festivals", "casinos",
                         "theater", "opera", "sports events", "jazz clubs", "music venues"],
        "nightlife": ["bars", "clubs", "nightlife", "jazz clubs", "music venues", "casinos"],
        "culinary": ["culinary tours", "food tours", "cooking classes", "wine tasting",
                    "restaurants", "fine dining", "street food tours", "seafood dining",
                    "sushi dining", "tapas tours"],
        "food": ["culinary tours", "food tours", "cooking classes", "restaurants",
                "fine dining", "street food tours", "seafood dining", "sushi dining"],
        "water sports": ["snorkeling", "diving", "surfing", "kayaking", "rafting",
                        "swimming", "beach", "beaches", "water sports"],
        "water activities": ["snorkeling", "diving", "surfing", "kayaking", "swimming",
                             "beach", "beaches", "water sports"],
        "sports": ["cycling", "hiking", "surfing", "tennis", "golf", "beach volleyball",
                  "skiing", "snowboarding", "ice skating", "sports events"],
        "photography": ["photography tours", "photo tours", "photography", "stargazing"],
        "nature": ["hiking", "bird watching", "wildlife viewing", "stargazing",
                  "glacier tours", "volcano tours", "cave exploration", "hot springs",
                  "Northern Lights viewing"],
        "indoor": ["museums", "art galleries", "spa treatments", "cooking classes",
                  "wine tasting", "casinos", "theater", "opera", "shopping"],
    }
    
    def __init__(self):
        """Initialize the activity matcher."""
        # Build reverse lookup for faster matching
        self._synonym_map = {}
        for key, synonyms in self.SYNONYMS.items():
            for synonym in synonyms:
                normalized = self._normalize(synonym)
                if normalized not in self._synonym_map:
                    self._synonym_map[normalized] = set()
                self._synonym_map[normalized].add(self._normalize(key))
                # Also add all synonyms to the set
                for s in synonyms:
                    self._synonym_map[normalized].add(self._normalize(s))
    
    def _normalize(self, activity: str) -> str:
        """
        Normalize activity string for matching.
        
        Args:
            activity: Activity string to normalize
        
        Returns:
            Normalized activity string
        """
        # Convert to lowercase and strip
        normalized = activity.lower().strip()
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _fuzzy_match(self, query_activity: str, data_activity: str, threshold: float = 0.8) -> bool:
        """
        Check if two activities match using fuzzy matching.
        
        Args:
            query_activity: Activity from query
            data_activity: Activity from data
            threshold: Similarity threshold (0-1)
        
        Returns:
            True if activities match
        """
        query_norm = self._normalize(query_activity)
        data_norm = self._normalize(data_activity)
        
        # Exact match
        if query_norm == data_norm:
            return True
        
        # Check if one contains the other (for compound activities)
        if query_norm in data_norm or data_norm in query_norm:
            return True
        
        # Check for plural/singular variations
        if self._is_plural_variant(query_norm, data_norm):
            return True
        
        # Simple similarity check (Levenshtein-like)
        similarity = self._simple_similarity(query_norm, data_norm)
        if similarity >= threshold:
            return True
        
        return False
    
    def _is_plural_variant(self, a1: str, a2: str) -> bool:
        """Check if two strings are plural/singular variants."""
        # Remove common endings and compare
        a1_base = a1.rstrip('s').rstrip('es').rstrip('ing')
        a2_base = a2.rstrip('s').rstrip('es').rstrip('ing')
        
        if a1_base == a2_base and len(a1_base) > 2:
            return True
        
        # Check if one is the other + 's' or 'es'
        if a1 == a2 + 's' or a1 == a2 + 'es':
            return True
        if a2 == a1 + 's' or a2 == a1 + 'es':
            return True
        
        return False
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple similarity between two strings."""
        if not s1 or not s2:
            return 0.0
        
        # Count common characters
        common = sum(1 for c in s1 if c in s2)
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return 1.0
        
        return common / max_len
    
    def expand_activity(self, activity: str) -> Set[str]:
        """
        Expand an activity to include synonyms, variations, and category mappings.
        
        Args:
            activity: Activity string to expand
        
        Returns:
            Set of expanded activity strings
        """
        normalized = self._normalize(activity)
        expanded = {normalized, activity.lower().strip()}
        
        # Check if it's a category term
        if normalized in self.CATEGORIES:
            # Expand category to specific activities
            expanded.update(self.CATEGORIES[normalized])
            # Also add normalized versions
            for cat_activity in self.CATEGORIES[normalized]:
                expanded.add(self._normalize(cat_activity))
        
        # Check for category patterns (e.g., "outdoor activities")
        for category, activities in self.CATEGORIES.items():
            if normalized in category or category in normalized:
                expanded.update(activities)
                for cat_activity in activities:
                    expanded.add(self._normalize(cat_activity))
        
        # Add from synonym map
        if normalized in self._synonym_map:
            expanded.update(self._synonym_map[normalized])
        
        # Add plural/singular variants
        if normalized.endswith('s'):
            expanded.add(normalized[:-1])  # Remove 's'
        else:
            expanded.add(normalized + 's')  # Add 's'
        
        # Add from direct synonyms
        if normalized in self.SYNONYMS:
            for synonym in self.SYNONYMS[normalized]:
                expanded.add(self._normalize(synonym))
        
        return expanded
    
    def get_category_activities(self, category: str) -> List[str]:
        """
        Get all activities in a category.
        
        Args:
            category: Category name (e.g., "outdoor", "wellness")
        
        Returns:
            List of activities in the category
        """
        normalized = self._normalize(category)
        return self.CATEGORIES.get(normalized, [])
    
    def match_activities(self, query_activities: List[str], data_activities: List[str]) -> bool:
        """
        Check if any query activity matches any data activity.
        
        Args:
            query_activities: List of activities from query
            data_activities: List of activities from data
        
        Returns:
            True if there's a match
        """
        if not query_activities or not data_activities:
            return False
        
        # Expand query activities
        expanded_queries = set()
        for q_activity in query_activities:
            expanded_queries.update(self.expand_activity(q_activity))
        
        # Check each data activity against expanded queries
        for data_activity in data_activities:
            data_norm = self._normalize(data_activity)
            
            # Check exact match in expanded set
            if data_norm in expanded_queries:
                return True
            
            # Check fuzzy match
            for expanded_query in expanded_queries:
                if self._fuzzy_match(expanded_query, data_activity):
                    return True
        
        return False
    
    def find_matching_activities(self, query_activities: List[str], data_activities: List[str]) -> List[str]:
        """
        Find which data activities match the query activities.
        
        Args:
            query_activities: List of activities from query
            data_activities: List of activities from data
        
        Returns:
            List of matching data activities
        """
        matches = []
        
        if not query_activities or not data_activities:
            return matches
        
        # Expand query activities
        expanded_queries = set()
        for q_activity in query_activities:
            expanded_queries.update(self.expand_activity(q_activity))
        
        # Check each data activity
        for data_activity in data_activities:
            data_norm = self._normalize(data_activity)
            
            # Check exact match
            if data_norm in expanded_queries:
                matches.append(data_activity)
                continue
            
            # Check fuzzy match
            for expanded_query in expanded_queries:
                if self._fuzzy_match(expanded_query, data_activity):
                    matches.append(data_activity)
                    break
        
        return matches


if __name__ == "__main__":
    # Test the matcher
    matcher = ActivityMatcher()
    
    # Test cases
    test_cases = [
        (["city tour"], ["city tours"]),
        (["photography tour"], ["photography tours"]),
        (["snorkeling"], ["snorkel"]),
        (["wine tasting"], ["wine tastings"]),
        (["hiking"], ["hike"]),
    ]
    
    print("Testing Activity Matcher:")
    print("="*60)
    for query_acts, data_acts in test_cases:
        result = matcher.match_activities(query_acts, data_acts)
        print(f"Query: {query_acts}")
        print(f"Data: {data_acts}")
        print(f"Match: {result}")
        print()

