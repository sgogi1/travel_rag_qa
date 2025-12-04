"""
Improved retrieval system using BM25 + structured filtering + query rewriting.
Demonstrates high-recall retrieval with structured data.
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional
from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.query import And, Or, Term, Every

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query_rewriter import QueryRewriter
from activity_matcher import ActivityMatcher


class ImprovedRetriever:
    """Improved retriever with structured filtering and query rewriting."""
    
    def __init__(self, index_path: str):
        """
        Initialize retriever with index path.
        
        Args:
            index_path: Path to the Whoosh index directory
        """
        if not os.path.exists(index_path):
            raise ValueError(f"Index not found at {index_path}")
        
        self.index_path = index_path
        self.ix = index.open_dir(index_path)
        self.searcher = self.ix.searcher()
        self.query_parser = QueryParser("content", schema=self.ix.schema)
        self.rewriter = QueryRewriter()
        self.activity_matcher = ActivityMatcher()
    
    def search_with_filters(
        self,
        query: str,
        city: Optional[str] = None,
        country: Optional[str] = None,
        activities: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform search with structured filters.
        
        Args:
            query: Text query for BM25
            city: Filter by city name
            country: Filter by country
            activities: Filter by activities list
            limit: Maximum number of results
        
        Returns:
            List of retrieved documents with scores
        """
        # Build BM25 query
        parsed_query = self.query_parser.parse(query)
        queries = [parsed_query]
        
        # Add structured filters
        if city:
            # Search in content field for city name (since name/region are STORED, not indexed)
            # Use text search in content field for city matching
            city_parser = QueryParser("content", schema=self.ix.schema)
            city_query = city_parser.parse(city.lower())
            queries.append(city_query)
        
        if country:
            # Search in content field for country (since country is STORED, not indexed)
            country_parser = QueryParser("content", schema=self.ix.schema)
            country_query = country_parser.parse(country.lower())
            queries.append(country_query)
        
        if activities and len(activities) > 0:
            # Expand activities with synonyms and fuzzy matching
            expanded_activities = set()
            for activity in activities:
                expanded = self.activity_matcher.expand_activity(activity)
                expanded_activities.update(expanded)
                # Also add the original
                expanded_activities.add(activity.lower().strip())
            
            # Create queries for all expanded activities
            activity_queries = []
            for expanded_activity in expanded_activities:
                # Normalize the activity
                normalized = expanded_activity.lower().strip()
                # Try exact match
                activity_queries.append(Term("activities", normalized))
                
                # Add plural/singular variations
                if normalized.endswith('s') and len(normalized) > 1:
                    # Remove 's' for singular
                    activity_queries.append(Term("activities", normalized[:-1]))
                elif not normalized.endswith('s'):
                    # Add 's' for plural
                    activity_queries.append(Term("activities", normalized + 's'))
                
                # Also try with 'es' ending
                if normalized.endswith('es'):
                    activity_queries.append(Term("activities", normalized[:-2]))
                elif normalized.endswith('s') and not normalized.endswith('es'):
                    activity_queries.append(Term("activities", normalized + 'es'))
            
            # Remove duplicates by converting to set of strings, then back to Terms
            unique_activities = set()
            for q in activity_queries:
                if hasattr(q, 'text'):
                    unique_activities.add(q.text)
            
            # Recreate queries from unique set
            activity_queries = [Term("activities", act) for act in unique_activities]
            
            # Use OR to match any of the expanded activities
            if activity_queries:
                activity_query = Or(activity_queries) if len(activity_queries) > 1 else activity_queries[0]
                queries.append(activity_query)
        
        # If we have activity filters, also search in content field as backup
        # This ensures we find results even if structured matching is too strict
        if activities and len(activities) > 0:
            # Build content search query for activities
            activity_terms = []
            for activity in activities:
                expanded = self.activity_matcher.expand_activity(activity)
                activity_terms.extend([a for a in expanded])
                # Add original
                activity_terms.append(activity.lower().strip())
            
            # Create content query for activities
            activity_text = ' OR '.join(set(activity_terms[:10]))  # Limit to avoid too many terms
            activity_content_parser = QueryParser("content", schema=self.ix.schema)
            activity_content_query = activity_content_parser.parse(activity_text)
            
            # Combine structured activity query with content search using OR
            if activity_queries:
                # Use OR to match either structured field OR content
                structured_activity_query = Or(activity_queries) if len(activity_queries) > 1 else activity_queries[0]
                combined_activity_query = Or([structured_activity_query, activity_content_query])
                # Replace activity query in queries list
                queries = [q for q in queries if not isinstance(q, (Or, Term)) or (hasattr(q, 'fieldname') and q.fieldname != 'activities')]
                queries.append(combined_activity_query)
        
        # Combine all queries with AND
        final_query = And(queries) if len(queries) > 1 else queries[0]
        
        # Search
        results = self.searcher.search(final_query, limit=limit)
        
        # If we have activity filters but got no results, try a more lenient search
        # (fallback to content search with activity terms)
        if activities and len(activities) > 0 and len(results) == 0:
            # Build a more lenient query - search in content field for activity terms
            activity_terms = []
            for activity in activities:
                expanded = self.activity_matcher.expand_activity(activity)
                for exp_act in expanded:
                    activity_terms.append(exp_act)
                    # Add plural/singular variants
                    if exp_act.endswith('s'):
                        activity_terms.append(exp_act[:-1])
                    else:
                        activity_terms.append(exp_act + 's')
            
            # Search in content field for these terms
            activity_text = ' OR '.join(set(activity_terms))
            fallback_parser = QueryParser("content", schema=self.ix.schema)
            fallback_query = fallback_parser.parse(activity_text)
            
            # Combine with other filters (city, country) if present
            fallback_queries = [fallback_query]
            if city:
                city_parser = QueryParser("content", schema=self.ix.schema)
                city_query = city_parser.parse(city.lower())
                fallback_queries.append(city_query)
            if country:
                country_parser = QueryParser("content", schema=self.ix.schema)
                country_query = country_parser.parse(country.lower())
                fallback_queries.append(country_query)
            
            if len(fallback_queries) > 1:
                final_query = And(fallback_queries)
            else:
                final_query = fallback_queries[0]
            
            results = self.searcher.search(final_query, limit=limit)
        
        # Format results
        # Note: We don't filter here because the query already handles matching
        # The fuzzy matching is applied in the query construction above
        formatted_results = []
        for result in results:
            doc = json.loads(result["raw_data"])
            doc_activities = json.loads(result.get("extracted_activities", "[]"))
            
            formatted_results.append({
                "doc_id": result["doc_id"],
                "doc_type": result["doc_type"],
                "name": result["name"],
                "country": result.get("country", ""),
                "region": result.get("region", ""),
                "activities": doc_activities,
                "score": result.score,
                "document": doc
            })
        
        return formatted_results
    
    def search(self, user_query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Perform improved search with automatic query rewriting.
        
        Args:
            user_query: Natural language query from user
            limit: Maximum number of results
        
        Returns:
            Dict with rewritten query and results
        """
        # Rewrite query to extract structured filters
        rewritten = self.rewriter.rewrite_query(user_query)
        
        # Extract filters
        city = rewritten.get("city")
        country = rewritten.get("country")
        activities = rewritten.get("activities", [])
        
        # Use original query for BM25 text search
        text_query = user_query
        
        # Perform search with filters
        results = self.search_with_filters(
            query=text_query,
            city=city,
            country=country,
            activities=activities,
            limit=limit
        )
        
        return {
            "original_query": user_query,
            "rewritten_query": rewritten,
            "results": results,
            "num_results": len(results)
        }
    
    def close(self):
        """Close the searcher."""
        if self.searcher:
            self.searcher.close()


if __name__ == "__main__":
    # Test improved retriever
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes", "improved")
    
    if os.path.exists(index_path):
        retriever = ImprovedRetriever(index_path)
        
        test_queries = [
            "Snorkeling near Lisbon",
            "Wine tasting in Tuscany",
            "City tours in Paris",
            "Hiking in Iceland"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            result = retriever.search(query, limit=5)
            
            print(f"\nRewritten Query:")
            print(f"  City: {result['rewritten_query'].get('city')}")
            print(f"  Country: {result['rewritten_query'].get('country')}")
            print(f"  Activities: {result['rewritten_query'].get('activities')}")
            
            print(f"\nResults ({result['num_results']}):")
            for i, doc in enumerate(result['results'], 1):
                print(f"\n{i}. {doc['name']} ({doc['doc_type']}) - Score: {doc['score']:.3f}")
                print(f"   Region: {doc.get('region', 'N/A')}")
                print(f"   Activities: {', '.join(doc.get('activities', [])[:5])}")
        
        retriever.close()
    else:
        print(f"Index not found at {index_path}. Please build the index first.")

