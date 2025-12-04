"""
LLM-powered query rewriting to extract structured filters from natural language queries.
"""

import os
import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class QueryRewriter:
    """Rewrites natural language queries into structured filter queries."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def rewrite_query(self, user_query: str) -> Dict[str, Any]:
        """
        Rewrite a natural language query into structured filters.
        
        Args:
            user_query: Natural language query from user
        
        Returns:
            Dict with structured filters: {city, country, activities, original_query}
        """
        prompt = f"""Convert the following user query about travel into a structured filter query.

Extract:
1. City/destination name (if mentioned)
2. Country (if mentioned)
3. Activities/services requested (e.g., snorkeling, wine tasting, city tours)
4. Activity categories (e.g., "outdoor activities", "wellness", "adventure", "cultural", "culinary")

IMPORTANT: If the query mentions a category like "outdoor activities", "adventure", "wellness", etc., 
expand it to include specific activities:
- "outdoor activities" or "adventure" → include: hiking, snorkeling, diving, kayaking, etc.
- "wellness" → include: yoga, spa treatments, meditation, etc.
- "cultural" → include: museums, temple visits, historical tours, etc.
- "culinary" or "food" → include: culinary tours, cooking classes, wine tasting, etc.

User query: "{user_query}"

Return ONLY a JSON object with this exact structure:
{{
  "city": "city_name_or_null",
  "country": "country_name_or_null",
  "activities": ["activity1", "activity2", "activity3"],
  "original_query": "{user_query}"
}}

If a field is not mentioned, use null. Activities should be normalized (lowercase, singular forms preferred).
If a category is mentioned, expand it to specific activities in the activities array.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from travel queries. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up the response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            
            # Normalize activities
            if "activities" in result and isinstance(result["activities"], list):
                result["activities"] = [str(a).lower().strip() for a in result["activities"] if a]
            else:
                result["activities"] = []
            
            # Normalize city and country
            if result.get("city") and result["city"].lower() == "null":
                result["city"] = None
            if result.get("country") and result["country"].lower() == "null":
                result["country"] = None
            
            result["original_query"] = user_query
            
            return result
            
        except Exception as e:
            print(f"Error rewriting query: {e}")
            # Fallback to original query
            return {
                "city": None,
                "country": None,
                "activities": [],
                "original_query": user_query
            }


if __name__ == "__main__":
    # Test the rewriter
    rewriter = QueryRewriter()
    
    test_queries = [
        "Snorkeling near Lisbon",
        "Wine tasting in Tuscany",
        "City tours in Paris",
        "I want to go hiking in Iceland"
    ]
    
    for query in test_queries:
        result = rewriter.rewrite_query(query)
        print(f"\nQuery: {query}")
        print(f"Rewritten: {json.dumps(result, indent=2)}")

