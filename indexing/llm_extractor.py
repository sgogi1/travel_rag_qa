"""
LLM-powered structured field extraction during indexing.
Extracts activities/services from document descriptions.
"""

import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class ActivityExtractor:
    """Extracts structured activities from travel documents using LLM."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def extract_activities(self, description: str, doc_type: str = "destination") -> List[str]:
        """
        Extract activities/services from a document description.
        
        Args:
            description: The document description text
            doc_type: Type of document ("destination" or "guide")
        
        Returns:
            List of extracted activity strings
        """
        prompt = f"""Extract a structured list of activities, services, or experiences mentioned in the following {doc_type} description.

Return ONLY a JSON array of activity strings. Each activity should be a concise, normalized term (e.g., "snorkeling", "wine tasting", "city tours").

Description:
{description}

Return format (JSON array only, no other text):
["activity1", "activity2", "activity3"]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from travel descriptions. Always return valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up the response (remove markdown code blocks if present)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            activities = json.loads(content)
            
            # Ensure it's a list of strings
            if isinstance(activities, list):
                return [str(a).lower().strip() for a in activities if a]
            else:
                return []
                
        except Exception as e:
            print(f"Error extracting activities: {e}")
            return []
    
    def extract_structured_fields(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured fields from a document.
        
        Args:
            document: Document dict with at least a "description" field
        
        Returns:
            Document with added "extracted_activities" field
        """
        description = document.get("description", "")
        doc_type = document.get("type", "destination")
        
        activities = self.extract_activities(description, doc_type)
        
        result = document.copy()
        result["extracted_activities"] = activities
        
        return result


if __name__ == "__main__":
    # Test the extractor
    extractor = ActivityExtractor()
    
    test_description = """
    Lisbon offers stunning architecture, delicious seafood, and vibrant neighborhoods. 
    Enjoy Fado music, explore historic Alfama district, visit Belem Tower, and experience nearby beaches. 
    Snorkeling and water sports available along the coast.
    """
    
    activities = extractor.extract_activities(test_description, "destination")
    print(f"Extracted activities: {activities}")

