"""
Generate embeddings for documents and queries using OpenAI.
"""

import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generates embeddings using OpenAI's embedding model."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding generator.
        
        Args:
            model: OpenAI embedding model to use
                   Options: "text-embedding-3-small", "text-embedding-3-large", 
                           "text-embedding-ada-002"
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Generate individual embeddings for this batch as fallback
                for text in batch:
                    try:
                        embedding = self.generate_embedding(text)
                        embeddings.append(embedding)
                    except:
                        # Use zero vector as fallback
                        embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model.
        
        Returns:
            Dimension of embedding vectors
        """
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return model_dimensions.get(self.model, 1536)


if __name__ == "__main__":
    # Test the embedding generator
    generator = EmbeddingGenerator()
    
    test_text = "Snorkeling in crystal clear waters with tropical fish"
    embedding = generator.generate_embedding(test_text)
    
    print(f"Embedding generated: {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}")

