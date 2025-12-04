"""
Build search indexes for travel documents.
Supports both baseline (naive) and improved (structured) indexing.
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, KEYWORD, STORED
from whoosh.analysis import StandardAnalyzer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_extractor import ActivityExtractor
from retrieval.embedding_generator import EmbeddingGenerator
from retrieval.qdrant_store import QdrantStore


class IndexBuilder:
    """Builds and manages search indexes for travel documents."""
    
    def __init__(self, index_dir: str = "indexes", build_vector_index: bool = True):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.extractor = ActivityExtractor()
        self.build_vector_index = build_vector_index
        if build_vector_index:
            self.embedding_generator = EmbeddingGenerator()
            self.qdrant_store = QdrantStore()
    
    def create_baseline_schema(self) -> Schema:
        """Create schema for baseline (naive) search."""
        return Schema(
            doc_id=ID(stored=True, unique=True),
            doc_type=STORED,  # "destination" or "guide"
            name=STORED,
            country=STORED,
            region=STORED,
            content=TEXT(analyzer=StandardAnalyzer()),  # Combined text field
            raw_data=STORED  # Store full document
        )
    
    def create_improved_schema(self) -> Schema:
        """Create schema for improved search with structured fields."""
        return Schema(
            doc_id=ID(stored=True, unique=True),
            doc_type=STORED,
            name=STORED,
            country=STORED,
            region=STORED,
            content=TEXT(analyzer=StandardAnalyzer()),  # Still include for BM25
            activities=KEYWORD(stored=True, lowercase=True, commas=True),  # Structured activities
            extracted_activities=STORED,  # Store as list
            raw_data=STORED
        )
    
    def load_documents(self, destinations_path: str, guides_path: str) -> List[Dict[str, Any]]:
        """Load documents from JSON files."""
        documents = []
        
        # Load destinations
        with open(destinations_path, 'r', encoding='utf-8') as f:
            destinations = json.load(f)
            for dest in destinations:
                doc = dest.copy()
                doc["type"] = "destination"
                doc["region"] = f"{dest['name']}, {dest['country']}"
                documents.append(doc)
        
        # Load guides
        with open(guides_path, 'r', encoding='utf-8') as f:
            guides = json.load(f)
            for guide in guides:
                doc = guide.copy()
                doc["type"] = "guide"
                documents.append(doc)
        
        return documents
    
    def build_baseline_index(self, documents: List[Dict[str, Any]], index_name: str = "baseline"):
        """Build baseline index (naive approach - just text search)."""
        schema = self.create_baseline_schema()
        index_path = os.path.join(self.index_dir, index_name)
        
        # Create index
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)
        
        ix = index.create_in(index_path, schema)
        writer = ix.writer()
        
        for i, doc in enumerate(documents):
            doc_id = f"{doc['type']}_{i}"
            
            # Combine all text fields for naive search
            content_parts = [
                doc.get("name", ""),
                doc.get("country", ""),
                doc.get("region", ""),
                doc.get("description", "")
            ]
            content = " ".join(filter(None, content_parts))
            
            writer.add_document(
                doc_id=doc_id,
                doc_type=doc["type"],
                name=doc.get("name", ""),
                country=doc.get("country", ""),
                region=doc.get("region", ""),
                content=content,
                raw_data=json.dumps(doc)
            )
        
        writer.commit()
        print(f"Built baseline index with {len(documents)} documents")
        return index_path
    
    def build_improved_index(self, documents: List[Dict[str, Any]], index_name: str = "improved"):
        """Build improved index with structured activity extraction."""
        schema = self.create_improved_schema()
        index_path = os.path.join(self.index_dir, index_name)
        
        # Create index
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)
        
        ix = index.create_in(index_path, schema)
        writer = ix.writer()
        
        for i, doc in enumerate(documents):
            doc_id = f"{doc['type']}_{i}"
            
            # Extract activities using LLM
            print(f"Extracting activities for {doc.get('name', doc_id)}...")
            doc_with_activities = self.extractor.extract_structured_fields(doc)
            extracted_activities = doc_with_activities.get("extracted_activities", [])
            
            # Also use original activities if available (for comparison)
            original_activities = doc.get("activities", [])
            all_activities = list(set(extracted_activities + original_activities))
            
            # Combine text fields for BM25
            content_parts = [
                doc.get("name", ""),
                doc.get("country", ""),
                doc.get("region", ""),
                doc.get("description", "")
            ]
            content = " ".join(filter(None, content_parts))
            
            # Store activities as comma-separated string for KEYWORD field
            activities_str = ",".join(all_activities) if all_activities else ""
            
            writer.add_document(
                doc_id=doc_id,
                doc_type=doc["type"],
                name=doc.get("name", ""),
                country=doc.get("country", ""),
                region=doc.get("region", ""),
                content=content,
                activities=activities_str,
                extracted_activities=json.dumps(all_activities),
                raw_data=json.dumps(doc)
            )
        
        writer.commit()
        print(f"Built improved index with {len(documents)} documents")
        
        # Build vector index if enabled
        if self.build_vector_index:
            print("\nBuilding vector index with Qdrant...")
            self.build_vector_index_for_documents(documents)
        
        return index_path
    
    def build_vector_index_for_documents(self, documents: List[Dict[str, Any]]):
        """
        Build vector index in Qdrant for documents.
        
        Args:
            documents: List of documents to index
        """
        print(f"Generating embeddings for {len(documents)} documents...")
        
        # Prepare documents for vector indexing
        indexed_docs = []
        texts_to_embed = []
        
        for i, doc in enumerate(documents):
            # Create text representation for embedding
            text_parts = [
                doc.get("name", ""),
                doc.get("country", ""),
                doc.get("region", ""),
                doc.get("description", "")
            ]
            if doc.get("activities"):
                text_parts.append(", ".join(doc["activities"]))
            
            text = " ".join(filter(None, text_parts))
            texts_to_embed.append(text)
            
            # Prepare document metadata
            indexed_doc = {
                "doc_id": f"{doc.get('type', 'unknown')}_{i}",
                "doc_type": doc.get("type", "unknown"),
                "name": doc.get("name", ""),
                "country": doc.get("country", ""),
                "region": doc.get("region", ""),
                "activities": doc.get("activities", []),
                "description": doc.get("description", ""),
                "raw_data": doc
            }
            indexed_docs.append(indexed_doc)
        
        # Generate embeddings in batches
        print("Generating embeddings (this may take a few minutes)...")
        embeddings = self.embedding_generator.generate_embeddings_batch(texts_to_embed, batch_size=100)
        
        print(f"Generated {len(embeddings)} embeddings")
        print("Storing in Qdrant...")
        
        # Store in Qdrant
        self.qdrant_store.add_documents(indexed_docs, embeddings)
        
        print(f"✅ Vector index built with {len(indexed_docs)} documents")


if __name__ == "__main__":
    # Build indexes
    builder = IndexBuilder()
    
    # Get data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    destinations_path = os.path.join(data_dir, "destinations.json")
    guides_path = os.path.join(data_dir, "guides.json")
    
    # Generate data if needed
    if not os.path.exists(destinations_path):
        import sys
        sys.path.append(data_dir)
        from generate_sample_data import generate_data
        generate_data()
    
    # Load documents
    documents = builder.load_documents(destinations_path, guides_path)
    
    # Build indexes
    print("Building baseline index...")
    baseline_path = builder.build_baseline_index(documents)
    
    print("\nBuilding improved index (this may take a few minutes due to LLM calls)...")
    improved_path = builder.build_improved_index(documents)
    
    print(f"\nIndexes built successfully!")
    print(f"Baseline: {baseline_path}")
    print(f"Improved: {improved_path}")

