"""
LangChain-based index builder for travel documents.
Migrates from custom Whoosh/Qdrant to LangChain framework.
"""

import os
import json
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indexing.llm_extractor import ActivityExtractor


class LangChainIndexBuilder:
    """Builds indexes using LangChain framework."""
    
    def __init__(self, qdrant_path: str = "./qdrant_db", collection_name: str = "travel_documents"):
        """
        Initialize LangChain index builder.
        
        Args:
            qdrant_path: Path to Qdrant database
            collection_name: Name of Qdrant collection
        """
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.extractor = ActivityExtractor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_documents(self, destinations_path: str, guides_path: str) -> List[Dict[str, Any]]:
        """Load documents from JSON files."""
        documents = []
        
        if os.path.exists(destinations_path):
            with open(destinations_path, 'r') as f:
                destinations = json.load(f)
                documents.extend(destinations)
        
        if os.path.exists(guides_path):
            with open(guides_path, 'r') as f:
                guides = json.load(f)
                documents.extend(guides)
        
        return documents
    
    def document_to_langchain_doc(self, doc: Dict[str, Any], index: int) -> Document:
        """
        Convert a document dict to LangChain Document.
        
        Args:
            doc: Document dictionary
            index: Document index
        
        Returns:
            LangChain Document with metadata
        """
        # Extract activities using LLM
        doc_with_activities = self.extractor.extract_structured_fields(doc)
        extracted_activities = doc_with_activities.get("extracted_activities", [])
        original_activities = doc.get("activities", [])
        all_activities = list(set(extracted_activities + original_activities))
        
        # Create text content
        content_parts = [
            doc.get("name", ""),
            doc.get("country", ""),
            doc.get("region", ""),
            doc.get("description", "")
        ]
        if all_activities:
            content_parts.append(f"Activities: {', '.join(all_activities)}")
        
        content = " ".join(filter(None, content_parts))
        
        # Create metadata
        metadata = {
            "doc_id": f"{doc.get('type', 'unknown')}_{index}",
            "doc_type": doc.get("type", "unknown"),
            "name": doc.get("name", ""),
            "country": doc.get("country", ""),
            "region": doc.get("region", ""),
            "activities": all_activities,  # Store as list for filtering
            "activities_str": ",".join(all_activities),  # Store as string for search
            "raw_data": json.dumps(doc)  # Store original data
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def build_vector_index(self, documents: List[Dict[str, Any]], recreate: bool = True):
        """
        Build vector index using LangChain QdrantVectorStore.
        
        Args:
            documents: List of document dictionaries
            recreate: Whether to recreate the collection
        """
        print(f"Building LangChain vector index for {len(documents)} documents...")
        
        # Convert documents to LangChain Documents
        langchain_docs = []
        for i, doc in enumerate(documents):
            print(f"Processing {i+1}/{len(documents)}: {doc.get('name', 'Unknown')}")
            langchain_doc = self.document_to_langchain_doc(doc, i)
            langchain_docs.append(langchain_doc)
        
        # Create or load Qdrant vector store
        if recreate and os.path.exists(self.qdrant_path):
            # Remove existing collection
            try:
                import shutil
                shutil.rmtree(self.qdrant_path)
                print(f"Removed existing Qdrant database at {self.qdrant_path}")
            except Exception as e:
                print(f"Warning: Could not remove existing database: {e}")
        
        print("Creating Qdrant vector store...")
        vector_store = Qdrant.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings,
            path=self.qdrant_path,
            collection_name=self.collection_name,
        )
        
        print(f"✅ LangChain vector index built with {len(langchain_docs)} documents")
        return vector_store
    
    def load_vector_store(self) -> Qdrant:
        """Load existing Qdrant vector store."""
        return Qdrant(
            embedding=self.embeddings,
            path=self.qdrant_path,
            collection_name=self.collection_name,
        )


if __name__ == "__main__":
    # Build LangChain index
    builder = LangChainIndexBuilder()
    
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
    
    # Build index
    print("\nBuilding LangChain vector index...")
    vector_store = builder.build_vector_index(documents)
    
    print(f"\n✅ Index built successfully!")
    print(f"Collection: {builder.collection_name}")
    print(f"Path: {builder.qdrant_path}")

