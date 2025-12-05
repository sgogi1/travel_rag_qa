"""
FastAPI backend for Travel Agency RAG System.
Provides endpoints for search, query rewriting, and chat.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.baseline_retriever import BaselineRetriever
from retrieval.improved_retriever import ImprovedRetriever
from retrieval.query_rewriter import QueryRewriter
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever

# LangChain retrievers (optional)
try:
    from retrieval.langchain_retriever import LangChainVectorRetriever, LangChainHybridRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LangChainVectorRetriever = None
    LangChainHybridRetriever = None

app = FastAPI(
    title="Travel Agency RAG System",
    description="Retrieval-Augmented Generation system for travel agency Q&A",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retrievers
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes")
BASELINE_INDEX = os.path.join(INDEX_DIR, "baseline")
IMPROVED_INDEX = os.path.join(INDEX_DIR, "improved")

baseline_retriever = None
improved_retriever = None
vector_retriever = None
hybrid_retriever = None
query_rewriter = None

# LangChain retrievers
langchain_vector_retriever = None
langchain_hybrid_retriever = None

# Initialize retrievers on startup
@app.on_event("startup")
async def startup_event():
    global baseline_retriever, improved_retriever, query_rewriter
    global langchain_vector_retriever, langchain_hybrid_retriever
    
    try:
        if os.path.exists(BASELINE_INDEX):
            baseline_retriever = BaselineRetriever(BASELINE_INDEX)
        if os.path.exists(IMPROVED_INDEX):
            improved_retriever = ImprovedRetriever(IMPROVED_INDEX)
        query_rewriter = QueryRewriter()
        
        # Initialize LangChain retrievers if available
        if LANGCHAIN_AVAILABLE:
            try:
                qdrant_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qdrant_db")
                if os.path.exists(qdrant_path):
                    langchain_vector_retriever = LangChainVectorRetriever(qdrant_path=qdrant_path)
                    langchain_hybrid_retriever = LangChainHybridRetriever(
                        qdrant_path=qdrant_path,
                        whoosh_index_path=IMPROVED_INDEX if os.path.exists(IMPROVED_INDEX) else None
                    )
                    print("✅ LangChain retrievers initialized")
            except Exception as e:
                print(f"Warning: Could not initialize LangChain retrievers: {e}")
        # Initialize vector and hybrid retrievers
        try:
            vector_retriever = VectorRetriever()
            hybrid_retriever = HybridRetriever(IMPROVED_INDEX)
        except Exception as ve:
            print(f"Warning: Vector search not available: {ve}")
            print("Note: Vector index may need to be built. Run indexing/index_builder.py")
    except Exception as e:
        print(f"Warning: Could not initialize retrievers: {e}")


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    use_improved: bool = True
    use_vector: bool = False
    use_hybrid: bool = False
    use_langchain: bool = False  # Use LangChain retrievers
    limit: int = 10


class RewriteRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    query: str
    use_improved: bool = True
    limit: int = 5


# Endpoints
@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Travel Agency RAG System API", "status": "running"}


@app.post("/api/search")
async def search(request: SearchRequest):
    """
    Search endpoint - returns retrieved documents.
    
    Supports baseline, improved, vector, and hybrid retrieval methods.
    """
    try:
        # LangChain retrievers (if requested and available)
        if request.use_langchain and LANGCHAIN_AVAILABLE:
            if request.use_hybrid and langchain_hybrid_retriever:
                result = langchain_hybrid_retriever.search(request.query, limit=request.limit)
                return {
                    "method": "langchain_hybrid",
                    "original_query": result["original_query"],
                    "rewritten_query": result.get("rewritten_query"),
                    "results": result["results"],
                    "num_results": result["num_results"]
                }
            elif langchain_vector_retriever:
                result = langchain_vector_retriever.search(request.query, limit=request.limit)
                return {
                    "method": "langchain_vector",
                    "original_query": result["original_query"],
                    "rewritten_query": result.get("rewritten_query"),
                    "results": result["results"],
                    "num_results": result["num_results"]
                }
            else:
                raise HTTPException(status_code=503, detail="LangChain retrievers not available. Please build LangChain index first.")
        
        # Hybrid search (BM25 + Vector)
        if request.use_hybrid:
            if not hybrid_retriever:
                raise HTTPException(status_code=503, detail="Hybrid retriever not available. Please build vector index first.")
            
            result = hybrid_retriever.search(request.query, limit=request.limit, use_hybrid=True)
            return {
                "method": "hybrid",
                "original_query": result["query"],
                "rewritten_query": result.get("bm25_rewritten_query"),
                "results": result["results"],
                "num_results": result["num_results"],
                "bm25_count": result.get("bm25_count", 0),
                "vector_count": result.get("vector_count", 0)
            }
        
        # Vector search only
        elif request.use_vector:
            if not vector_retriever:
                raise HTTPException(status_code=503, detail="Vector retriever not available. Please build vector index first.")
            
            result = vector_retriever.search(request.query, limit=request.limit)
            return {
                "method": "vector",
                "original_query": result["query"],
                "rewritten_query": None,
                "results": result["results"],
                "num_results": result["num_results"]
            }
        
        # Improved (BM25 + structured)
        elif request.use_improved:
            if not improved_retriever:
                raise HTTPException(status_code=503, detail="Improved index not available. Please build the index first.")
            
            result = improved_retriever.search(request.query, limit=request.limit)
            return {
                "method": "improved",
                "original_query": result["original_query"],
                "rewritten_query": result["rewritten_query"],
                "results": result["results"],
                "num_results": result["num_results"]
            }
        
        # Baseline (BM25 only)
        else:
            if not baseline_retriever:
                raise HTTPException(status_code=503, detail="Baseline index not available. Please build the index first.")
            
            results = baseline_retriever.search(request.query, limit=request.limit)
            return {
                "method": "baseline",
                "original_query": request.query,
                "rewritten_query": None,
                "results": results,
                "num_results": len(results)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rewrite-query")
async def rewrite_query(request: RewriteRequest):
    """
    Query rewriting endpoint - converts natural language to structured filters.
    """
    try:
        if not query_rewriter:
            raise HTTPException(status_code=503, detail="Query rewriter not available.")
        
        rewritten = query_rewriter.rewrite_query(request.query)
        return {
            "original_query": request.query,
            "rewritten_query": rewritten
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint - combines retrieval with LLM-generated answer.
    
    This endpoint retrieves relevant documents and could be extended
    to generate answers using an LLM (currently returns retrieved docs).
    """
    try:
        if request.use_improved:
            if not improved_retriever:
                raise HTTPException(status_code=503, detail="Improved index not available.")
            
            result = improved_retriever.search(request.query, limit=request.limit)
            
            # Format context for LLM (could be extended to actually call LLM)
            context_docs = result["results"]
            context_text = "\n\n".join([
                f"Document {i+1}: {doc['name']} ({doc['doc_type']})\n"
                f"Region: {doc.get('region', 'N/A')}\n"
                f"Description: {doc['document'].get('description', '')[:200]}..."
                for i, doc in enumerate(context_docs)
            ])
            
            return {
                "query": request.query,
                "rewritten_query": result["rewritten_query"],
                "context_documents": context_docs,
                "context_text": context_text,
                "answer": f"Found {len(context_docs)} relevant documents. See context_documents for details."
            }
        else:
            if not baseline_retriever:
                raise HTTPException(status_code=503, detail="Baseline index not available.")
            
            results = baseline_retriever.search(request.query, limit=request.limit)
            
            context_text = "\n\n".join([
                f"Document {i+1}: {doc['name']} ({doc['doc_type']})\n"
                f"Region: {doc.get('region', 'N/A')}\n"
                f"Description: {doc['document'].get('description', '')[:200]}..."
                for i, doc in enumerate(results)
            ])
            
            return {
                "query": request.query,
                "rewritten_query": None,
                "context_documents": results,
                "context_text": context_text,
                "answer": f"Found {len(results)} relevant documents. See context_documents for details."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    vector_available = vector_retriever is not None
    hybrid_available = hybrid_retriever is not None
    
    return {
        "status": "healthy",
        "baseline_index": os.path.exists(BASELINE_INDEX),
        "improved_index": os.path.exists(IMPROVED_INDEX),
        "vector_search": vector_available,
        "hybrid_search": hybrid_available
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

