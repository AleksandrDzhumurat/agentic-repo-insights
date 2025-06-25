import os
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from vectoriser import VECTOR_INDEX_DIR_NAME, get_data_dir, get_latest_file


class QueryRequest(BaseModel):
    query: str
    max_results: int = 7


class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    total_found: int


class AppState:
    """Global application state to store vector database"""
    vector_db: Optional[FAISS] = None
    embedding_model: Optional[HuggingFaceEmbeddings] = None
    is_initialized: bool = False


app_state = AppState()

def load_vector_db(path: str) -> Optional[FAISS]:
    """Load vector database from disk"""
    try:
        cache_folder = get_data_dir('models')
        embedding_model = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small", 
            cache_folder=cache_folder
        )
        app_state.embedding_model = embedding_model
        print(f'Loading FAISS index from {path}')
        vector_db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        print(f"Vector database loaded from: {path}")
        return vector_db
    except Exception as e:
        print(f"Failed to load vector database: {e}")
        return None

async def startup_event():
    print("Starting up application...")
    load_dotenv()
    latest_index = get_latest_file(get_data_dir(VECTOR_INDEX_DIR_NAME)) # 'e66ffe2a423c76906f1fc18c49f260df'
    vector_db_path = os.path.join(get_data_dir(VECTOR_INDEX_DIR_NAME), latest_index)
    app_state.vector_db = load_vector_db(vector_db_path)
    
    if app_state.vector_db is not None:
        print("Vector database loaded successfully")
        app_state.is_initialized = True
    else:
        print("No existing vector database found. Use /build endpoint to create one.")


async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down application...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    await shutdown_event()


app = FastAPI(
    title="RAG Knowledge Base API",
    description="API for document search using FAISS vector database and embeddings",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Knowledge Base API",
        "status": "running",
        "initialized": app_state.is_initialized
    }

@app.get("/status")
async def get_status():
    """Get application status"""
    return {
        "initialized": app_state.is_initialized,
        "vector_db_ready": app_state.vector_db is not None,
        "total_documents": app_state.vector_db.index.ntotal if app_state.vector_db else 0
    }


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base using semantic similarity search"""
    
    if not app_state.is_initialized:
        raise HTTPException(
            status_code=400, 
            detail="Knowledge base not initialized."
        )
    
    try:
        docs = app_state.vector_db.similarity_search(
            request.query, 
            k=request.max_results
        )
        
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "index": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "length": len(doc.page_content)
            })
        
        return QueryResponse(
            query=request.query,
            results=results,
            total_found=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats")
async def get_database_stats():
    """Get statistics about the vector database"""
    
    if not app_state.is_initialized:
        raise HTTPException(
            status_code=400, 
            detail="Knowledge base not initialized. Use /build endpoint first."
        )
    
    try:
        total_docs = app_state.vector_db.index.ntotal
        embedding_dim = app_state.vector_db.index.d
        
        return {
            "total_documents": total_docs,
            "embedding_dimension": embedding_dim,
            "index_type": type(app_state.vector_db.index).__name__,
            "embedding_model": "thenlper/gte-small"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Replace 'main' with your actual filename
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"],  # Only reload on changes to current directory
    )