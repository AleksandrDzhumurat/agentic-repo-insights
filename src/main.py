import os
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from vectoriser import (
    chunk_documents,
    create_vector_database,
    get_data_dir,
    get_raw_knowledge_base,
)


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


def save_vector_db(vector_db: FAISS, path: str):
    """Save vector database to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vector_db.save_local(path)
    print(f"Vector database saved to: {path}")


def load_vector_db(path: str) -> Optional[FAISS]:
    """Load vector database from disk"""
    try:
        cache_folder = get_data_dir('models')
        embedding_model = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small", 
            cache_folder=cache_folder
        )
        # Store embedding model in app state
        app_state.embedding_model = embedding_model
        
        vector_db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        print(f"Vector database loaded from: {path}")
        return vector_db
    except Exception as e:
        print(f"Failed to load vector database: {e}")
        return None


async def startup_event():
    print("Starting up application...")
    load_dotenv()
    vector_db_path = get_data_dir('faiss_index')
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

@app.get("/build", response_model=dict)
async def build_knowledge_base():
    """Build or rebuild the knowledge base from input directory"""
    
    vector_db_path = get_data_dir('faiss_index')
    try:
        # Build knowledge base
        knowledge_base = get_raw_knowledge_base(os.environ['DOCS_DIR'])
        documents = [
            Document(
                page_content=doc["text"], 
                metadata={"source": doc["source"]}
            ) for doc in knowledge_base
        ]
        chunked_docs = chunk_documents(documents)
        app_state.vector_db = create_vector_database(chunked_docs)
        save_vector_db(app_state.vector_db, vector_db_path)
        app_state.is_initialized = True
        return {
            "message": "Knowledge base built successfully",
            "status": "success",
            "stats": {
                "raw_files": len(knowledge_base),
                "documents": len(documents),
                "chunks": len(chunked_docs)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build knowledge base: {str(e)}")

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