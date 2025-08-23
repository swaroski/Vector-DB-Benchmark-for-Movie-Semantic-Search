#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.loader import MovieLensLoader
from embeddings.embed import MovieEmbedder
from databases import VectorDB, FaissDB, ChromaDB, PineconeDB, WeaviateDB, QdrantDB, MilvusDB, TopKDB
from utils.metrics import BenchmarkMetrics


class SearchQuery(BaseModel):
    query: str
    database: str
    top_k: int = 10


class BenchmarkRequest(BaseModel):
    databases: List[str]
    sample_size: int = 1000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class SearchResult(BaseModel):
    movies: List[Dict[str, Any]]
    query_time: float
    total_results: int


class BenchmarkStatus(BaseModel):
    status: str
    progress: float
    message: str
    results: Optional[List[Dict]] = None


class MovieVectorServer:
    def __init__(self):
        self.app = FastAPI(title="Movie Vector Database Benchmark", version="1.0.0")
        self.embedder = None
        self.movie_data = []
        self.embedded_data = []
        self.databases = {}
        self.benchmark_status = {"status": "idle", "progress": 0.0, "message": "Ready"}
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.setup_routes()
        self.setup_cors()
    
    def setup_cors(self):
        """Setup CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def read_root():
            """Serve the main HTML page."""
            frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
            if frontend_path.exists():
                return HTMLResponse(content=frontend_path.read_text(), status_code=200)
            else:
                return HTMLResponse(content="<h1>Movie Vector Benchmark</h1><p>Frontend not found</p>")
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "version": "1.0.0"}
        
        @self.app.get("/api/databases")
        async def get_available_databases():
            """Get list of available databases."""
            databases = {
                "faiss": {"name": "Faiss", "type": "local", "available": True},
                "chroma": {"name": "ChromaDB", "type": "local", "available": True},
                "qdrant": {"name": "Qdrant", "type": "self-hosted", "available": self._check_database_availability("qdrant")},
                "milvus": {"name": "Milvus", "type": "self-hosted", "available": self._check_database_availability("milvus")},
                "weaviate": {"name": "Weaviate", "type": "self-hosted", "available": self._check_database_availability("weaviate")},
                "pinecone": {"name": "Pinecone", "type": "cloud", "available": bool(os.getenv("PINECONE_API_KEY"))},
                "topk": {"name": "TopK", "type": "cloud", "available": bool(os.getenv("TOPK_API_KEY"))}
            }
            return databases
        
        @self.app.post("/api/initialize")
        async def initialize_data(background_tasks: BackgroundTasks):
            """Initialize embeddings and data."""
            if self.embedder is None:
                background_tasks.add_task(self._initialize_data_background)
                return {"message": "Initialization started", "status": "initializing"}
            else:
                return {"message": "Already initialized", "status": "ready"}
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current system status."""
            return self.benchmark_status
        
        @self.app.post("/api/search", response_model=SearchResult)
        async def search_movies(query: SearchQuery):
            """Search for movies using specified database."""
            if not self.embedder or not self.embedded_data:
                raise HTTPException(status_code=400, detail="System not initialized. Call /api/initialize first.")
            
            try:
                # Get database instance
                db = self._get_database_instance(query.database)
                if query.database not in self.databases:
                    # Initialize database if not already done
                    await self._setup_database(query.database, db)
                    self.databases[query.database] = db
                else:
                    db = self.databases[query.database]
                
                # Generate query embedding
                query_embedding = self.embedder.generate_embeddings([query.query], show_progress=False)[0]
                
                # Search
                import time
                start_time = time.time()
                results = db.search(query_embedding.tolist(), query.top_k)
                query_time = time.time() - start_time
                
                return SearchResult(
                    movies=results,
                    query_time=query_time,
                    total_results=len(results)
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @self.app.post("/api/benchmark")
        async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
            """Run benchmark on selected databases."""
            background_tasks.add_task(self._run_benchmark_background, request)
            return {"message": "Benchmark started", "status": "running"}
        
        @self.app.get("/api/benchmark/results")
        async def get_benchmark_results():
            """Get latest benchmark results."""
            return self.benchmark_status
    
    def _check_database_availability(self, db_name: str) -> bool:
        """Check if a database is available."""
        try:
            if db_name == "qdrant":
                import requests
                response = requests.get("http://localhost:6333/health", timeout=2)
                return response.status_code == 200
            elif db_name == "milvus":
                from pymilvus import connections
                connections.connect(host="localhost", port="19530", timeout=2)
                return True
            elif db_name == "weaviate":
                import requests
                response = requests.get("http://localhost:8080/v1/.well-known/ready", timeout=2)
                return response.status_code == 200
            return False
        except Exception:
            return False
    
    def _get_database_instance(self, db_name: str) -> VectorDB:
        """Get database instance by name."""
        if db_name == "faiss":
            return FaissDB()
        elif db_name == "chroma":
            return ChromaDB()
        elif db_name == "qdrant":
            return QdrantDB()
        elif db_name == "milvus":
            return MilvusDB()
        elif db_name == "weaviate":
            return WeaviateDB()
        elif db_name == "pinecone":
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="Pinecone API key not configured")
            return PineconeDB(api_key=api_key)
        elif db_name == "topk":
            api_key = os.getenv("TOPK_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="TopK API key not configured")
            return TopKDB(api_key=api_key)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown database: {db_name}")
    
    async def _setup_database(self, db_name: str, db: VectorDB):
        """Setup database with embeddings."""
        # Setup database
        embedding_dim = self.embedder.embedding_dim
        db.setup(embedding_dim)
        
        # Prepare data for insertion
        vectors = [item['embedding'] for item in self.embedded_data]
        payloads = [{k: v for k, v in item.items() if k != 'embedding'} 
                   for item in self.embedded_data]
        
        # Insert data
        db.upsert(vectors, payloads)
    
    def _initialize_data_background(self):
        """Initialize data in background task."""
        try:
            self.benchmark_status = {"status": "loading", "progress": 0.1, "message": "Loading movie data..."}
            
            # Load movie data (sample for web interface)
            data_path = Path(__file__).parent.parent.parent / "data"
            
            if not data_path.exists():
                self.benchmark_status = {"status": "error", "progress": 0.0, "message": "Data directory not found. Please download MovieLens 20M dataset to data/ directory."}
                return
            
            loader = MovieLensLoader(str(data_path))
            datasets = loader.load_data(sample_size=2000)  # Reasonable size for web interface
            
            self.benchmark_status = {"status": "processing", "progress": 0.3, "message": "Creating movie features..."}
            
            movie_features = loader.get_movie_features()
            self.movie_data = loader.create_text_for_embedding(movie_features)
            
            self.benchmark_status = {"status": "embedding", "progress": 0.5, "message": "Generating embeddings..."}
            
            # Generate embeddings
            self.embedder = MovieEmbedder()
            self.embedded_data = self.embedder.embed_movies(self.movie_data, batch_size=32)
            
            self.benchmark_status = {"status": "ready", "progress": 1.0, "message": f"Ready! Loaded {len(self.embedded_data)} movies"}
            
        except Exception as e:
            self.benchmark_status = {"status": "error", "progress": 0.0, "message": f"Initialization failed: {str(e)}"}
    
    def _run_benchmark_background(self, request: BenchmarkRequest):
        """Run benchmark in background task."""
        try:
            from benchmark import MovieVectorBenchmark
            
            self.benchmark_status = {"status": "running", "progress": 0.0, "message": "Starting benchmark..."}
            
            # Create benchmark instance
            benchmark = MovieVectorBenchmark()
            
            # Override configuration
            benchmark.config['data']['sample_size'] = request.sample_size
            benchmark.config['embeddings']['model'] = request.embedding_model
            
            # Enable selected databases
            for db_name in benchmark.config['databases']:
                benchmark.config['databases'][db_name]['enabled'] = db_name in request.databases
            
            # Run benchmark
            results = benchmark.run_benchmark()
            
            # Format results for web interface
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'database': result.database_name,
                    'ingest_time': round(result.ingest_time, 2),
                    'throughput': round(result.ingest_throughput, 0),
                    'avg_latency': round(result.query_latency_mean * 1000, 2),
                    'p95_latency': round(result.query_latency_p95 * 1000, 2),
                    'recall_at_10': round(result.recall_at_k.get(10, 0), 3),
                    'hit_rate': round(result.hit_rate, 3),
                    'total_vectors': result.total_vectors
                })
            
            self.benchmark_status = {
                "status": "completed", 
                "progress": 1.0, 
                "message": f"Benchmark completed for {len(formatted_results)} databases",
                "results": formatted_results
            }
            
        except Exception as e:
            self.benchmark_status = {"status": "error", "progress": 0.0, "message": f"Benchmark failed: {str(e)}"}


# Global server instance
server = MovieVectorServer()
app = server.app

# Mount static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


def main():
    """Run the server."""
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)]
    )


if __name__ == "__main__":
    main()