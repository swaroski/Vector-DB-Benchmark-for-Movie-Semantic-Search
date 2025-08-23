import os
from typing import List, Dict, Any
from .base import VectorDB

# Try to import TopK SDK - it may not be available
try:
    from topk_sdk import Client
    from topk_sdk.schema import text, f32_vector, vector_index, keyword_index, int as topk_int
    from topk_sdk.query import select, field, fn
    TOPK_AVAILABLE = True
except ImportError:
    TOPK_AVAILABLE = False


class TopKDB(VectorDB):
    def __init__(self, region: str = None, api_key: str = None, collection: str = "movies"):
        """
        Initialize TopK client.
        
        Args:
            region: TopK region
            api_key: TopK API key
            collection: Name of the collection
        """
        if not TOPK_AVAILABLE:
            raise ImportError("TopK SDK not available. Install with: pip install topk-sdk")
        
        self.region = region or os.getenv("TOPK_REGION", "aws-us-east-1-elastica")
        self.api_key = api_key or os.getenv("TOPK_API_KEY")
        self.collection = collection
        
        if not self.api_key:
            raise ValueError("TopK API key is required")
        
        self.client = Client(api_key=self.api_key, region=self.region)
        self.dim = None

    def setup(self, dim: int) -> None:
        """Initialize TopK collection with specified dimensions."""
        self.dim = dim
        
        # Create collection with schema matching movie fields
        schema = {
            "id": text().required(),
            "vector": f32_vector(dimension=dim).required().index(vector_index(metric="cosine")),
            "movieId": text().required().index(keyword_index()),
            "title": text().required().index(keyword_index()),
            "genres": text().index(keyword_index()),
            "text": text(),
        }
        
        try:
            self.client.collections().create(self.collection, schema=schema)
        except Exception as e:
            if "already exists" not in str(e):
                raise

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Insert vectors and metadata into TopK."""
        if not vectors or not payloads:
            return
        
        # Prepare documents for TopK
        docs = []
        for i, (vec, meta) in enumerate(zip(vectors, payloads)):
            movie_id = meta.get("movieId", i)
            doc = {
                "_id": str(movie_id),
                "id": str(movie_id),
                "vector": vec,
                "movieId": str(movie_id),
                "title": str(meta.get("title", "Unknown")),
                "genres": str(meta.get("genres", "")),
                "text": str(meta.get("text", "")),
            }
            docs.append(doc)
        
        # Insert in batches
        BATCH_SIZE = 200
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            self.client.collection(self.collection).upsert(batch)

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors in TopK."""
        # Vector search using TopK SDK query API
        col = self.client.collection(self.collection)
        docs = col.query(
            select(
                "id",
                "movieId",
                "title", 
                "genres",
                "text",
                vector_similarity=fn.vector_distance("vector", query),
            ).topk(field("vector_similarity"), top_k, asc=False)
        )
        
        # Format results to match our interface
        results = []
        for d in docs:
            result = {
                "id": d.get("id"),
                "movieId": d.get("movieId"),
                "title": d.get("title"),
                "genres": d.get("genres"),
                "text": d.get("text"),
                "score": float(d.get("vector_similarity", 0.0)),
            }
            results.append(result)
        
        return results

    def teardown(self) -> None:
        """Clean up TopK resources."""
        try:
            # Optionally delete collection
            pass
        except Exception:
            pass

    def close(self) -> None:
        """Close TopK client."""
        # TopK SDK handles connection management
        pass