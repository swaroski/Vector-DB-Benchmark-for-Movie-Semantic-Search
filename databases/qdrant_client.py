from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff, SearchParams
from .base import VectorDB

HTTP_TIMEOUT = 300.0  # seconds
BATCH_SIZE = 200  # standardized batch size
PARALLEL = 2  # 0 = auto, or small integer


class QdrantDB(VectorDB):
    def __init__(self, url: str = "http://localhost:6333", collection: str = "movies"):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant instance URL
            collection: Name of the collection
        """
        self.client = QdrantClient(url=url, timeout=HTTP_TIMEOUT)
        self.collection = collection

    def setup(self, dim: int) -> None:
        """Initialize Qdrant collection with specified dimensions."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Insert vectors and metadata into Qdrant."""
        if not vectors or not payloads:
            return
        
        # Use the high level bulk uploader. It handles batching and retries.
        self.client.upload_collection(
            collection_name=self.collection,
            vectors=vectors,
            payload=payloads,
            ids=list(range(len(vectors))),
            batch_size=BATCH_SIZE,
            parallel=PARALLEL,  # 0 picks a sensible default based on CPU
            max_retries=5,
            wait=True,  # wait for the whole upload to finish
        )

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant."""
        from qdrant_client.models import PointStruct
        
        try:
            # Try new API first
            res = self.client.query_points(
                collection_name=self.collection,
                query=query,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128),
            )
            # Handle new API response format
            results = []
            for point in res.points:
                result = point.payload.copy() if point.payload else {}
                result['id'] = point.id
                result['score'] = float(point.score)
                results.append(result)
            return results
            
        except (AttributeError, Exception):
            # Fall back to old API if new one doesn't exist
            res = self.client.search(
                collection_name=self.collection,
                query_vector=query,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128),
            )
            # Handle old API response format
            results = []
            for r in res:
                result = r.payload.copy() if r.payload else {}
                result['id'] = r.id
                result['score'] = float(r.score)
                results.append(result)
            return results

    def teardown(self) -> None:
        """Clean up Qdrant resources."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def close(self) -> None:
        """Close Qdrant client."""
        # QdrantClient does not require explicit close, but method is needed for interface
        pass