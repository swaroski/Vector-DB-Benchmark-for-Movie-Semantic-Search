import math
from typing import List, Dict, Any
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from .base import VectorDB


class MilvusDB(VectorDB):
    def __init__(self, host: str = "localhost", port: str = "19530", collection: str = "movies"):
        """
        Initialize Milvus client.
        
        Args:
            host: Milvus host
            port: Milvus port
            collection: Name of the collection
        """
        self.host = host
        self.port = port
        self.collection_name = collection
        self.col = None

    @staticmethod
    def _safe_str(x, default=""):
        """Safely convert value to string."""
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return str(x)

    def setup(self, dim: int) -> None:
        """Initialize Milvus collection with specified dimensions."""
        # Raise gRPC message limits on the client
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            max_send_message_length=256 * 1024 * 1024,  # 256 MB
            max_receive_message_length=256 * 1024 * 1024,  # 256 MB
        )
        
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        # Define schema for movie data
        fields = [
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
            ),
            FieldSchema(name="movieId", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="genres", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        ]
        
        schema = CollectionSchema(fields, description="Movie embeddings")
        self.col = Collection(self.collection_name, schema, consistency_level="Strong")
        # Index and load will happen after inserts for efficiency

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Insert vectors and metadata into Milvus."""
        if not vectors or not payloads:
            return
        
        # Prepare columns
        N = len(vectors)
        ids = list(range(N))
        movie_ids = [int(p.get("movieId", i)) for i, p in enumerate(payloads)]
        titles = [self._safe_str(p.get("title", "Unknown"))[:512] for p in payloads]  # Limit length
        genres = [self._safe_str(p.get("genres", "Unknown"))[:256] for p in payloads]
        texts = [self._safe_str(p.get("text", ""))[:2048] for p in payloads]  # Limit length
        
        # Insert in batches
        BATCH = 200  # standardized batch size
        for i in range(0, N, BATCH):
            sl = slice(i, i + BATCH)
            self.col.insert([
                ids[sl],
                movie_ids[sl],
                vectors[sl],
                titles[sl],
                genres[sl],
                texts[sl],
            ])
        
        self.col.flush()
        
        # Build HNSW index (align with other backends)
        self.col.create_index(
            field_name="vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 128},
            },
        )
        self.col.load()

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors in Milvus."""
        if self.col is None:
            connections.connect(alias="default", host=self.host, port=self.port)
            assert utility.has_collection(self.collection_name), "Milvus collection missing"
            self.col = Collection(self.collection_name)
            self.col.load()
        
        # Search parameters
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        
        # Perform search
        results = self.col.search(
            data=[query],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["movieId", "title", "genres", "text"],
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    'id': hit.id,
                    'movieId': hit.entity.get('movieId'),
                    'title': hit.entity.get('title'),
                    'genres': hit.entity.get('genres'),
                    'text': hit.entity.get('text'),
                    'score': float(hit.score),
                    'distance': float(1.0 - hit.score)  # Convert similarity to distance
                }
                formatted_results.append(result)
        
        return formatted_results

    def teardown(self) -> None:
        """Clean up Milvus resources."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

    def close(self) -> None:
        """Close Milvus client."""
        # Stub implementation; add resource cleanup if needed
        pass