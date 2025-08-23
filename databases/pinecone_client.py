from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from .base import VectorDB
import os
import time


class PineconeDB(VectorDB):
    def __init__(self, api_key: str = None, environment: str = "us-east-1-aws", 
                 index_name: str = "movies", dimension: int = 384):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key (will use PINECONE_API_KEY env var if not provided)
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            dimension: Dimension of vectors
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.pc = None
        self.index = None

    def setup(self, dim: int) -> None:
        """Initialize Pinecone client and create/connect to index."""
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        self.dimension = dim
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists
        existing_indexes = [idx['name'] for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            # Create index if it doesn't exist
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Insert vectors and metadata into Pinecone."""
        if not vectors or not payloads:
            return
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for i, (vector, payload) in enumerate(zip(vectors, payloads)):
            # Use movieId as ID if available, otherwise generate one
            vector_id = str(payload.get('movieId', f'movie_{i}'))
            
            # Filter metadata to ensure compatibility with Pinecone
            metadata = {k: v for k, v in payload.items() 
                       if k != 'embedding' and isinstance(v, (str, int, float, bool))}
            
            vectors_to_upsert.append({
                'id': vector_id,
                'values': vector,
                'metadata': metadata
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        if not self.index:
            return []
        
        # Query Pinecone
        results = self.index.query(
            vector=query,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            result = match.metadata.copy() if match.metadata else {}
            result['score'] = float(match.score)
            result['id'] = match.id
            formatted_results.append(result)
        
        return formatted_results

    def teardown(self) -> None:
        """Clean up Pinecone resources."""
        if self.pc and self.index_name:
            try:
                existing_indexes = [idx['name'] for idx in self.pc.list_indexes()]
                if self.index_name in existing_indexes:
                    self.pc.delete_index(self.index_name)
            except Exception:
                pass

    def close(self) -> None:
        """Close Pinecone client."""
        self.index = None
        self.pc = None