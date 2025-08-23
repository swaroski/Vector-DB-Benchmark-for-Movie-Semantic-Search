import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from .base import VectorDB
import uuid


class ChromaDB(VectorDB):
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "movies"):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def setup(self, dim: int) -> None:
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            # Delete existing collection to start fresh
            self.client.delete_collection(name=self.collection_name)
        except (ValueError, Exception):
            # Collection doesn't exist, which is fine
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Insert vectors and metadata into ChromaDB."""
        if not vectors or not payloads:
            return
            
        # Process in batches to handle ChromaDB batch size limits
        batch_size = 1000  # Conservative batch size
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            
            # Generate unique IDs for this batch
            ids = [str(uuid.uuid4()) for _ in range(len(batch_vectors))]
            
            # ChromaDB expects metadata as dict with string values
            metadatas = []
            documents = []
            
            for payload in batch_payloads:
                # Convert all metadata values to strings for ChromaDB
                metadata = {k: str(v) for k, v in payload.items() if k not in ['embedding']}
                metadatas.append(metadata)
                
                # Use title or movieId as document content
                doc_text = payload.get('title', payload.get('movieId', ''))
                documents.append(str(doc_text))
            
            # Add batch to collection
            self.collection.add(
                embeddings=batch_vectors,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors in ChromaDB."""
        if not self.collection:
            return []
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['metadatas'] and results['metadatas'][0]:
            for i in range(len(results['metadatas'][0])):
                result = results['metadatas'][0][i].copy()
                result['score'] = float(1.0 / (1.0 + results['distances'][0][i]))  # Convert distance to similarity
                result['distance'] = float(results['distances'][0][i])
                result['document'] = results['documents'][0][i] if results['documents'] else ''
                formatted_results.append(result)
        
        return formatted_results

    def teardown(self) -> None:
        """Clean up ChromaDB resources."""
        if self.client and self.collection:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
        self.collection = None

    def close(self) -> None:
        """Close ChromaDB client."""
        # ChromaDB client doesn't require explicit closing
        self.client = None
        self.collection = None