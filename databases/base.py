from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDB(ABC):
    """Abstract base class for vector database implementations."""
    
    @abstractmethod
    def setup(self, dim: int) -> None:
        """Initialize the vector database with specified dimensions."""
        pass
    
    @abstractmethod
    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Insert or update vectors with associated metadata."""
        pass
    
    @abstractmethod
    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors and return top_k results."""
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Clean up database resources."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connections."""
        pass