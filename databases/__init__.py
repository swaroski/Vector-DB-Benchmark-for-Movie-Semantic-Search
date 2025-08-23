from .base import VectorDB

# Import database clients with graceful error handling
available_databases = ['VectorDB']

try:
    from .faiss_client import FaissDB
    available_databases.append('FaissDB')
except ImportError:
    FaissDB = None

try:
    from .chroma_client import ChromaDB
    available_databases.append('ChromaDB')
except ImportError:
    ChromaDB = None

try:
    from .pinecone_client import PineconeDB
    available_databases.append('PineconeDB')
except ImportError:
    PineconeDB = None

try:
    from .weaviate_client import WeaviateDB
    available_databases.append('WeaviateDB')
except ImportError:
    WeaviateDB = None

try:
    from .qdrant_client import QdrantDB
    available_databases.append('QdrantDB')
except ImportError:
    QdrantDB = None

try:
    from .milvus_client import MilvusDB
    available_databases.append('MilvusDB')
except ImportError:
    MilvusDB = None

try:
    from .topk_client import TopKDB
    available_databases.append('TopKDB')
except ImportError:
    TopKDB = None

__all__ = available_databases