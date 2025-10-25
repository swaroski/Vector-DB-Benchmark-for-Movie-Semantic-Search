import weaviate
import warnings
from typing import List, Dict, Any
from .base import VectorDB

# Suppress Weaviate warnings
warnings.filterwarnings("ignore", category=ResourceWarning, module="weaviate.warnings")
warnings.filterwarnings("ignore", message=".*unclosed.*", category=ResourceWarning)


class WeaviateDB(VectorDB):
    def __init__(self, url: str = "http://localhost:8080", api_key: str = None, class_name: str = "Movie"):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate instance URL (for cloud: just the cluster URL without https://)
            api_key: Weaviate API key (for cloud instances)
            class_name: Name of the Weaviate class/collection
        """
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self.client = None
        self.is_cloud = api_key is not None

    def _ensure_connected(self):
        """Ensure client is connected to Weaviate v4."""
        try:
            if hasattr(self.client, 'is_connected') and not self.client.is_connected():
                self.client.connect()
        except Exception:
            pass

    def setup(self, dim: int) -> None:
        """Initialize Weaviate client and create schema."""
        try:
            import weaviate
            from weaviate.classes.init import AdditionalConfig, Timeout, Auth
            
            if self.is_cloud:
                cluster_url = self.url
                if not cluster_url.startswith('http'):
                    cluster_url = f'https://{cluster_url}'
                
                # Use simple connect_to_custom for cloud
                self.client = weaviate.WeaviateClient(
                    connection_params=weaviate.connect.ConnectionParams.from_params(
                        http_host=self.url,
                        http_port=443,
                        http_secure=True,
                        grpc_host=self.url,
                        grpc_port=50051,
                        grpc_secure=True
                    ),
                    auth_client_secret=Auth.api_key(self.api_key),
                    additional_config=AdditionalConfig(
                        timeout=Timeout(init=30, query=60, insert=120)
                    )
                )
                self.client.connect()
                print(f"Connected to Weaviate Cloud: {self.url}")
            else:
                self.client = weaviate.connect_to_local(
                    host="localhost",
                    port=8080,
                    skip_init_checks=True,
                    additional_config=AdditionalConfig(
                        timeout=Timeout(init=30, query=60, insert=120)
                    )
                )
                
                import requests
                health = requests.get("http://localhost:8080/v1/.well-known/ready", timeout=5)
                if health.status_code != 200:
                    raise Exception("Weaviate not ready")
                print("Connected to Weaviate local")
        except Exception as e:
            print(f"Connection failed: {e}")
            raise Exception(f"Could not connect to Weaviate: {e}")
        
        # Check if class exists and create schema using v4 API
        try:
            from weaviate.classes.config import Configure, Property, DataType, VectorDistances
            
            existing_classes = [cls.name for cls in self.client.collections.list_all().values()]
            if self.class_name not in existing_classes:
                # Create class schema using v4 API with proper vectorizer config
                self.client.collections.create(
                    name=self.class_name,
                    properties=[
                        Property(name="movieId", data_type=DataType.INT),
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="genres", data_type=DataType.TEXT),
                        Property(name="tags", data_type=DataType.TEXT)
                    ],
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    )
                )
                print(f"Created Weaviate collection: {self.class_name}")
        except Exception as e:
            print(f"Error setting up Weaviate schema: {e}")

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Insert vectors and metadata into Weaviate using v4 API."""
        if not vectors or not payloads:
            return
        
        self._ensure_connected()
        collection = self.client.collections.get(self.class_name)
        
        # Prepare objects for insertion
        objects = []
        for vector, payload in zip(vectors, payloads):
            # Clean payload for Weaviate
            clean_payload = {}
            for key, value in payload.items():
                if key != 'embedding':
                    if isinstance(value, list):
                        clean_payload[key] = str(value)  # Convert lists to strings
                    else:
                        clean_payload[key] = value
            
            objects.append(
                weaviate.classes.data.DataObject(
                    properties=clean_payload,
                    vector=vector
                )
            )
        
        # Insert objects in batches
        batch_size = 100
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            collection.data.insert_many(batch)

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors in Weaviate using v4 API."""
        self._ensure_connected()
        collection = self.client.collections.get(self.class_name)
        
        # Perform vector search
        response = collection.query.near_vector(
            near_vector=query,
            limit=top_k,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        
        # Format results
        results = []
        for obj in response.objects:
            result = obj.properties.copy()
            result['score'] = 1.0 / (1.0 + obj.metadata.distance)  # Convert distance to similarity
            result['distance'] = obj.metadata.distance
            result['id'] = str(obj.uuid)
            results.append(result)
        
        return results

    def teardown(self) -> None:
        """Clean up Weaviate resources using v4 API."""
        if self.client:
            try:
                self.client.collections.delete(self.class_name)
            except Exception:
                pass

    def close(self) -> None:
        """Close Weaviate client."""
        if self.client:
            try:
                if hasattr(self.client, 'close'):
                    self.client.close()
            except Exception:
                pass
        self.client = None