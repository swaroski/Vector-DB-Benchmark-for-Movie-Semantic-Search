import weaviate
import warnings
from typing import List, Dict, Any
from .base import VectorDB

# Suppress Weaviate warnings
warnings.filterwarnings("ignore", category=ResourceWarning, module="weaviate.warnings")
warnings.filterwarnings("ignore", message=".*unclosed.*", category=ResourceWarning)


class WeaviateDB(VectorDB):
    def __init__(self, url: str = "http://localhost:8080", class_name: str = "Movie"):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate instance URL
            class_name: Name of the Weaviate class/collection
        """
        self.url = url
        self.class_name = class_name
        self.client = None

    def _ensure_connected(self):
        """Ensure client is connected to Weaviate."""
        try:
            if not self.client.is_connected():
                self.client.connect()
        except Exception:
            # If client is closed, re-instantiate
            host = self.url.replace("http://", "").replace("https://", "").split(":")[0]
            port = int(self.url.split(":")[-1]) if ":" in self.url.split("//")[-1] else 8080
            self.client = weaviate.connect_to_local(host=host, port=port)

    def setup(self, dim: int) -> None:
        """Initialize Weaviate client and create schema."""
        host = self.url.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(self.url.split(":")[-1]) if ":" in self.url.split("//")[-1] else 8080
        
        self.client = weaviate.connect_to_local(host=host, port=port)
        
        # Check if class exists
        try:
            existing_classes = [cls.name for cls in self.client.collections.list_all().values()]
            if self.class_name not in existing_classes:
                # Create class schema
                self.client.collections.create(
                    name=self.class_name,
                    properties=[
                        weaviate.classes.config.Property(
                            name="movieId",
                            data_type=weaviate.classes.config.DataType.INT
                        ),
                        weaviate.classes.config.Property(
                            name="title",
                            data_type=weaviate.classes.config.DataType.TEXT
                        ),
                        weaviate.classes.config.Property(
                            name="genres",
                            data_type=weaviate.classes.config.DataType.TEXT
                        ),
                        weaviate.classes.config.Property(
                            name="tags",
                            data_type=weaviate.classes.config.DataType.TEXT
                        )
                    ],
                    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none()
                )
        except Exception as e:
            print(f"Error setting up Weaviate schema: {e}")

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """Insert vectors and metadata into Weaviate."""
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
        """Search for similar vectors in Weaviate."""
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
        """Clean up Weaviate resources."""
        if self.client:
            try:
                self.client.collections.delete(self.class_name)
            except Exception:
                pass

    def close(self) -> None:
        """Close Weaviate client."""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None