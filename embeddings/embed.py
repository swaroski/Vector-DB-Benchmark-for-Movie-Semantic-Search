import os
import numpy as np
import polars as pl
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


class MovieEmbedder:
    """Generate embeddings for movies using various methods."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: Optional[str] = None):
        """
        Initialize the movie embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, 
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def embed_movies(self, movie_data: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Generate embeddings for movie data.
        
        Args:
            movie_data: List of dictionaries with 'text' and 'metadata' keys
            batch_size: Batch size for embedding generation
            
        Returns:
            List of dictionaries with embeddings and metadata
        """
        texts = [item['text'] for item in movie_data]
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        # Combine embeddings with metadata
        embedded_data = []
        for i, (item, embedding) in enumerate(zip(movie_data, embeddings)):
            embedded_item = {
                'embedding': embedding.tolist(),
                'text': item['text'],
                **item['metadata']
            }
            embedded_data.append(embedded_item)
        
        return embedded_data
    
    def save_embeddings(self, embedded_data: List[Dict], output_path: str):
        """
        Save embeddings to a parquet file.
        
        Args:
            embedded_data: List of dictionaries with embeddings and metadata
            output_path: Path to save the parquet file
        """
        # Convert to Polars DataFrame for efficient storage
        df_data = []
        for item in embedded_data:
            row = item.copy()
            # Convert embedding to string for parquet storage
            row['embedding'] = str(row['embedding'])
            df_data.append(row)
        
        df = pl.DataFrame(df_data)
        df.write_parquet(output_path)
        print(f"Embeddings saved to {output_path}")
    
    def load_embeddings(self, input_path: str) -> List[Dict]:
        """
        Load embeddings from a parquet file.
        
        Args:
            input_path: Path to the parquet file
            
        Returns:
            List of dictionaries with embeddings and metadata
        """
        df = pl.read_parquet(input_path)
        embedded_data = []
        
        for row in df.to_dicts():
            # Convert embedding string back to list
            row['embedding'] = eval(row['embedding'])
            embedded_data.append(row)
        
        print(f"Loaded {len(embedded_data)} embeddings from {input_path}")
        return embedded_data
    
    @staticmethod
    def create_sample_queries() -> List[Dict[str, str]]:
        """
        Create sample queries for testing the vector database.
        
        Returns:
            List of sample queries with descriptions
        """
        queries = [
            {
                "query": "action movies with high ratings",
                "description": "Popular action films"
            },
            {
                "query": "romantic comedy from the 90s",
                "description": "90s romantic comedies"
            },
            {
                "query": "sci-fi thriller with time travel",
                "description": "Time travel sci-fi thrillers"
            },
            {
                "query": "animated family movies for children",
                "description": "Family-friendly animated films"
            },
            {
                "query": "dark psychological drama with twist ending",
                "description": "Dark psychological dramas"
            },
            {
                "query": "superhero movies with ensemble cast",
                "description": "Superhero ensemble films"
            },
            {
                "query": "indie film with experimental narrative",
                "description": "Experimental indie films"
            },
            {
                "query": "horror movies with supernatural elements",
                "description": "Supernatural horror films"
            },
            {
                "query": "classic film noir from the 1940s",
                "description": "1940s film noir classics"
            },
            {
                "query": "musical with memorable songs and dance numbers",
                "description": "Musical films with great songs"
            }
        ]
        return queries