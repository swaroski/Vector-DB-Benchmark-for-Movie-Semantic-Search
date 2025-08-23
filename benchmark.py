#!/usr/bin/env python3

import argparse
import os
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from data.loader import MovieLensLoader
from embeddings.embed import MovieEmbedder
from databases import VectorDB, PineconeDB, WeaviateDB, FaissDB, ChromaDB
from utils.metrics import BenchmarkMetrics, BenchmarkResult


class MovieVectorBenchmark:
    """Main benchmarking class for movie vector databases."""
    
    def __init__(self, config_path: str = None):
        """Initialize the benchmark with configuration."""
        self.config = self.load_config(config_path)
        self.embedder = None
        self.movie_data = None
        self.embedded_data = None
        self.queries = None
        
    def load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file."""
        default_config = {
            'data': {
                'path': './data',
                'sample_size': None  # None for full dataset
            },
            'embeddings': {
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'batch_size': 32,
                'cache_path': './embeddings_cache.parquet'
            },
            'databases': {
                'faiss': {
                    'enabled': True,
                    'index_path': './faiss_index.bin',
                    'metadata_path': './faiss_metadata.pkl'
                },
                'chroma': {
                    'enabled': True,
                    'persist_directory': './chroma_db'
                },
                'pinecone': {
                    'enabled': False,  # Requires API key
                    'api_key': None,
                    'index_name': 'movies'
                },
                'weaviate': {
                    'enabled': False,  # Requires running instance
                    'url': 'http://localhost:8080'
                }
            },
            'benchmark': {
                'top_k': 10,
                'query_count': 10
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge with defaults
            default_config.update(user_config)
        
        return default_config
    
    def prepare_data(self):
        """Load and prepare movie data for benchmarking."""
        print("Loading MovieLens data...")
        loader = MovieLensLoader(self.config['data']['path'])
        datasets = loader.load_data(sample_size=self.config['data']['sample_size'])
        
        print("Creating movie features...")
        movie_features = loader.get_movie_features()
        self.movie_data = loader.create_text_for_embedding(movie_features)
        
        print(f"Prepared {len(self.movie_data)} movies for embedding")
    
    def generate_embeddings(self):
        """Generate embeddings for movie data."""
        cache_path = self.config['embeddings']['cache_path']
        
        # Check if embeddings are cached
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            self.embedder = MovieEmbedder(self.config['embeddings']['model'])
            self.embedded_data = self.embedder.load_embeddings(cache_path)
        else:
            print("Generating embeddings...")
            self.embedder = MovieEmbedder(self.config['embeddings']['model'])
            self.embedded_data = self.embedder.embed_movies(
                self.movie_data, 
                batch_size=self.config['embeddings']['batch_size']
            )
            # Cache embeddings
            self.embedder.save_embeddings(self.embedded_data, cache_path)
    
    def prepare_queries(self):
        """Prepare queries for benchmarking."""
        # Use sample queries from embedder
        sample_queries = self.embedder.create_sample_queries()
        
        # Limit number of queries
        query_count = min(len(sample_queries), self.config['benchmark']['query_count'])
        selected_queries = sample_queries[:query_count]
        
        # Generate embeddings for queries
        query_texts = [q['query'] for q in selected_queries]
        query_embeddings = self.embedder.generate_embeddings(query_texts, show_progress=False)
        
        self.queries = []
        for i, (query_info, embedding) in enumerate(zip(selected_queries, query_embeddings)):
            self.queries.append({
                'text': query_info['query'],
                'description': query_info['description'],
                'embedding': embedding.tolist(),
                'ground_truth': BenchmarkMetrics.create_heuristic_ground_truth(
                    query_info['query'], self.embedded_data
                )
            })
        
        print(f"Prepared {len(self.queries)} queries for benchmarking")
    
    def get_database_instance(self, db_name: str, db_config: Dict) -> VectorDB:
        """Get database instance based on configuration."""
        if db_name == 'faiss':
            return FaissDB(
                index_path=db_config.get('index_path', './faiss_index.bin'),
                metadata_path=db_config.get('metadata_path', './faiss_metadata.pkl')
            )
        elif db_name == 'chroma':
            return ChromaDB(
                persist_directory=db_config.get('persist_directory', './chroma_db')
            )
        elif db_name == 'pinecone':
            api_key = db_config.get('api_key') or os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("Pinecone API key is required")
            return PineconeDB(
                api_key=api_key,
                index_name=db_config.get('index_name', 'movies')
            )
        elif db_name == 'weaviate':
            return WeaviateDB(
                url=db_config.get('url', 'http://localhost:8080')
            )
        else:
            raise ValueError(f"Unknown database: {db_name}")
    
    def benchmark_database(self, db_name: str, db_config: Dict) -> BenchmarkResult:
        """Benchmark a specific database."""
        print(f"\nBenchmarking {db_name.upper()}...")
        
        try:
            # Get database instance
            db = self.get_database_instance(db_name, db_config)
            
            # Setup database
            embedding_dim = self.embedder.embedding_dim
            db.setup(embedding_dim)
            
            # Measure ingestion time
            vectors = [item['embedding'] for item in self.embedded_data]
            payloads = [{k: v for k, v in item.items() if k != 'embedding'} 
                       for item in self.embedded_data]
            
            print(f"Ingesting {len(vectors)} vectors...")
            _, ingest_time = BenchmarkMetrics.measure_ingest_time(
                db.upsert, vectors, payloads
            )
            print(f"Ingestion completed in {ingest_time:.2f} seconds")
            
            # Measure query performance
            print("Running queries...")
            query_latencies = []
            retrieved_results = []
            
            for query in self.queries:
                latency_start = time.time()
                results = db.search(query['embedding'], self.config['benchmark']['top_k'])
                latency_end = time.time()
                
                query_latencies.append(latency_end - latency_start)
                retrieved_results.append(results)
            
            # Create ground truth list
            ground_truth = [query['ground_truth'] for query in self.queries]
            
            # Create benchmark result
            result = BenchmarkMetrics.create_benchmark_result(
                database_name=db_name,
                ingest_time=ingest_time,
                total_vectors=len(vectors),
                vector_dimension=embedding_dim,
                query_latencies=query_latencies,
                retrieved_results=retrieved_results,
                ground_truth=ground_truth
            )
            
            # Cleanup
            db.close()
            
            return result
            
        except Exception as e:
            print(f"Error benchmarking {db_name}: {str(e)}")
            return None
    
    def run_benchmark(self) -> List[BenchmarkResult]:
        """Run benchmark on all enabled databases."""
        print("Starting Movie Vector Database Benchmark")
        print("=" * 50)
        
        # Prepare data and embeddings
        self.prepare_data()
        self.generate_embeddings()
        self.prepare_queries()
        
        # Run benchmarks
        results = []
        db_configs = self.config['databases']
        
        for db_name, db_config in db_configs.items():
            if db_config.get('enabled', False):
                result = self.benchmark_database(db_name, db_config)
                if result:
                    results.append(result)
        
        return results
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results in a formatted table."""
        if not results:
            print("No results to display.")
            return
        
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        # Create results table
        table_data = []
        for result in results:
            table_data.append({
                'Database': result.database_name.upper(),
                'Ingest Time (s)': f"{result.ingest_time:.2f}",
                'Throughput (vec/s)': f"{result.ingest_throughput:.0f}",
                'Avg Query Latency (ms)': f"{result.query_latency_mean * 1000:.2f}",
                'P95 Query Latency (ms)': f"{result.query_latency_p95 * 1000:.2f}",
                'Recall@10': f"{result.recall_at_k.get(10, 0):.3f}",
                'Hit Rate': f"{result.hit_rate:.3f}"
            })
        
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        print()


def main():
    parser = argparse.ArgumentParser(description='Movie Vector Database Benchmark')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-path', type=str, help='Path to MovieLens data directory')
    parser.add_argument('--sample-size', type=int, help='Number of movies to sample')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model to use')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = MovieVectorBenchmark(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        benchmark.config['data']['path'] = args.data_path
    if args.sample_size:
        benchmark.config['data']['sample_size'] = args.sample_size
    if args.model:
        benchmark.config['embeddings']['model'] = args.model
    
    # Run benchmark
    try:
        results = benchmark.run_benchmark()
        benchmark.print_results(results)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")


if __name__ == "__main__":
    main()