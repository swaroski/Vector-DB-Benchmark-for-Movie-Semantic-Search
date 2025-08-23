#!/usr/bin/env python3

import argparse
import os
import time
from pathlib import Path

from data.loader import MovieLensLoader
from embeddings.embed import MovieEmbedder


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for MovieLens dataset')
    parser.add_argument('--data-path', type=str, default='./data',
                       help='Path to MovieLens data directory')
    parser.add_argument('--output', type=str, default='./embeddings_cache.parquet',
                       help='Output file for embeddings')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of movies to sample (None for full dataset)')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model to use')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MOVIE EMBEDDINGS GENERATION")
    print("=" * 60)
    
    # Check if output file already exists
    if os.path.exists(args.output):
        overwrite = input(f"Output file {args.output} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Aborted.")
            return
    
    # Load MovieLens data
    print(f"\n1. Loading MovieLens data from {args.data_path}...")
    start_time = time.time()
    
    loader = MovieLensLoader(args.data_path)
    datasets = loader.load_data(sample_size=args.sample_size)
    
    print(f"   ✓ Loaded in {time.time() - start_time:.2f} seconds")
    
    # Create movie features
    print("\n2. Creating movie features...")
    start_time = time.time()
    
    movie_features = loader.get_movie_features()
    movie_data = loader.create_text_for_embedding(movie_features)
    
    print(f"   ✓ Created features for {len(movie_data)} movies in {time.time() - start_time:.2f} seconds")
    
    # Generate embeddings
    print(f"\n3. Generating embeddings using {args.model}...")
    start_time = time.time()
    
    embedder = MovieEmbedder(model_name=args.model)
    embedded_data = embedder.embed_movies(movie_data, batch_size=args.batch_size)
    
    generation_time = time.time() - start_time
    print(f"   ✓ Generated {len(embedded_data)} embeddings in {generation_time:.2f} seconds")
    print(f"   ✓ Throughput: {len(embedded_data) / generation_time:.2f} embeddings/second")
    
    # Save embeddings
    print(f"\n4. Saving embeddings to {args.output}...")
    start_time = time.time()
    
    embedder.save_embeddings(embedded_data, args.output)
    
    print(f"   ✓ Saved in {time.time() - start_time:.2f} seconds")
    
    # Summary
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Dataset: MovieLens 20M {'(sampled)' if args.sample_size else '(full)'}")
    print(f"Movies processed: {len(embedded_data):,}")
    print(f"Embedding model: {args.model}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    print(f"Output file: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
    print()
    print("Next steps:")
    print("1. Start your vector databases (Docker containers)")
    print("2. Run the benchmark: python benchmark.py")
    print("3. Launch web UI: cd ui/backend && python server.py")


if __name__ == "__main__":
    main()