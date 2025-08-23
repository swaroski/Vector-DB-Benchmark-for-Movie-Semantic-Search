#!/usr/bin/env python3

import os
from pathlib import Path

# Ensure we're in the right directory
project_root = Path(__file__).parent
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")
print(f"Data directory exists: {(project_root / 'data').exists()}")
print(f"movie.csv exists: {(project_root / 'data' / 'movie.csv').exists()}")

# Test the data loader directly
try:
    from data.loader import MovieLensLoader
    loader = MovieLensLoader('./data')
    print("✅ MovieLensLoader created successfully")
    
    # Try to load a small sample
    datasets = loader.load_data(sample_size=10)
    print(f"✅ Successfully loaded {len(datasets)} datasets")
    print("Available datasets:", list(datasets.keys()))
    
except Exception as e:
    print(f"❌ MovieLensLoader failed: {e}")
    import traceback
    traceback.print_exc()

# Test the server initialization path logic
try:
    from pathlib import Path
    
    # Test the same logic as in server.py
    possible_data_paths = [
        Path(__file__).parent.parent.parent / "data",  # Relative to server file
        Path.cwd() / "data",  # Relative to current working directory
        Path("/home/sxb834/workspace/movie-vector-benchmark/data")  # Absolute path
    ]
    
    print("\nTesting data path resolution:")
    for i, path in enumerate(possible_data_paths):
        exists = path.exists()
        movie_exists = (path / "movie.csv").exists() if exists else False
        print(f"  {i+1}. {path}")
        print(f"     Directory exists: {exists}")
        print(f"     movie.csv exists: {movie_exists}")
        if exists and movie_exists:
            print("     ✅ This path would work!")
            break
    
except Exception as e:
    print(f"❌ Path resolution test failed: {e}")

# Test server import
try:
    from ui.backend.server import MovieVectorServer
    server = MovieVectorServer()
    print("✅ Server created successfully")
except Exception as e:
    print(f"❌ Server creation failed: {e}")
    import traceback
    traceback.print_exc()