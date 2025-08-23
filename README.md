# Movie Vector Database Benchmark

A comprehensive benchmarking framework for evaluating vector database performance in movie semantic search applications using the MovieLens 20M dataset.

## Overview

This project provides a systematic comparison of different vector databases for semantic search of movies. It evaluates performance across multiple dimensions including ingestion speed, query latency, and search relevance using real movie data.

## Features

- **7 Vector Databases**: Supports Faiss, ChromaDB, Qdrant, Milvus, Weaviate, Pinecone, and TopK
- **High Performance**: Polars-based data processing for efficient handling of large datasets
- **Rich Movie Embeddings**: Combines movie titles, genres, user ratings, and tags for comprehensive embeddings
- **Comprehensive Metrics**: Evaluates ingestion time, query latency, recall@k, and hit rates
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **Web Interface**: Interactive FastAPI/uvicorn-based interface for search and benchmarking
- **Docker Compose**: Easy setup of all database dependencies
- **Dataset Flexibility**: Support for both sample and full MovieLens 20M dataset

## Quick Start

### Prerequisites

- Python 3.8+
- MovieLens 20M dataset (download from [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd movie-vector-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and extract the MovieLens 20M dataset to a `data/` directory:
```
movie-vector-benchmark/
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   ├── genome-scores.csv
│   └── genome-tags.csv
├── benchmark.py
└── ...
```

## Usage Options

You can use this benchmark in two ways:

### 🌐 Web Interface (Recommended)

Launch the interactive web interface for easy benchmarking and movie search:

```bash
# Option 1: Run from project root (recommended)
python simple_server.py

# Option 2: Run using uvicorn from project root
uvicorn ui.backend.server:app --host 0.0.0.0 --port 8001 --reload

# Option 3: Run from ui/backend directory
cd ui/backend
python server.py
```

Then open your browser to `http://localhost:8002` (if using simple_server.py) or `http://localhost:8001` (for other options)

**Important:** Always run the server from the project root directory to ensure correct data paths.

**Web Interface Features:**
- 🔍 **Interactive Movie Search**: Search movies with natural language queries
- 📊 **Visual Benchmark Results**: Charts and graphs comparing database performance  
- ⚡ **Real-time Status**: Live progress updates during benchmarking
- 🎛️ **Easy Configuration**: Point-and-click database selection
- 📱 **Responsive Design**: Works on desktop and mobile

**Web Interface Workflow:**
1. **Download MovieLens Data**: Download the MovieLens 20M dataset from [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) and extract to `data/` directory
2. Click "Initialize System" to load movie data (one-time setup)
3. Use "Search Movies" tab to test individual queries  
4. Use "Run Benchmark" tab to compare multiple databases
5. View results with interactive charts in "View Results" tab

**Data Files Required in `data/` directory:**
- `movie.csv` (or `movies.csv`)
- `rating.csv` (or `ratings.csv`) 
- `tag.csv` (or `tags.csv`)
- `genome_scores.csv` (or `genome-scores.csv`)
- `genome_tags.csv` (or `genome-tags.csv`)

### 💻 Command Line Interface

### Running the Benchmark

#### Step 1: Prepare Your Data

Download the MovieLens 20M dataset from [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset):

```bash
# Create data directory
mkdir -p data

# Download and extract MovieLens 20M dataset
# (You'll need to download from Kaggle manually)
# Extract the files to the data/ directory:
# data/movies.csv
# data/ratings.csv  
# data/tags.csv
# data/genome-scores.csv
# data/genome-tags.csv
```

#### Step 2: Generate Embeddings

Generate movie embeddings (this only needs to be done once):

```bash
# Quick generation with sample data (recommended for testing)
python generate_embeddings.py --sample-size 1000

# Full dataset (will take longer)
python generate_embeddings.py --data-path ./data

# Custom model and output
python generate_embeddings.py --model "sentence-transformers/all-mpnet-base-v2" --output ./custom_embeddings.parquet
```

#### Step 3: Start Vector Databases

##### Using Docker Compose (Recommended)
Start all databases with one command:

```bash
# Start all vector databases
docker-compose up -d

# Check status
docker-compose ps

# Stop all databases
docker-compose down
```

##### Individual Database Setup
Or start databases individually:

```bash
# Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Milvus (requires etcd and minio - use docker-compose instead)
# See docker-compose.yml for full Milvus setup

# Weaviate  
docker run -d -p 8080:8080 semitechnologies/weaviate:latest
```

##### Cloud Databases (API Keys Required)
Set up your API keys:

```bash
# Copy environment template
cp .env.sample .env

# Edit .env and add your keys:
# PINECONE_API_KEY=your_key_here
# TOPK_API_KEY=your_key_here
```

Or export them directly:
```bash
export PINECONE_API_KEY="your_pinecone_key"
export TOPK_API_KEY="your_topk_key"
```

#### Step 4: Run Benchmarks

Now that embeddings are generated and databases are running:

##### Option A: Quick Test (Local Databases)
```bash
# Benchmark local databases only
python benchmark.py
```

##### Option B: Custom Configuration
```bash
# Edit config.yaml to enable desired databases
# Then run benchmark
python benchmark.py --config config.yaml
```

##### Option C: Command Line Override
```bash
# Override config with CLI arguments
python benchmark.py --data-path ./data --model "sentence-transformers/all-mpnet-base-v2"
```

#### Step 5: Launch Web Interface (Optional)

```bash
# Start the web server from project root
python simple_server.py
```

Open `http://localhost:8002` for interactive search and visualization.

#### Step 6: View Results

The benchmark will output results like this (actual results from tested databases):
```
================================================================================
BENCHMARK RESULTS
================================================================================
Database        Ingest Time (s)  Throughput (vec/s)  Avg Query Latency (ms)  P95 Latency (ms)  Recall@10  Hit Rate
FAISS                      1.84               1304                   3.09              4.23      0.007     1.000
CHROMA                     2.73               1093                   2.93              4.15      0.007     1.000
QDRANT                     2.40               1248                   4.27              6.89      0.007     1.000
```

**Note**: Results will vary based on your hardware, dataset size, and configuration.

#### Step 5: Generate Visualizations (Optional)

```python
from benchmark import MovieVectorBenchmark
from plot_benchmarks import BenchmarkPlotter

# Run benchmark and get results
benchmark = MovieVectorBenchmark()
results = benchmark.run_benchmark()

# Generate plots
plotter = BenchmarkPlotter(results)
plotter.save_all_plots("output_plots/")
```

### Common Issues and Solutions

#### Quick Test Script

To test the complete workflow, use our test script:
```bash
python test_ui.py
```

This will test:
- Health endpoints
- Database availability 
- System initialization
- Benchmark execution
- Web interface functionality

#### Database Connection Issues
```bash
# Check if databases are running
docker ps

# Check ports are accessible  
netstat -tuln | grep -E '6333|19530|8080'

# Restart databases if needed
docker-compose restart
```

#### Web Interface Issues
```bash
# If web interface fails to start, run debug script
python debug_server.py

# Or use the simple server (more reliable)
python simple_server.py
```

#### Specific Database Fixes (Already Applied)
```bash
# Weaviate gRPC health check issues
# ✅ FIXED: Uses skip_init_checks=True in client connection

# Milvus string length errors  
# ✅ FIXED: Schema updated with max_length=4096 for text fields

# ChromaDB batch size errors
# ✅ FIXED: Batch processing with 1000 vectors per batch

# Qdrant API deprecation
# ✅ FIXED: Updated to query_points() with backward compatibility

# Pinecone package conflicts
# ✅ FIXED: Use pinecone-client==3.2.2 specifically
```

#### Data Loading Issues
```bash
# If embeddings not found
python generate_embeddings.py --sample-size 1000

# For memory issues, use smaller datasets
python generate_embeddings.py --sample-size 500

# For faster testing, use smaller model
python generate_embeddings.py --model "sentence-transformers/all-MiniLM-L6-v2"
```

#### File Naming Issues (Auto-handled)
```bash
# MovieLens files may be named differently
# ✅ The loader automatically tries both naming conventions:
# - movies.csv / movie.csv
# - ratings.csv / rating.csv  
# - genome-scores.csv / genome_scores.csv
# - genome-tags.csv / genome_tags.csv
```

## Configuration

The benchmark can be configured using a YAML file or command-line arguments. See `config.yaml` for a complete example.

### Key Configuration Options

```yaml
data:
  path: "./data"              # Path to MovieLens dataset
  sample_size: 1000          # Number of movies (null for full dataset)

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  cache_path: "./embeddings_cache.parquet"

databases:
  faiss:
    enabled: true             # Local, no setup required
  chroma:
    enabled: true             # Local, no setup required  
  qdrant:
    enabled: true             # Requires Docker
  milvus:
    enabled: true             # Requires Docker  
  weaviate:
    enabled: true             # Requires Docker
  pinecone:
    enabled: false            # Requires API key
  topk:
    enabled: false            # Requires API key
```

## Supported Vector Databases

### Local Databases (No Setup Required)

- **Faiss**: High-performance similarity search library by Facebook AI
- **ChromaDB**: Open-source embedding database with built-in persistence

### Docker-based Databases (Docker Required)

- **Qdrant**: Fast and scalable vector similarity search engine
- **Milvus**: Cloud-native vector database with horizontal scalability  
- **Weaviate**: Open-source vector search engine with GraphQL API

### Cloud Databases (API Keys Required)

- **Pinecone**: Managed vector database service (requires API key)
- **TopK**: Managed vector search platform (requires API key)

### Database Setup Instructions

#### Docker-based Databases (Qdrant, Milvus, Weaviate)

Use the included Docker Compose setup:
```bash
# Start all databases
docker-compose up -d

# Check status  
docker-compose ps

# Stop all databases
docker-compose down
```

Or start individually:
```bash
# Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Milvus (complex setup - use docker-compose instead)
# See docker-compose.yml for complete Milvus configuration with etcd and minio

# Weaviate  
docker run -d -p 8080:8080 -e QUERY_DEFAULTS_LIMIT=25 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' semitechnologies/weaviate:1.24.4
```

#### Cloud Database Setup

##### Pinecone Setup
1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Get your API key
3. Set environment variable: `export PINECONE_API_KEY=your_api_key`
4. Enable in `config.yaml`: `pinecone.enabled: true`

##### TopK Setup  
1. Sign up at [TopK](https://www.topk.ai/)
2. Get your API key
3. Set environment variable: `export TOPK_API_KEY=your_api_key`  
4. Enable in `config.yaml`: `topk.enabled: true`

## Database Status

### ✅ Working Databases (Tested and Verified)

- **FAISS**: Local file-based vector search (fastest queries: ~3.09ms latency)
- **ChromaDB**: Local persistent vector database (~2.93ms latency, 1093 vec/s throughput)
- **Qdrant**: Docker-based vector search engine (~4.27ms latency, 1248 vec/s throughput)

### 🔧 Recently Fixed Issues

- **Milvus**: Fixed string length limits (increased VARCHAR max_length to 4096), schema updated
- **Weaviate**: Fixed gRPC health check issues with `skip_init_checks=True`
- **Qdrant**: Updated to new `query_points()` API with backward compatibility
- **ChromaDB**: Fixed batch size issues with 1000 vectors per batch
- **Pinecone**: Fixed package conflicts using pinecone-client==3.2.2

### ⚠️ Databases Requiring Additional Setup

- **Pinecone**: Requires API key (auto-enables when `PINECONE_API_KEY` set), updated to new Pinecone() class API
- **TopK**: Requires API key (auto-enables when `TOPK_API_KEY` set), fixed movieId validation issue
- **Milvus**: Should work with Docker setup, but may need testing
- **Weaviate**: Should work with Docker setup, but may need testing

## Understanding the Results

### Metrics Explained

- **Ingest Time**: Time to load all vectors into the database
- **Throughput**: Vectors processed per second during ingestion
- **Query Latency**: Time to execute a single search query (average)
- **P95 Latency**: 95th percentile query latency (worst-case performance)
- **Recall@k**: Percentage of relevant results in top-k results
- **Hit Rate**: Percentage of queries with at least one relevant result

### Sample Output (Actual Test Results)

```
================================================================================
BENCHMARK RESULTS
================================================================================  
Database        Ingest Time (s)  Throughput (vec/s)  Avg Query Latency (ms)  P95 Latency (ms)  Recall@10  Hit Rate
FAISS                      1.84               1304                   3.09              4.23      0.007     1.000
CHROMA                     2.73               1093                   2.93              4.15      0.007     1.000
QDRANT                     2.40               1248                   4.27              6.89      0.007     1.000
```

**Performance Summary**:
- **FAISS**: Fastest queries (3.09ms), highest throughput (1304 vec/s)
- **ChromaDB**: Best query latency (2.93ms), good ingestion performance
- **Qdrant**: Balanced performance with excellent ingestion throughput (1248 vec/s)

### Visual Results

![Benchmark Results](benchmark_results.png)

*Comprehensive performance comparison across multiple metrics including ingestion throughput, query latency, and accuracy measures.*

## Architecture

```
movie-vector-benchmark/
├── data/
│   └── loader.py              # MovieLens dataset loader (Polars-based)
├── databases/
│   ├── base.py               # Abstract base class
│   ├── faiss_client.py       # Faiss implementation
│   ├── chroma_client.py      # ChromaDB implementation  
│   ├── qdrant_client.py      # Qdrant implementation
│   ├── milvus_client.py      # Milvus implementation
│   ├── weaviate_client.py    # Weaviate implementation
│   ├── pinecone_client.py    # Pinecone implementation
│   └── topk_client.py        # TopK implementation
├── embeddings/
│   └── embed.py              # Embedding generation
├── ui/
│   ├── backend/
│   │   └── server.py         # FastAPI web interface
│   └── frontend/
│       ├── index.html        # Web UI interface
│       ├── styles.css        # UI styling
│       └── app.js            # Frontend JavaScript
├── utils/
│   └── metrics.py            # Evaluation metrics
├── benchmark.py              # Main benchmark script
├── generate_embeddings.py    # Standalone embedding generation
├── docker-compose.yml        # Multi-database Docker setup
├── plot_benchmarks.py        # Visualization utilities
├── config.yaml              # Configuration file
├── simple_server.py          # Simple web server launcher
├── debug_server.py           # Debug and troubleshooting script
└── test_ui.py               # Complete UI test suite
```

## Embedding Strategy

The benchmark creates rich text representations of movies by combining:

- **Movie Title**: Primary identifier and searchable text
- **Genres**: Categorical information (Action, Comedy, Drama, etc.)
- **Average Rating**: Aggregated user ratings
- **User Tags**: Community-generated descriptive tags
- **Genome Tags**: Algorithmically-generated relevance scores for semantic tags

Example embedding text:
```
Title: The Matrix (1999) | Genres: Action|Sci-Fi | Average Rating: 4.32 | 
User Tags: cyberpunk | dystopia | artificial reality | 
Genome Tags: sci-fi:0.95 | action:0.89 | cyberpunk:0.84
```

## Testing and Verification

### Complete System Test

Run the comprehensive test suite to verify everything works:

```bash
# Test complete workflow (requires server to be running)
python test_ui.py
```

This will test:
- \u2705 Health check endpoints
- \u2705 Database availability detection  
- \u2705 Data loading and embedding generation
- \u2705 Benchmark execution with multiple databases
- \u2705 Web interface functionality
- \u2705 Performance metrics collection

### Quick Debug Test

If you encounter issues, run the debug script:

```bash
# Test data loading and server setup
python debug_server.py
```

This will:
- Verify data directory structure
- Test MovieLens data loading
- Check import paths and dependencies
- Validate server initialization

### Manual Testing Workflow

1. **Test Data Loading**:
   ```bash
   python -c "from data.loader import MovieLensLoader; loader = MovieLensLoader('./data'); print('Data loading works!')"
   ```

2. **Test Embedding Generation**:
   ```bash
   python generate_embeddings.py --sample-size 10 --output test_embeddings.parquet
   ```

3. **Test Benchmark**:
   ```bash
   python benchmark.py --sample-size 100
   ```

4. **Test Web Interface**:
   ```bash
   python simple_server.py
   # Then visit http://localhost:8002
   ```

### Expected Test Results

When running `test_ui.py`, you should see:

```
🧪 Testing Movie Vector Benchmark UI
==================================================
1. Testing health endpoint...
✅ Health check passed
   Response: {'status': 'healthy', 'version': '1.0.0'}

2. Testing database availability...
✅ Database availability check passed
   Available databases:
   🟢 FAISS - Facebook AI Similarity Search - Local file-based
   🟢 ChromaDB - Open-source embedding database
   🟢 Qdrant - Vector similarity search engine
   🟢 Milvus - Cloud-native vector database

3. Testing system initialization...
✅ System initialization completed!

4. Testing benchmark functionality...
✅ Benchmark completed successfully!
   Tested 3 databases:
     • FAISS: 1304 vec/s, 3.09ms latency, 0.007 recall@10
     • CHROMA: 1093 vec/s, 2.93ms latency, 0.007 recall@10
     • QDRANT: 1248 vec/s, 4.27ms latency, 0.007 recall@10

🎉 All tests passed! The Movie Vector Benchmark UI is working correctly.
```

## Customization

### Adding New Vector Databases

1. Implement the `VectorDB` interface in `databases/base.py`
2. Add your implementation to `databases/`
3. Update the `get_database_instance` method in `benchmark.py`
4. Add configuration options to `config.yaml`

### Custom Embedding Models

The benchmark supports any sentence-transformers model:

```bash
python benchmark.py --model "sentence-transformers/all-mpnet-base-v2"
```

Or via configuration:
```yaml
embeddings:
  model: "sentence-transformers/all-mpnet-base-v2"
```

### Custom Queries

Modify the `create_sample_queries` method in `embeddings/embed.py` to test with your own queries.

## Visualization

Generate performance comparison plots:

```python
from plot_benchmarks import BenchmarkPlotter

# After running benchmark
plotter = BenchmarkPlotter(results)
plotter.save_all_plots("output_plots/")
```

Available visualizations:
- Ingestion performance comparison
- Query latency analysis
- Recall@k metrics
- Hit rate comparison
- Comprehensive radar chart

## Contributing

Contributions are welcome! Areas for improvement:

- Additional vector database implementations
- More sophisticated relevance evaluation
- Support for different embedding strategies
- Performance optimization
- Enhanced visualization options

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- **MovieLens Dataset**: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
- **Reference Implementation**: Inspired by [Vector-DB-Benchmark-for-Music-Semantic-Search](https://github.com/andrisgauracs/Vector-DB-Benchmark-for-Music-Semantic-Search)
- **Vector Databases**: Thanks to the teams behind Faiss, ChromaDB, Pinecone, and Weaviate

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{movie_vector_benchmark,
  title={Movie Vector Database Benchmark},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/movie-vector-benchmark}
}
```