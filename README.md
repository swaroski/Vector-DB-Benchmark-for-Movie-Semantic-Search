# Movie Vector Database Benchmark

A comprehensive benchmarking framework for evaluating vector database performance in movie semantic search applications using the MovieLens 20M dataset.

## Overview

This project provides a systematic comparison of different vector databases for semantic search of movies. It evaluates performance across multiple dimensions including ingestion speed, query latency, and search relevance using real movie data.

## Features

- **Multiple Vector Databases**: Supports Faiss, ChromaDB, Pinecone, and Weaviate
- **Rich Movie Embeddings**: Combines movie titles, genres, user ratings, and tags for comprehensive embeddings
- **Comprehensive Metrics**: Evaluates ingestion time, query latency, recall@k, and hit rates
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **Visualization**: Built-in plotting utilities for performance comparison
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

### Running the Benchmark

1. **Quick Start (with sample data)**:
```bash
python benchmark.py --sample-size 1000
```

2. **Custom configuration**:
```bash
python benchmark.py --config config.yaml --data-path ./data
```

3. **Full dataset benchmark**:
```bash
python benchmark.py --data-path ./data
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
    enabled: true
  chroma:
    enabled: true
  pinecone:
    enabled: false            # Requires API key
  weaviate:
    enabled: false            # Requires running instance
```

## Supported Vector Databases

### Local Databases (No Setup Required)

- **Faiss**: High-performance similarity search library by Facebook AI
- **ChromaDB**: Open-source embedding database with built-in persistence

### Cloud/Self-Hosted Databases

- **Pinecone**: Managed vector database service (requires API key)
- **Weaviate**: Open-source vector search engine (requires running instance)

### Setting Up Cloud/Self-Hosted Databases

#### Pinecone Setup
1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Get your API key
3. Set environment variable: `export PINECONE_API_KEY=your_api_key`
4. Enable in `config.yaml`: `pinecone.enabled: true`

#### Weaviate Setup
1. **Local Docker**:
```bash
docker run -d -p 8080:8080 semitechnologies/weaviate:latest
```
2. **Cloud**: Use Weaviate Cloud Services
3. Enable in `config.yaml`: `weaviate.enabled: true`

## Understanding the Results

### Metrics Explained

- **Ingest Time**: Time to load all vectors into the database
- **Throughput**: Vectors processed per second during ingestion
- **Query Latency**: Time to execute a single search query
- **Recall@k**: Percentage of relevant results in top-k results
- **Hit Rate**: Percentage of queries with at least one relevant result

### Sample Output

```
================================================================================
BENCHMARK RESULTS
================================================================================
Database  Ingest Time (s)  Throughput (vec/s)  Avg Query Latency (ms)  P95 Query Latency (ms)  Recall@10  Hit Rate
FAISS                1.23                 813                   2.45                    4.12      0.156     0.800
CHROMA               2.87                 348                   8.91                   15.23      0.142     0.700
```

## Architecture

```
movie-vector-benchmark/
├── data/
│   └── loader.py           # MovieLens dataset loader
├── databases/
│   ├── base.py            # Abstract base class
│   ├── faiss_client.py    # Faiss implementation
│   ├── chroma_client.py   # ChromaDB implementation
│   ├── pinecone_client.py # Pinecone implementation
│   └── weaviate_client.py # Weaviate implementation
├── embeddings/
│   └── embed.py           # Embedding generation
├── utils/
│   └── metrics.py         # Evaluation metrics
├── benchmark.py           # Main benchmark script
├── plot_benchmarks.py     # Visualization utilities
└── config.yaml           # Configuration file
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