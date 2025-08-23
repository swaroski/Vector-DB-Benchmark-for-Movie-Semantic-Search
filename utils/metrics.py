import time
import numpy as np
from typing import List, Dict, Any, Callable
import statistics
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    database_name: str
    ingest_time: float
    ingest_throughput: float  # vectors per second
    query_latency_mean: float
    query_latency_std: float
    query_latency_p95: float
    recall_at_k: Dict[int, float]  # recall at different k values
    hit_rate: float
    total_vectors: int
    vector_dimension: int
    query_count: int


class BenchmarkMetrics:
    """Utility class for calculating benchmark metrics."""
    
    @staticmethod
    def measure_ingest_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """
        Measure the execution time of a function.
        
        Returns:
            Tuple of (function_result, execution_time_seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def measure_query_latency(search_func: Callable, queries: List[List[float]], 
                            top_k: int = 10) -> List[float]:
        """
        Measure query latency for multiple search operations.
        
        Args:
            search_func: Function that performs the search
            queries: List of query vectors
            top_k: Number of top results to retrieve
            
        Returns:
            List of query latencies in seconds
        """
        latencies = []
        
        for query in queries:
            start_time = time.time()
            _ = search_func(query, top_k)
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        return latencies
    
    @staticmethod
    def calculate_recall_at_k(retrieved_results: List[List[Dict]], 
                            ground_truth: List[List[str]], 
                            k_values: List[int] = None) -> Dict[int, float]:
        """
        Calculate recall@k metric.
        
        Args:
            retrieved_results: List of retrieved results for each query
            ground_truth: List of relevant items for each query
            k_values: List of k values to calculate recall for
            
        Returns:
            Dictionary mapping k values to recall scores
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]
        
        recall_scores = {k: [] for k in k_values}
        
        for retrieved, relevant in zip(retrieved_results, ground_truth):
            if not relevant:  # Skip if no ground truth
                continue
                
            relevant_set = set(relevant)
            
            for k in k_values:
                # Get top k retrieved items
                top_k_retrieved = [str(item.get('movieId', item.get('id', ''))) 
                                 for item in retrieved[:k]]
                top_k_retrieved_set = set(top_k_retrieved)
                
                # Calculate recall
                intersect = len(relevant_set.intersection(top_k_retrieved_set))
                recall = intersect / len(relevant_set) if relevant_set else 0
                recall_scores[k].append(recall)
        
        # Average recall scores
        avg_recall = {}
        for k in k_values:
            if recall_scores[k]:
                avg_recall[k] = statistics.mean(recall_scores[k])
            else:
                avg_recall[k] = 0.0
        
        return avg_recall
    
    @staticmethod
    def calculate_hit_rate(retrieved_results: List[List[Dict]], 
                          ground_truth: List[List[str]], k: int = 10) -> float:
        """
        Calculate hit rate (percentage of queries with at least one relevant result in top k).
        
        Args:
            retrieved_results: List of retrieved results for each query
            ground_truth: List of relevant items for each query
            k: Number of top results to consider
            
        Returns:
            Hit rate as a float between 0 and 1
        """
        hits = 0
        total_queries = 0
        
        for retrieved, relevant in zip(retrieved_results, ground_truth):
            if not relevant:  # Skip if no ground truth
                continue
                
            total_queries += 1
            relevant_set = set(relevant)
            
            # Get top k retrieved items
            top_k_retrieved = [str(item.get('movieId', item.get('id', ''))) 
                             for item in retrieved[:k]]
            top_k_retrieved_set = set(top_k_retrieved)
            
            # Check if there's at least one hit
            if len(relevant_set.intersection(top_k_retrieved_set)) > 0:
                hits += 1
        
        return hits / total_queries if total_queries > 0 else 0.0
    
    @staticmethod
    def create_heuristic_ground_truth(query_text: str, all_movies: List[Dict]) -> List[str]:
        """
        Create heuristic ground truth based on text matching.
        This is a simple approach - in practice, you'd want human-annotated ground truth.
        
        Args:
            query_text: The search query
            all_movies: All available movies with metadata
            
        Returns:
            List of movie IDs that are considered relevant
        """
        query_lower = query_text.lower()
        relevant_movies = []
        
        # Simple keyword matching for genres
        genre_keywords = {
            'action': ['action'],
            'comedy': ['comedy'],
            'drama': ['drama'],
            'horror': ['horror'],
            'romance': ['romance'],
            'sci-fi': ['sci-fi', 'science fiction'],
            'thriller': ['thriller'],
            'animation': ['animation'],
            'musical': ['musical'],
            'western': ['western']
        }
        
        # Year patterns
        year_patterns = ['90s', '1990', '1940s', '2000s']
        
        for movie in all_movies:
            movie_text = f"{movie.get('title', '')} {movie.get('genres', '')}".lower()
            
            # Genre matching
            for keyword_group in genre_keywords.values():
                for keyword in keyword_group:
                    if keyword in query_lower and keyword in movie_text:
                        relevant_movies.append(str(movie.get('movieId', movie.get('id', ''))))
                        break
            
            # Rating matching (for "high ratings" queries)
            if 'high rating' in query_lower:
                avg_rating = movie.get('avg_rating', 0)
                if avg_rating is not None and avg_rating >= 4.0:  # Consider 4.0+ as high rating
                    relevant_movies.append(str(movie.get('movieId', movie.get('id', ''))))
        
        return list(set(relevant_movies))  # Remove duplicates
    
    @staticmethod
    def create_benchmark_result(database_name: str, ingest_time: float, 
                              total_vectors: int, vector_dimension: int,
                              query_latencies: List[float], 
                              retrieved_results: List[List[Dict]],
                              ground_truth: List[List[str]]) -> BenchmarkResult:
        """
        Create a comprehensive benchmark result.
        
        Args:
            database_name: Name of the vector database
            ingest_time: Time taken to ingest all vectors
            total_vectors: Total number of vectors ingested
            vector_dimension: Dimension of the vectors
            query_latencies: List of query latencies
            retrieved_results: Retrieved results for each query
            ground_truth: Ground truth for each query
            
        Returns:
            BenchmarkResult object
        """
        # Calculate metrics
        ingest_throughput = total_vectors / ingest_time if ingest_time > 0 else 0
        
        latency_mean = statistics.mean(query_latencies) if query_latencies else 0
        latency_std = statistics.stdev(query_latencies) if len(query_latencies) > 1 else 0
        latency_p95 = np.percentile(query_latencies, 95) if query_latencies else 0
        
        recall_at_k = BenchmarkMetrics.calculate_recall_at_k(retrieved_results, ground_truth)
        hit_rate = BenchmarkMetrics.calculate_hit_rate(retrieved_results, ground_truth)
        
        return BenchmarkResult(
            database_name=database_name,
            ingest_time=ingest_time,
            ingest_throughput=ingest_throughput,
            query_latency_mean=latency_mean,
            query_latency_std=latency_std,
            query_latency_p95=latency_p95,
            recall_at_k=recall_at_k,
            hit_rate=hit_rate,
            total_vectors=total_vectors,
            vector_dimension=vector_dimension,
            query_count=len(query_latencies)
        )