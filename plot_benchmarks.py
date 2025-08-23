#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List
import argparse

from utils.metrics import BenchmarkResult


class BenchmarkPlotter:
    """Visualization class for benchmark results."""
    
    def __init__(self, results: List[BenchmarkResult]):
        """Initialize with benchmark results."""
        self.results = results
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert benchmark results to pandas DataFrame."""
        data = []
        for result in self.results:
            row = {
                'database': result.database_name,
                'ingest_time': result.ingest_time,
                'throughput': result.ingest_throughput,
                'avg_latency_ms': result.query_latency_mean * 1000,
                'p95_latency_ms': result.query_latency_p95 * 1000,
                'recall_at_1': result.recall_at_k.get(1, 0),
                'recall_at_5': result.recall_at_k.get(5, 0),
                'recall_at_10': result.recall_at_k.get(10, 0),
                'recall_at_20': result.recall_at_k.get(20, 0),
                'hit_rate': result.hit_rate,
                'total_vectors': result.total_vectors,
                'vector_dimension': result.vector_dimension
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_ingest_performance(self, figsize=(10, 6)):
        """Plot ingestion time and throughput."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Ingestion time
        bars1 = ax1.bar(self.df['database'], self.df['ingest_time'], 
                       color='skyblue', alpha=0.7)
        ax1.set_ylabel('Ingestion Time (seconds)')
        ax1.set_title('Vector Ingestion Time')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}s', ha='center', va='bottom')
        
        # Throughput
        bars2 = ax2.bar(self.df['database'], self.df['throughput'], 
                       color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Throughput (vectors/second)')
        ax2.set_title('Vector Ingestion Throughput')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_query_performance(self, figsize=(10, 6)):
        """Plot query latency metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Average latency
        bars1 = ax1.bar(self.df['database'], self.df['avg_latency_ms'], 
                       color='lightgreen', alpha=0.7)
        ax1.set_ylabel('Average Latency (ms)')
        ax1.set_title('Average Query Latency')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}ms', ha='center', va='bottom')
        
        # P95 latency
        bars2 = ax2.bar(self.df['database'], self.df['p95_latency_ms'], 
                       color='gold', alpha=0.7)
        ax2.set_ylabel('P95 Latency (ms)')
        ax2.set_title('95th Percentile Query Latency')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_recall_metrics(self, figsize=(12, 8)):
        """Plot recall@k for different k values."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        recall_metrics = ['recall_at_1', 'recall_at_5', 'recall_at_10', 'recall_at_20']
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20']
        colors = ['orchid', 'lightblue', 'lightgreen', 'salmon']
        
        for ax, metric, title, color in zip(axes, recall_metrics, titles, colors):
            bars = ax.bar(self.df['database'], self.df[metric], color=color, alpha=0.7)
            ax.set_ylabel('Recall')
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_hit_rate(self, figsize=(8, 6)):
        """Plot hit rate for each database."""
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(self.df['database'], self.df['hit_rate'], 
                     color='mediumpurple', alpha=0.7)
        ax.set_ylabel('Hit Rate')
        ax.set_title('Hit Rate (Queries with ≥1 Relevant Result)')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_combined_performance(self, figsize=(15, 10)):
        """Create a comprehensive performance comparison."""
        fig = plt.figure(figsize=figsize)
        
        # Create subplots with different sizes
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Ingestion metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(self.df['database'], self.df['ingest_time'], color='skyblue', alpha=0.7)
        ax1.set_title('Ingest Time (s)')
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(self.df['database'], self.df['throughput'], color='lightcoral', alpha=0.7)
        ax2.set_title('Throughput (vec/s)')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Query latency
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(self.df['database'], self.df['avg_latency_ms'], color='lightgreen', alpha=0.7)
        ax3.set_title('Avg Latency (ms)')
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Recall metrics
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(self.df['database'], self.df['recall_at_5'], color='gold', alpha=0.7)
        ax4.set_title('Recall@5')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.bar(self.df['database'], self.df['recall_at_10'], color='orchid', alpha=0.7)
        ax5.set_title('Recall@10')
        ax5.set_ylim(0, 1)
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(self.df['database'], self.df['hit_rate'], color='mediumpurple', alpha=0.7)
        ax6.set_title('Hit Rate')
        ax6.set_ylim(0, 1)
        ax6.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Radar chart for overall comparison
        ax7 = fig.add_subplot(gs[2, :])
        self._create_radar_chart(ax7)
        
        plt.suptitle('Vector Database Performance Comparison', fontsize=16, y=0.95)
        return fig
    
    def _create_radar_chart(self, ax):
        """Create a radar chart comparing all databases."""
        # Normalize metrics to 0-1 scale for radar chart
        normalized_df = self.df.copy()
        
        # Invert latency (lower is better) and normalize
        max_latency = normalized_df['avg_latency_ms'].max()
        normalized_df['latency_score'] = 1 - (normalized_df['avg_latency_ms'] / max_latency)
        
        # Invert ingest time (lower is better) and normalize
        max_ingest = normalized_df['ingest_time'].max()
        normalized_df['ingest_score'] = 1 - (normalized_df['ingest_time'] / max_ingest)
        
        # Normalize throughput (higher is better)
        max_throughput = normalized_df['throughput'].max()
        normalized_df['throughput_score'] = normalized_df['throughput'] / max_throughput
        
        # Metrics for radar chart
        metrics = ['latency_score', 'ingest_score', 'throughput_score', 
                  'recall_at_10', 'hit_rate']
        metric_labels = ['Query Speed', 'Ingest Speed', 'Throughput', 
                        'Recall@10', 'Hit Rate']
        
        # Number of variables
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot for each database
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.df)))
        
        for i, (_, row) in enumerate(normalized_df.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['database'], 
                   color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize the radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Comparison', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def save_all_plots(self, output_dir='plots'):
        """Save all plots to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Individual plots
        plots = [
            ('ingest_performance', self.plot_ingest_performance),
            ('query_performance', self.plot_query_performance),
            ('recall_metrics', self.plot_recall_metrics),
            ('hit_rate', self.plot_hit_rate),
            ('combined_performance', self.plot_combined_performance)
        ]
        
        saved_files = []
        for plot_name, plot_func in plots:
            fig = plot_func()
            filepath = os.path.join(output_dir, f'{plot_name}.png')
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            plt.close(fig)
        
        return saved_files


def main():
    """Command line interface for plotting benchmark results."""
    parser = argparse.ArgumentParser(description='Plot benchmark results')
    parser.add_argument('--show', action='store_true', 
                       help='Show plots interactively')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--output-dir', default='plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Note: This is a standalone plotting utility
    # In practice, you'd load results from benchmark.py
    print("This is a plotting utility. Run benchmark.py first to generate results.")
    print("Then import BenchmarkPlotter and use it with your results.")


if __name__ == "__main__":
    main()