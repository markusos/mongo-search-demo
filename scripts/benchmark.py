#!/usr/bin/env -S uv run python
"""Benchmark script for ingestion pipeline performance testing.

This script runs controlled benchmarks of the ingestion pipeline with different
configurations to measure performance improvements.

Usage:
    python scripts/benchmark.py [OPTIONS]

Options:
    --articles N       Number of articles to benchmark (default: 1000)
    --runs N          Number of benchmark runs per configuration (default: 1)
    --output FILE     Output file for results (default: benchmark_results.json)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config_loader import MongoDBConfig, load_config
from src.ingest_pipeline import IngestionPipeline
from src.mongodb_manager import MongoDBManager


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for benchmarking.

    Args:
        verbose: Enable verbose logging
    """
    logger.remove()  # Remove default handler
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


def clean_cache(cache_path: str) -> None:
    """Clean the embedding cache before benchmark run.

    Args:
        cache_path: Path to embedding cache directory
    """
    import shutil

    cache_dir = Path(cache_path)
    if cache_dir.exists():
        logger.info(f"Cleaning embedding cache at {cache_path}...")
        # Count files before deletion
        file_count = sum(
            1 for subdir in cache_dir.iterdir() if subdir.is_dir() for _ in subdir.glob("*.json")
        )
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Deleted {file_count} cached embeddings")
    else:
        logger.info(f"Cache directory {cache_path} does not exist, creating it...")
        cache_dir.mkdir(parents=True, exist_ok=True)


def clean_database(mongodb_config: MongoDBConfig) -> None:
    """Clean the database before benchmark run.

    Args:
        mongodb_config: MongoDB configuration
    """
    logger.info("Cleaning database...")
    with MongoDBManager(config=mongodb_config) as db_manager:
        deleted = db_manager.delete_all_articles()
        logger.info(f"Deleted {deleted} documents")


def run_benchmark(
    config,
    benchmark_name: str,
    run_number: int = 1,
    mock_embed: bool = False,
) -> dict:
    """Run a single benchmark with given configuration.

    Args:
        config: Application configuration
        benchmark_name: Name of this benchmark
        run_number: Run number (for multiple runs)
        mock_embed: Whether to use mock embedding generator

    Returns:
        Dictionary with benchmark results
    """
    logger.info("=" * 80)
    logger.info(f"BENCHMARK: {benchmark_name} (Run {run_number})")
    logger.info("=" * 80)

    # Set mock embedding flag if requested
    if mock_embed:
        logger.info("Using mock embedding generator (no LMStudio required)")
        config.embedding.use_mock_embeddings = True

    # Always clean cache before benchmark run for accurate measurements
    if config.embedding.cache_embeddings:
        clean_cache(config.embedding.cache_path)

    # Clean database before run
    clean_database(config.mongodb)

    # Create pipeline
    pipeline = IngestionPipeline(config=config)

    # Run the pipeline and measure time
    start_time = time.time()
    start_cpu_time = time.process_time()

    try:
        stats = pipeline.run()
        success = True
        error_message = None
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        stats = pipeline.get_stats()
        success = False
        error_message = str(e)
    finally:
        pipeline.close()

    end_time = time.time()
    end_cpu_time = time.process_time()

    # Calculate metrics
    wall_time = end_time - start_time
    cpu_time = end_cpu_time - start_cpu_time

    articles_per_sec = stats.articles_processed / wall_time if wall_time > 0 else 0
    chunks_per_sec = stats.chunks_created / wall_time if wall_time > 0 else 0
    embeddings_per_sec = (
        (stats.embeddings_generated + stats.embeddings_cached) / wall_time if wall_time > 0 else 0
    )

    # Compile results
    results = {
        "benchmark_name": benchmark_name,
        "run_number": run_number,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "error_message": error_message,
        "mock_embedding_used": mock_embed,
        "config": {
            "wikipedia": config.wikipedia.model_dump(),
            "text_processing": config.text_processing.model_dump(),
            "embedding": config.embedding.model_dump(),
            "pipeline": config.pipeline.model_dump(),
        },
        "timing": {
            "wall_time_seconds": round(wall_time, 2),
            "cpu_time_seconds": round(cpu_time, 2),
            "cpu_utilization_percent": round(
                (cpu_time / wall_time * 100) if wall_time > 0 else 0, 1
            ),
        },
        "throughput": {
            "articles_per_second": round(articles_per_sec, 2),
            "chunks_per_second": round(chunks_per_sec, 2),
            "embeddings_per_second": round(embeddings_per_sec, 2),
        },
        "statistics": {
            "articles_processed": stats.articles_processed,
            "articles_failed": stats.articles_failed,
            "articles_skipped": stats.articles_skipped,
            "chunks_created": stats.chunks_created,
            "chunks_failed": stats.chunks_failed,
            "embeddings_generated": stats.embeddings_generated,
            "embeddings_cached": stats.embeddings_cached,
            "documents_inserted": stats.documents_inserted,
        },
    }

    # Log summary
    logger.info("=" * 80)
    logger.info(f"BENCHMARK RESULTS: {benchmark_name}")
    logger.info("-" * 80)
    logger.info(f"Wall Time:        {wall_time:.2f}s")
    logger.info(f"CPU Time:         {cpu_time:.2f}s")
    logger.info(f"CPU Utilization:  {cpu_time / wall_time * 100:.1f}%")
    logger.info(f"Articles/sec:     {articles_per_sec:.2f}")
    logger.info(f"Chunks/sec:       {chunks_per_sec:.2f}")
    logger.info(f"Embeddings/sec:   {embeddings_per_sec:.2f}")
    logger.info(
        f"Articles:         {stats.articles_processed} processed, {stats.articles_failed} failed"
    )
    logger.info(f"Chunks:           {stats.chunks_created} created")
    logger.info(
        f"Cache Hit Rate:   {stats.embeddings_cached / (stats.embeddings_generated + stats.embeddings_cached) * 100 if (stats.embeddings_generated + stats.embeddings_cached) > 0 else 0:.1f}%"
    )
    logger.info("=" * 80)

    return results


def compare_benchmarks(baseline: dict, optimized: dict) -> None:
    """Compare two benchmark results and log improvements.

    Args:
        baseline: Baseline benchmark results
        optimized: Optimized benchmark results
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)

    # Calculate improvements
    wall_time_improvement = (
        (baseline["timing"]["wall_time_seconds"] - optimized["timing"]["wall_time_seconds"])
        / baseline["timing"]["wall_time_seconds"]
        * 100
    )

    throughput_improvement = (
        (
            optimized["throughput"]["articles_per_second"]
            - baseline["throughput"]["articles_per_second"]
        )
        / baseline["throughput"]["articles_per_second"]
        * 100
    )

    speedup = baseline["timing"]["wall_time_seconds"] / optimized["timing"]["wall_time_seconds"]

    logger.info(f"Baseline:  {baseline['benchmark_name']}")
    logger.info(f"Optimized: {optimized['benchmark_name']}")
    logger.info("-" * 80)
    logger.info(
        f"Wall Time:        {baseline['timing']['wall_time_seconds']:.2f}s -> {optimized['timing']['wall_time_seconds']:.2f}s"
    )
    logger.info(f"Improvement:      {wall_time_improvement:+.1f}% faster")
    logger.info(f"Speedup:          {speedup:.2f}x")
    logger.info("-" * 80)
    logger.info(
        f"Throughput:       {baseline['throughput']['articles_per_second']:.2f} -> {optimized['throughput']['articles_per_second']:.2f} articles/sec"
    )
    logger.info(f"Improvement:      {throughput_improvement:+.1f}%")
    logger.info("-" * 80)
    logger.info(
        f"CPU Utilization:  {baseline['timing']['cpu_utilization_percent']:.1f}% -> {optimized['timing']['cpu_utilization_percent']:.1f}%"
    )
    logger.info("=" * 80)


def save_results(results: list[dict], output_file: str) -> None:
    """Save benchmark results to JSON file.

    Args:
        results: List of benchmark results
        output_file: Output file path
    """
    output_path = Path(output_file)

    # Load existing results if file exists
    if output_path.exists():
        with open(output_path) as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    # Append new results
    existing_results.extend(results)

    # Save
    with open(output_path, "w") as f:
        json.dump(existing_results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Benchmark ingestion pipeline performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--articles",
        type=int,
        default=1000,
        help="Number of articles to benchmark (default: 1000)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per configuration (default: 1)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="benchmark",
        help="Name for this benchmark run (default: benchmark)",
    )

    parser.add_argument(
        "--mock-embed",
        action="store_true",
        help="Use mock embedding generator (no LMStudio required)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Load base configuration
    try:
        base_config = load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Find XML file
    xml_path = Path("data/enwiki-latest-pages-articles-multistream.xml.bz2")
    if not xml_path.exists():
        logger.error(f"Wikipedia XML file not found: {xml_path}")
        sys.exit(1)

    logger.info(f"Using XML file: {xml_path}")
    logger.info(f"Benchmarking with {args.articles} articles")
    logger.info("")

    all_results = []

    # Override config values for benchmark
    base_config.wikipedia.xml_path = str(xml_path)
    base_config.wikipedia.max_articles = args.articles
    base_config.pipeline.checkpoint_interval = (
        10000  # Disable frequent checkpointing during benchmark
    )
    base_config.logging.cache_stats_interval = 0  # Disable periodic logging for cleaner output
    base_config.logging.embedding_verbose = False

    # Run benchmark(s)
    for run in range(1, args.runs + 1):
        result = run_benchmark(base_config, args.name, run, mock_embed=args.mock_embed)
        all_results.append(result)

    # Save results
    save_results(all_results, args.output)

    logger.info("")
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
