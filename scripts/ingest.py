#!/usr/bin/env -S uv run python
"""Script to ingest Wikipedia articles into MongoDB.

This script orchestrates the complete pipeline:
1. Parse Wikipedia XML dump
2. Process and chunk articles
3. Generate embeddings
4. Store in MongoDB

Usage:
    uv run python scripts/ingest_wikipedia.py [OPTIONS]

Options:
    --max-articles N    Maximum number of articles to process (overrides env)
    --resume           Resume from last checkpoint
    --clean            Clean database before ingesting
    --help             Show this help message
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config_loader import load_config
from src.ingest_pipeline import IngestionPipeline


def setup_logging(verbose: bool = False, log_level: str = "INFO") -> None:
    """Configure logging.

    Args:
        verbose: Enable verbose logging (overrides log_level)
        log_level: Log level from config (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()  # Remove default handler
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        # Allow environment variable to override config
        level = os.getenv("LOG_LEVEL", log_level)
        logger.add(sys.stderr, level=level)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia articles into MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--xml-file",
        type=str,
        help="Path to Wikipedia XML dump (default: from current directory)",
    )

    parser.add_argument(
        "--max-articles",
        type=int,
        help="Maximum number of articles to process (overrides MAX_ARTICLES env var)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean database before ingesting (WARNING: deletes all data)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N articles (default: 100)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )

    return parser.parse_args()


def find_wikipedia_xml() -> str | None:
    """Find Wikipedia XML file in current directory.

    Returns:
        Path to XML file or None if not found
    """
    # Check for compressed file first
    xml_files = list(Path.cwd().glob("enwiki*.xml.bz2"))
    if xml_files:
        return str(xml_files[0])

    # Check for uncompressed file
    xml_files = list(Path.cwd().glob("enwiki*.xml"))
    if xml_files:
        return str(xml_files[0])

    return None


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Load configuration from config.yaml first
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.error("Please ensure config/config.yaml exists and is valid")
        sys.exit(1)

    # Setup logging with config values
    setup_logging(args.verbose, config.logging.level)

    # Get max articles from args or config
    max_articles = args.max_articles
    if max_articles is None:
        max_articles = config.wikipedia.max_articles

    if max_articles is None or max_articles == 0:
        max_articles = None  # Unlimited
        logger.info("Processing unlimited articles (may take a long time!)")
    else:
        logger.info(f"Processing maximum of {max_articles:,} articles")

    # Find XML file
    xml_file = args.xml_file or config.wikipedia.xml_path
    if not xml_file:
        logger.error("No Wikipedia XML file found")
        logger.error("Please specify --xml-file or place enwiki*.xml(.bz2) in current directory")
        sys.exit(1)

    if not Path(xml_file).exists():
        logger.error(f"XML file not found: {xml_file}")
        sys.exit(1)

    logger.info(f"Wikipedia XML: {xml_file}")
    logger.info(f"MongoDB: {config.mongodb.uri}")
    logger.info(f"Database: {config.mongodb.database}")
    logger.info(f"LMStudio: {config.embedding.lmstudio_url}")
    logger.info(f"Embedding Model: {config.embedding.model}")
    logger.info(f"Chunk Size: {config.text_processing.chunk_size} tokens")
    logger.info(f"Chunk Overlap: {config.text_processing.chunk_overlap} tokens")
    logger.info(f"Batch Size: {config.pipeline.batch_size}")

    try:
        # Override config values from command-line args
        if max_articles is not None:
            config.wikipedia.max_articles = max_articles
        if xml_file:
            config.wikipedia.xml_path = xml_file
        if args.checkpoint_interval:
            config.pipeline.checkpoint_interval = args.checkpoint_interval
        if args.no_cache:
            config.embedding.cache_embeddings = False
        if args.verbose:
            config.logging.embedding_verbose = True

        # Create pipeline (this initializes all components)
        logger.info("Initializing pipeline...")
        pipeline = IngestionPipeline(config=config)
        logger.info("✓ Pipeline initialized")

        # Clean database if requested
        if args.clean:
            logger.warning("Cleaning database (deleting all articles and chunks)...")
            pipeline.db_manager.delete_all_articles()
            pipeline.db_manager.chunks_collection.delete_many({})
            logger.info("✓ Database cleaned")

        # Show database statistics before ingestion
        articles_count = pipeline.db_manager.articles_collection.count_documents({})
        chunks_count = pipeline.db_manager.chunks_collection.count_documents({})
        logger.info(f"Database before: {articles_count:,} articles, {chunks_count:,} chunks")

        # Run ingestion
        logger.info("=" * 70)
        logger.info("Starting ingestion pipeline...")
        logger.info("=" * 70)

        import time

        start_time = time.time()
        checkpoint_file = (
            config.pipeline.checkpoint_path + "/pipeline_checkpoint.json" if args.resume else None
        )
        stats = pipeline.run(resume_from=checkpoint_file)
        elapsed_time = time.time() - start_time

        # Show final statistics
        logger.info("=" * 70)
        logger.info("Ingestion Complete!")
        logger.info("=" * 70)
        logger.info(f"Articles processed: {stats.articles_processed:,}")
        logger.info(f"Articles skipped: {stats.articles_skipped:,}")
        logger.info(f"Articles failed: {stats.articles_failed:,}")
        logger.info(f"Chunks created: {stats.chunks_created:,}")
        logger.info(f"Embeddings generated: {stats.embeddings_generated:,}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")

        if stats.articles_processed > 0:
            avg_time = elapsed_time / stats.articles_processed
            logger.info(f"Average per article: {avg_time:.2f} seconds")

        # Show database statistics after ingestion
        articles_count = pipeline.db_manager.articles_collection.count_documents({})
        chunks_count = pipeline.db_manager.chunks_collection.count_documents({})
        logger.info(f"Database after: {articles_count:,} articles, {chunks_count:,} chunks")

        logger.info("✓ Ingestion successful!")

    except KeyboardInterrupt:
        logger.warning("\n\nIngestion interrupted by user")
        logger.info("Progress has been saved. Use --resume to continue.")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    finally:
        if "pipeline" in locals():
            pipeline.db_manager.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()
