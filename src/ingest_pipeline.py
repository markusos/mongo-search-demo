"""Data ingestion pipeline orchestrator."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.config_loader import AppConfig
from src.embedding_service import CachedEmbeddingGenerator, EmbeddingCache, EmbeddingGenerator
from src.mongodb_manager import MongoDBManager
from src.text_processor import TextChunker, TextProcessor
from src.wiki_parser import WikiArticle, WikiXMLParser

# Global worker state (initialized per process)
_worker_processor = None
_worker_embedding_gen = None
_worker_db_manager = None


def _init_worker(config: AppConfig) -> None:
    """Initialize worker process with persistent connections.

    This runs once per worker process to set up reusable components.

    Args:
        config: Application configuration
    """
    global _worker_processor, _worker_embedding_gen, _worker_db_manager

    # Configure worker logging to write to file with configured log level
    # Remove all existing handlers and set up file logging for worker
    try:
        logger.remove()
    except ValueError:
        # No handlers to remove
        pass
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "workers.log", level=config.logging.level)

    # Initialize text processor
    chunker = TextChunker(config=config.text_processing)
    _worker_processor = TextProcessor(chunker=chunker)

    # Initialize embedding generator (reuse across articles)
    if config.embedding.use_mock_embeddings:
        from src.embedding_service import MockEmbeddingGenerator

        _worker_embedding_gen = MockEmbeddingGenerator(config=config.embedding)
    else:
        embedding_gen = EmbeddingGenerator(config=config.embedding)
        if config.embedding.cache_embeddings:
            cache = EmbeddingCache(cache_path=Path(config.embedding.cache_path))
            _worker_embedding_gen = CachedEmbeddingGenerator(
                embedding_gen, cache, verbose_logging=config.logging.embedding_verbose
            )
        else:
            _worker_embedding_gen = embedding_gen

    # Initialize MongoDB connection (reuse across articles)
    _worker_db_manager = MongoDBManager(config=config.mongodb)


def _process_article_worker(article: WikiArticle) -> dict:
    """Worker function to process a single article in parallel.

    Uses global worker state initialized by _init_worker() for connection reuse.

    Args:
        article: Wikipedia article to process

    Returns:
        Dictionary with processing results
    """
    global _worker_processor, _worker_embedding_gen, _worker_db_manager

    try:
        # Process article using persistent components
        chunks = _worker_processor.process_article(text=article.text, title=article.title)

        if not chunks:
            # Log skipped article to worker log file
            logger.warning(
                f"Article '{article.title}' (page_id: {article.page_id}) produced no chunks - skipping database insert"
            )

            return {
                "success": False,
                "skipped": True,
                "error": "No chunks produced",
                "chunks_created": 0,
                "chunks_failed": 0,
                "documents_inserted": 0,
                "embeddings_generated": 0,
                "embeddings_cached": 0,
                "article_title": article.title,
            }

        # Capture cache stats before generating embeddings
        cache_stats_before = None
        if isinstance(_worker_embedding_gen, CachedEmbeddingGenerator):
            cache_stats_before = _worker_embedding_gen.get_cache_stats()

        # Generate embeddings using persistent generator
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = list(_worker_embedding_gen.embed_batch(chunk_texts, show_progress=False))

        # Calculate cache stats delta for this article only
        embeddings_generated = 0
        embeddings_cached = 0
        if isinstance(_worker_embedding_gen, CachedEmbeddingGenerator):
            cache_stats_after = _worker_embedding_gen.get_cache_stats()
            embeddings_cached = cache_stats_after["cache_hits"] - cache_stats_before["cache_hits"]
            embeddings_generated = (
                cache_stats_after["cache_misses"] - cache_stats_before["cache_misses"]
            )

        # Prepare article document
        article_doc = {
            "page_id": article.page_id,
            "title": article.title,
            "full_text": article.text,
            "namespace": article.namespace,
            "timestamp": article.timestamp,
            "chunk_count": len(chunks),
            "metadata": {
                "word_count": len(article.text.split()),
                "language": "en",
            },
        }

        # Prepare chunk documents
        chunk_docs = []
        chunks_with_failed_embeddings = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is None:
                chunks_with_failed_embeddings += 1
                continue

            chunk_doc = {
                "page_id": article.page_id,
                "title": article.title,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "text": chunk.text,
                "embedding": embedding,
                "token_count": chunk.token_count,
            }
            chunk_docs.append(chunk_doc)

        # Insert using persistent MongoDB connection
        _worker_db_manager.insert_article(article_doc)
        result = _worker_db_manager.insert_chunks_bulk(chunk_docs, ordered=False)
        documents_inserted = result["inserted_count"]

        return {
            "success": True,
            "chunks_created": len(chunks),
            "chunks_failed": chunks_with_failed_embeddings,
            "documents_inserted": documents_inserted,
            "embeddings_generated": embeddings_generated,
            "embeddings_cached": embeddings_cached,
            "article_title": article.title,
        }

    except Exception as e:
        # Log full exception details to worker log file
        logger.exception(
            f"Failed to process article '{article.title}' (page_id: {article.page_id}): {e}"
        )
        return {
            "success": False,
            "error": str(e),
            "chunks_created": 0,
            "chunks_failed": 0,
            "documents_inserted": 0,
            "article_title": article.title,
        }


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""

    articles_processed: int = 0
    articles_failed: int = 0
    chunks_created: int = 0
    chunks_failed: int = 0
    embeddings_generated: int = 0
    embeddings_cached: int = 0
    documents_inserted: int = 0
    articles_skipped: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class IngestionPipeline:
    """Orchestrate Wikipedia data ingestion pipeline."""

    def __init__(self, config: AppConfig):
        """Initialize pipeline with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.stats = PipelineStats()
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Setting up pipeline components...")

        # Parser
        self.parser = WikiXMLParser(
            xml_path=self.config.wikipedia.xml_path,
            clean_markup=True,
            filter_redirects=True,
        )

        # Text processor
        chunker = TextChunker(config=self.config.text_processing)
        self.processor = TextProcessor(chunker=chunker)

        # Embedding generator
        if self.config.embedding.use_mock_embeddings:
            from src.embedding_service import MockEmbeddingGenerator

            self.embedding_gen = MockEmbeddingGenerator(config=self.config.embedding)
        else:
            embedding_gen = EmbeddingGenerator(config=self.config.embedding)

            if self.config.embedding.cache_embeddings:
                cache = EmbeddingCache(cache_path=Path(self.config.embedding.cache_path))
                self.embedding_gen = CachedEmbeddingGenerator(
                    embedding_gen, cache, verbose_logging=self.config.logging.embedding_verbose
                )
            else:
                self.embedding_gen = embedding_gen

        # MongoDB manager
        self.db_manager = MongoDBManager(config=self.config.mongodb)

        logger.info("Pipeline components initialized")

    def run(self, resume_from: str | None = None) -> PipelineStats:
        """Run the complete ingestion pipeline.

        Args:
            resume_from: Checkpoint file to resume from

        Returns:
            Pipeline statistics
        """
        logger.info("Starting ingestion pipeline...")

        # Load checkpoint if resuming
        start_article = 0
        if resume_from:
            start_article = self._load_checkpoint(resume_from)
            logger.info(f"Resuming from article {start_article}")

        # Ensure collections and indexes exist
        self.db_manager.setup_collections()

        # Process articles in batches
        article_batch = []
        article_count = 0

        articles = self.parser.parse_stream(max_articles=self.config.wikipedia.max_articles)

        # Create persistent worker pool for the entire run
        num_workers = self.config.pipeline.num_workers
        executor = ProcessPoolExecutor(
            max_workers=num_workers, initializer=_init_worker, initargs=(self.config,)
        )

        try:
            # Create progress bar with better formatting
            total = (
                self.config.wikipedia.max_articles if self.config.wikipedia.max_articles else None
            )
            pbar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

            with tqdm(
                total=total,
                desc="Processing articles",
                unit="article",
                bar_format=pbar_format,
            ) as pbar:
                # Configure logger to write through tqdm to avoid progress bar interference
                logger.remove()
                logger.add(
                    lambda msg: tqdm.write(msg, end=""),
                    level=self.config.logging.level,
                    colorize=True,
                )

                for article in articles:
                    article_count += 1

                    # Skip if resuming and not at start position yet
                    if article_count <= start_article:
                        pbar.update(1)
                        continue

                    article_batch.append(article)

                    # Process batch when full
                    if len(article_batch) >= self.config.pipeline.batch_size:
                        self._process_batch(article_batch, executor)
                        article_batch = []
                        pbar.update(self.config.pipeline.batch_size)

                    # Checkpoint at intervals
                    if article_count % self.config.pipeline.checkpoint_interval == 0:
                        self._save_checkpoint(article_count)

                # Process remaining articles
                if article_batch:
                    self._process_batch(article_batch, executor)
                    pbar.update(len(article_batch))

            # Final checkpoint
            self._save_checkpoint(article_count)

            # Log final statistics
            self._log_stats()

            logger.info("Ingestion pipeline completed successfully")
            return self.stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            # Save checkpoint on failure
            self._save_checkpoint(self.stats.articles_processed)
            raise
        finally:
            # Clean up the executor
            executor.shutdown(wait=True)

    def _process_batch(self, articles: list[WikiArticle], executor: ProcessPoolExecutor) -> None:
        """Process a batch of articles using the provided executor.

        Args:
            articles: Batch of Wikipedia articles to process
            executor: ProcessPoolExecutor to use for parallel processing
        """
        # Log batch summary (first and last article)
        if len(articles) > 0:
            if len(articles) == 1:
                logger.info(f"Processing batch: '{articles[0].title}'")
            else:
                logger.info(
                    f"Processing batch of {len(articles)} articles: '{articles[0].title}' ... '{articles[-1].title}'"
                )

        # Submit all articles for processing
        future_to_article = {
            executor.submit(_process_article_worker, article): article for article in articles
        }

        # Collect results as they complete
        for future in as_completed(future_to_article):
            article = future_to_article[future]
            try:
                result = future.result()
                if result["success"]:
                    self.stats.articles_processed += 1
                    self.stats.chunks_created += result["chunks_created"]
                    self.stats.chunks_failed += result.get("chunks_failed", 0)
                    self.stats.documents_inserted += result["documents_inserted"]
                    self.stats.embeddings_generated += result.get("embeddings_generated", 0)
                    self.stats.embeddings_cached += result.get("embeddings_cached", 0)
                else:
                    # Distinguish between skipped and failed articles
                    if result.get("skipped", False):
                        self.stats.articles_skipped += 1
                    else:
                        self.stats.articles_failed += 1
                        logger.error(
                            f"Failed to process article '{result.get('article_title', article.title)}': {result.get('error')}"
                        )
            except Exception as e:
                logger.error(f"Worker exception for '{article.title}': {e}")
                self.stats.articles_failed += 1

    def _save_checkpoint(self, article_count: int) -> None:
        """Save checkpoint to file.

        Args:
            article_count: Number of articles processed so far
        """
        checkpoint_path = Path(self.config.pipeline.checkpoint_path) / "pipeline_checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "article_count": article_count,
            "stats": self.stats.to_dict(),
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.debug(f"Checkpoint saved at article {article_count}")

    def _load_checkpoint(self, checkpoint_file: str) -> int:
        """Load checkpoint from file.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            Number of articles already processed
        """
        checkpoint_path = Path(checkpoint_file)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            return 0

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Restore stats
        if "stats" in checkpoint:
            for key, value in checkpoint["stats"].items():
                setattr(self.stats, key, value)

        article_count = checkpoint.get("article_count", 0)
        logger.info(f"Loaded checkpoint: {article_count} articles processed")

        return article_count

    def _log_stats(self) -> None:
        """Log pipeline statistics."""
        total_articles = (
            self.stats.articles_processed + self.stats.articles_skipped + self.stats.articles_failed
        )
        logger.info("=" * 60)
        logger.info("Pipeline Statistics:")
        logger.info(f"  Total Articles: {total_articles:,}")
        logger.info(f"  Articles Processed: {self.stats.articles_processed:,}")
        logger.info(f"  Articles Skipped: {self.stats.articles_skipped:,}")
        logger.info(f"  Articles Failed: {self.stats.articles_failed:,}")
        logger.info(f"  Chunks Created: {self.stats.chunks_created:,}")
        logger.info(f"  Chunks Failed: {self.stats.chunks_failed:,}")
        logger.info(f"  Embeddings Generated: {self.stats.embeddings_generated:,}")
        logger.info(f"  Embeddings Cached: {self.stats.embeddings_cached:,}")
        logger.info(f"  Documents Inserted: {self.stats.documents_inserted:,}")

        # Calculate and log cache hit rate from accumulated stats
        total_embeddings = self.stats.embeddings_cached + self.stats.embeddings_generated
        if total_embeddings > 0:
            hit_rate = self.stats.embeddings_cached / total_embeddings * 100
            logger.info(f"  Cache Hit Rate: {hit_rate:.1f}%")

        if self.stats.articles_processed > 0:
            avg_chunks = self.stats.chunks_created / self.stats.articles_processed
            logger.info(f"  Avg Chunks/Article: {avg_chunks:.2f}")

        logger.info("=" * 60)

    def get_stats(self) -> PipelineStats:
        """Get current pipeline statistics.

        Returns:
            Pipeline statistics
        """
        return self.stats

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "db_manager"):
            self.db_manager.close()
        logger.info("Pipeline resources cleaned up")
