"""Data ingestion pipeline orchestrator."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.embedding_service import CachedEmbeddingGenerator, EmbeddingCache, EmbeddingGenerator
from src.mongodb_manager import MongoDBManager
from src.text_processor import TextProcessor
from src.wiki_parser import WikiArticle, WikiXMLParser


@dataclass
class PipelineConfig:
    """Configuration for ingestion pipeline."""

    xml_path: str
    mongodb_uri: str
    database_name: str = "wikipedia_kb"
    articles_collection: str = "wiki_articles"
    chunks_collection: str = "wiki_chunks"
    batch_size: int = 100
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: str = "semantic"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q8_0"
    embedding_batch_size: int = 32
    max_articles: int | None = None
    checkpoint_interval: int = 1000
    checkpoint_path: str = "./checkpoints/pipeline_checkpoint.json"
    cache_embeddings: bool = True
    embedding_cache_path: str = "./embedding_cache"
    clean_markup: bool = True
    skip_redirects: bool = True


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

    def __init__(
        self,
        config: PipelineConfig,
        verbose_embedding_logs: bool = False,
        cache_stats_interval: int = 100,
    ):
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
            verbose_embedding_logs: Whether to log every embedding batch
            cache_stats_interval: Log cache stats every N articles (0 to disable)
        """
        self.config = config
        self.stats = PipelineStats()
        self.verbose_embedding_logs = verbose_embedding_logs
        self.cache_stats_interval = cache_stats_interval
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Setting up pipeline components...")

        # Parser
        self.parser = WikiXMLParser(
            xml_path=self.config.xml_path,
            clean_markup=self.config.clean_markup,
            filter_redirects=self.config.skip_redirects,
        )

        # Text processor
        from src.text_processor import TextChunker

        chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy,
        )
        self.processor = TextProcessor(chunker=chunker)

        # Embedding generator
        embedding_gen = EmbeddingGenerator(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        )

        if self.config.cache_embeddings:
            cache = EmbeddingCache(cache_path=Path(self.config.embedding_cache_path))
            self.embedding_gen = CachedEmbeddingGenerator(
                embedding_gen, cache, verbose_logging=self.verbose_embedding_logs
            )
        else:
            self.embedding_gen = embedding_gen

        # MongoDB manager
        self.db_manager = MongoDBManager(
            connection_string=self.config.mongodb_uri,
            database_name=self.config.database_name,
            articles_collection=self.config.articles_collection,
            chunks_collection=self.config.chunks_collection,
        )

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

        try:
            # Ensure collections and indexes exist
            self.db_manager.setup_collections()

            # Process articles in batches
            article_batch = []
            article_count = 0

            articles = self.parser.parse_stream(max_articles=self.config.max_articles)

            # Create progress bar with better formatting
            total = self.config.max_articles if self.config.max_articles else None
            pbar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            with tqdm(
                total=total,
                desc="Processing articles",
                unit="article",
                bar_format=pbar_format,
            ) as pbar:
                for article in articles:
                    article_count += 1

                    # Skip if resuming and not at start position yet
                    if article_count <= start_article:
                        pbar.update(1)
                        continue

                    article_batch.append(article)

                    # Process batch when full
                    if len(article_batch) >= self.config.batch_size:
                        self._process_batch(article_batch)
                        article_batch = []
                        pbar.update(self.config.batch_size)

                        # Log cache stats periodically
                        if (
                            self.cache_stats_interval > 0
                            and self.stats.articles_processed % self.cache_stats_interval == 0
                            and isinstance(self.embedding_gen, CachedEmbeddingGenerator)
                        ):
                            self._log_cache_stats(pbar)

                    # Checkpoint at intervals
                    if article_count % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(article_count)

                # Process remaining articles
                if article_batch:
                    self._process_batch(article_batch)
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

    def _process_batch(self, articles: list[WikiArticle]) -> None:
        """Process a batch of articles.

        Args:
            articles: Batch of Wikipedia articles to process
        """
        for article in articles:
            try:
                self._process_article(article)
                self.stats.articles_processed += 1
            except Exception as e:
                logger.error(
                    f"Failed to process article '{article.title}' (ID: {article.page_id}): {e}"
                )
                self.stats.articles_failed += 1

    def _process_article(self, article: WikiArticle) -> None:
        """Process a single article through the pipeline.

        Args:
            article: Wikipedia article to process
        """
        # Check if article already exists
        if self.db_manager.article_exists(article.page_id):
            logger.debug(f"Article {article.page_id} already exists, skipping")
            self.stats.articles_skipped += 1
            return

        # Step 1: Chunk the article
        chunks = self.processor.process_article(text=article.text, title=article.title)

        if not chunks:
            logger.warning(f"No chunks generated for article '{article.title}'")
            return

        self.stats.chunks_created += len(chunks)

        # Step 2: Generate embeddings for chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = list(self.embedding_gen.embed_batch(chunk_texts, show_progress=False))

        # Count cache hits if using cached generator
        if isinstance(self.embedding_gen, CachedEmbeddingGenerator):
            cache_stats = self.embedding_gen.get_cache_stats()
            self.stats.embeddings_cached = cache_stats["cache_hits"]
            self.stats.embeddings_generated = cache_stats["cache_misses"]

        # Step 3: Prepare article document
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

        # Step 4: Prepare chunk documents
        chunk_docs = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is None:
                logger.warning(f"Failed to generate embedding for chunk {i} of '{article.title}'")
                self.stats.chunks_failed += 1
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

        # Step 5: Insert into MongoDB
        try:
            # Insert article
            self.db_manager.insert_article(article_doc)

            # Bulk insert chunks
            if chunk_docs:
                result = self.db_manager.insert_chunks_bulk(chunk_docs, ordered=False)
                self.stats.documents_inserted += result["inserted_count"]

                if result["errors"]:
                    logger.warning(
                        f"Some chunks failed to insert for '{article.title}': "
                        f"{len(result['errors'])} errors"
                    )

        except Exception as e:
            logger.error(f"Failed to insert documents for '{article.title}': {e}")
            self.stats.articles_failed += 1
            raise

    def _save_checkpoint(self, article_count: int) -> None:
        """Save checkpoint to file.

        Args:
            article_count: Number of articles processed so far
        """
        checkpoint_path = Path(self.config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "article_count": article_count,
            "stats": self.stats.to_dict(),
            "config": asdict(self.config),
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

    def _log_cache_stats(self, pbar) -> None:
        """Log cache statistics during processing.

        Args:
            pbar: Progress bar instance
        """
        if isinstance(self.embedding_gen, CachedEmbeddingGenerator):
            cache_stats = self.embedding_gen.get_cache_stats()
            total = cache_stats["cache_hits"] + cache_stats["cache_misses"]
            hit_rate = (
                (cache_stats["cache_hits"] / total * 100) if total > 0 else 0
            )

            # Write to progress bar's postfix instead of separate log line
            pbar.set_postfix(
                {
                    "chunks": self.stats.chunks_created,
                    "cache_hit_rate": f"{hit_rate:.1f}%",
                }
            )

            logger.info(
                f"Progress: {self.stats.articles_processed} articles | "
                f"{self.stats.chunks_created} chunks | "
                f"Cache: {cache_stats['cache_hits']:,} hits, "
                f"{cache_stats['cache_misses']:,} misses ({hit_rate:.1f}% hit rate)"
            )

    def _log_stats(self) -> None:
        """Log pipeline statistics."""
        logger.info("=" * 60)
        logger.info("Pipeline Statistics:")
        logger.info(f"  Articles Processed: {self.stats.articles_processed:,}")
        logger.info(f"  Articles Failed: {self.stats.articles_failed:,}")
        logger.info(f"  Articles Skipped: {self.stats.articles_skipped:,}")
        logger.info(f"  Chunks Created: {self.stats.chunks_created:,}")
        logger.info(f"  Chunks Failed: {self.stats.chunks_failed:,}")
        logger.info(f"  Embeddings Generated: {self.stats.embeddings_generated:,}")
        logger.info(f"  Embeddings Cached: {self.stats.embeddings_cached:,}")
        logger.info(f"  Documents Inserted: {self.stats.documents_inserted:,}")

        # Log final cache stats if using cached embeddings
        if isinstance(self.embedding_gen, CachedEmbeddingGenerator):
            cache_stats = self.embedding_gen.get_cache_stats()
            total = cache_stats["cache_hits"] + cache_stats["cache_misses"]
            hit_rate = (cache_stats["cache_hits"] / total * 100) if total > 0 else 0
            logger.info(f"  Cache Hit Rate: {hit_rate:.1f}%")

        if self.stats.articles_processed > 0:
            avg_chunks = self.stats.chunks_created / self.stats.articles_processed
            logger.info(f"  Avg Chunks/Article: {avg_chunks:.2f}")

        if isinstance(self.embedding_gen, CachedEmbeddingGenerator):
            cache_stats = self.embedding_gen.get_cache_stats()
            logger.info(f"  Cache Hit Rate: {cache_stats['hit_rate']:.1%}")

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
