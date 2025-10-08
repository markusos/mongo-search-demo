"""Embedding generation service using LMStudio."""

import hashlib
import json
import time
from collections.abc import Generator
from pathlib import Path

import lmstudio as lms
from loguru import logger
from tqdm import tqdm

from src.config_loader import EmbeddingConfig


class MockEmbeddingGenerator:
    """Mock embedding generator for testing and benchmarking without LMStudio.

    This generator returns fixed vectors and can be serialized for multiprocessing.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize mock embedding generator.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self._embedding_dim = config.dimension
        self.batch_size = config.batch_size
        logger.info(f"Initialized mock embedding generator with dimension {self._embedding_dim}")

    def embed_single(self, text: str) -> list[float]:
        """Generate a mock embedding for a single text.

        Args:
            text: Input text

        Returns:
            Fixed embedding vector
        """
        if not text or not text.strip():
            return None
        return [0.1] * self._embedding_dim

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> Generator[list[float]]:
        """Generate mock embeddings for a batch of texts.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar (ignored in mock)

        Yields:
            Mock embedding vectors
        """
        for text in texts:
            if not text or not text.strip():
                yield None
            else:
                yield [0.1] * self._embedding_dim

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            Embedding dimension
        """
        return self._embedding_dim

    def validate_embedding(self, embedding: list[float]) -> bool:
        """Validate an embedding vector.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if valid, False otherwise
        """
        if embedding is None:
            return False
        if not isinstance(embedding, list):
            return False
        if len(embedding) != self._embedding_dim:
            return False
        return True


class EmbeddingGenerator:
    """Generate embeddings using LMStudio."""

    def __init__(
        self,
        config: EmbeddingConfig,
    ):
        """
        Initialize the embedding generator.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay

        # Initialize model
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = lms.embedding_model(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Get embedding dimension
        self._embedding_dim = None

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails after retries
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        for attempt in range(self.max_retries):
            try:
                embedding = self.model.embed(text)

                # Validate embedding
                self._validate_embedding(embedding)

                return embedding

            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to generate embedding after {self.max_retries} attempts")
                    raise RuntimeError(f"Embedding generation failed: {e}") from e

        raise RuntimeError("Unexpected error in embed_single")

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> Generator[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Yields:
            Embedding vectors for each text

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        logger.info(f"Generating embeddings for {len(texts)} texts")

        # Process in batches
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings", unit="batch")

        for i in iterator:
            batch = texts[i : i + self.batch_size]

            for text in batch:
                try:
                    embedding = self.embed_single(text)
                    yield embedding
                except Exception as e:
                    logger.error(f"Failed to embed text at index {i}: {e}")
                    # Yield None for failed embeddings
                    yield None

        logger.info(f"Completed embedding generation for {len(texts)} texts")

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension

        Raises:
            RuntimeError: If dimension cannot be determined
        """
        if self._embedding_dim is not None:
            return self._embedding_dim

        try:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed_single("test")
            self._embedding_dim = len(test_embedding)
            logger.info(f"Embedding dimension: {self._embedding_dim}")
            return self._embedding_dim
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {e}")
            raise RuntimeError("Could not determine embedding dimension") from e

    def _validate_embedding(self, embedding: list[float]) -> None:
        """
        Validate an embedding vector.

        Args:
            embedding: Embedding to validate

        Raises:
            ValueError: If embedding is invalid
        """
        if not embedding:
            raise ValueError("Embedding is empty")

        if not isinstance(embedding, list):
            raise ValueError(f"Embedding must be a list, got {type(embedding)}")

        # Check for NaN or Inf values
        if any(not (-1e10 < x < 1e10) for x in embedding):
            raise ValueError("Embedding contains NaN or Inf values")

        # Check dimension consistency
        if self._embedding_dim is not None and len(embedding) != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                f"got {len(embedding)}"
            )


class EmbeddingCache:
    """Cache for storing and retrieving embeddings."""

    def __init__(self, cache_path: str | Path):
        """
        Initialize the embedding cache.

        Args:
            cache_path: Path to cache directory
        """
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Embedding cache initialized at {self.cache_path}")

    def _get_text_hash(self, text: str) -> str:
        """
        Generate a hash for a text.

        Args:
            text: Text to hash

        Returns:
            Hash string
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cache_file(self, text_hash: str) -> Path:
        """
        Get the cache file path for a text hash.

        Args:
            text_hash: Hash of the text

        Returns:
            Path to cache file
        """
        # Use subdirectories based on first 2 chars of hash to avoid too many files
        subdir = self.cache_path / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.json"

    def get(self, text: str) -> list[float] | None:
        """
        Get cached embedding for a text.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None if not found
        """
        text_hash = self._get_text_hash(text)
        cache_file = self._get_cache_file(text_hash)

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    return data["embedding"]
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")
                return None

        return None

    def set(self, text: str, embedding: list[float]) -> None:
        """
        Cache an embedding for a text.

        Args:
            text: Text to cache
            embedding: Embedding to cache
        """
        text_hash = self._get_text_hash(text)
        cache_file = self._get_cache_file(text_hash)

        try:
            with open(cache_file, "w") as f:
                json.dump({"text_hash": text_hash, "embedding": embedding}, f)
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        import shutil

        if self.cache_path.exists():
            shutil.rmtree(self.cache_path)
            self.cache_path.mkdir(parents=True, exist_ok=True)
            logger.info("Embedding cache cleared")

    def get_cache_size(self) -> int:
        """
        Get the number of cached embeddings.

        Returns:
            Number of cached items
        """
        count = 0
        for subdir in self.cache_path.iterdir():
            if subdir.is_dir():
                count += sum(1 for _ in subdir.glob("*.json"))
        return count


class CachedEmbeddingGenerator:
    """Wrapper that adds caching to an embedding generator."""

    def __init__(
        self,
        generator: EmbeddingGenerator,
        cache: EmbeddingCache,
        cache_enabled: bool = True,
        verbose_logging: bool = False,
    ):
        """
        Initialize cached embedding generator.

        Args:
            generator: Base embedding generator
            cache: Cache instance
            cache_enabled: Whether to use cache
            verbose_logging: Whether to log every batch operation
        """
        self.generator = generator
        self.cache = cache
        self.cache_enabled = cache_enabled
        self.verbose_logging = verbose_logging

        self.cache_hits = 0
        self.cache_misses = 0

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.cache_enabled:
            # Try to get from cache
            cached = self.cache.get(text)
            if cached is not None:
                self.cache_hits += 1
                return cached

            self.cache_misses += 1

        # Generate new embedding
        embedding = self.generator.embed_single(text)

        # Cache if enabled
        if self.cache_enabled:
            self.cache.set(text, embedding)

        return embedding

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> Generator[list[float]]:
        """
        Generate embeddings for batch with caching.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Yields:
            Embedding vectors
        """
        if self.verbose_logging:
            logger.info(
                f"Generating embeddings for {len(texts)} texts (cache enabled: {self.cache_enabled})"
            )
        else:
            logger.debug(
                f"Generating embeddings for {len(texts)} texts (cache enabled: {self.cache_enabled})"
            )

        iterator = enumerate(texts)
        if show_progress:
            iterator = tqdm(iterator, total=len(texts), desc="Generating embeddings")

        for i, text in iterator:
            try:
                embedding = self.embed_single(text)
                yield embedding
            except Exception as e:
                logger.error(f"Failed to embed text at index {i}: {e}")
                yield None

        if self.cache_enabled and self.verbose_logging:
            logger.info(
                f"Cache stats - Hits: {self.cache_hits}, Misses: {self.cache_misses}, "
                f"Hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%"
            )
        elif self.cache_enabled:
            logger.debug(
                f"Cache stats - Hits: {self.cache_hits}, Misses: {self.cache_misses}, "
                f"Hit rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%"
            )

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.generator.get_embedding_dimension()

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": self.cache.get_cache_size(),
            "hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0
            ),
        }
