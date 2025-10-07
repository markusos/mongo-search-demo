"""Tests for embedding generation service."""

from unittest.mock import MagicMock, patch

import pytest

from src.config_loader import EmbeddingConfig
from src.embedding_service import (
    CachedEmbeddingGenerator,
    EmbeddingCache,
    EmbeddingGenerator,
)


@pytest.fixture
def embedding_config():
    """Create a test embedding configuration."""
    return EmbeddingConfig(
        model="test-model",
        dimension=768,
        batch_size=32,
        lmstudio_url="http://localhost:1234",
        cache_embeddings=True,
        cache_path="./test_cache",
    )


@pytest.fixture
def mock_lmstudio_model():
    """Create a mock LMStudio model."""
    model = MagicMock()
    # Return a 768-dimensional embedding
    model.embed.return_value = [0.1] * 768
    return model


@pytest.fixture
def embedding_generator(mock_lmstudio_model, embedding_config):
    """Create an EmbeddingGenerator with mocked model."""
    with patch("src.embedding_service.lms.embedding_model") as mock_lms:
        mock_lms.return_value = mock_lmstudio_model
        generator = EmbeddingGenerator(config=embedding_config)
        return generator


@pytest.fixture
def embedding_cache(tmp_path):
    """Create a temporary embedding cache."""
    cache_dir = tmp_path / "embedding_cache"
    return EmbeddingCache(cache_dir)


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class."""

    def test_initialization(self, mock_lmstudio_model, embedding_config):
        """Test generator initialization."""
        with patch("src.embedding_service.lms.embedding_model") as mock_lms:
            mock_lms.return_value = mock_lmstudio_model
            embedding_config.model = "test-model"
            embedding_config.batch_size = 16
            embedding_config.max_retries = 5

            generator = EmbeddingGenerator(
                config=embedding_config,
            )

            assert generator.model_name == "test-model"
            assert generator.batch_size == 16
            assert generator.max_retries == 5
            mock_lms.assert_called_once_with("test-model")

    def test_initialization_failure(self, embedding_config):
        """Test generator initialization with model loading failure."""
        with patch("src.embedding_service.lms.embedding_model") as mock_lms:
            mock_lms.side_effect = Exception("Model not found")

            with pytest.raises(Exception, match="Model not found"):
                EmbeddingGenerator(config=embedding_config)

    def test_embed_single_success(self, embedding_generator):
        """Test successful single text embedding."""
        text = "This is a test sentence."
        embedding = embedding_generator.embed_single(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_single_empty_text(self, embedding_generator):
        """Test embedding empty text raises error."""
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedding_generator.embed_single("")

        with pytest.raises(ValueError, match="Cannot embed empty text"):
            embedding_generator.embed_single("   ")

    def test_embed_single_with_retry(self, mock_lmstudio_model, embedding_config):
        """Test retry logic on failure."""
        with patch("src.embedding_service.lms.embedding_model") as mock_lms:
            # Fail first two times, succeed on third
            mock_lmstudio_model.embed.side_effect = [
                Exception("Network error"),
                Exception("Network error"),
                [0.1] * 768,
            ]
            mock_lms.return_value = mock_lmstudio_model

            # Update config for faster retries in test
            embedding_config.max_retries = 3
            embedding_config.retry_delay = 0.01
            generator = EmbeddingGenerator(config=embedding_config)
            embedding = generator.embed_single("test")

            assert len(embedding) == 768
            assert mock_lmstudio_model.embed.call_count == 3

    def test_embed_single_max_retries_exceeded(self, mock_lmstudio_model, embedding_config):
        """Test that max retries raises error."""
        with patch("src.embedding_service.lms.embedding_model") as mock_lms:
            mock_lmstudio_model.embed.side_effect = Exception("Persistent error")
            mock_lms.return_value = mock_lmstudio_model

            # Update config for faster retries in test
            embedding_config.max_retries = 2
            embedding_config.retry_delay = 0.01
            generator = EmbeddingGenerator(config=embedding_config)

            with pytest.raises(RuntimeError, match="Embedding generation failed"):
                generator.embed_single("test")

    def test_embed_batch_success(self, embedding_generator):
        """Test batch embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = list(embedding_generator.embed_batch(texts, show_progress=False))

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings if emb is not None)

    def test_embed_batch_empty_list(self, embedding_generator):
        """Test embedding empty batch raises error."""
        with pytest.raises(ValueError, match="Cannot embed empty list"):
            list(embedding_generator.embed_batch([]))

    def test_embed_batch_with_failures(self, mock_lmstudio_model, embedding_config):
        """Test batch embedding with some failures."""
        with patch("src.embedding_service.lms.embedding_model") as mock_lms:
            # Succeed, fail, succeed
            mock_lmstudio_model.embed.side_effect = [
                [0.1] * 768,
                Exception("Error"),
                [0.2] * 768,
            ]
            mock_lms.return_value = mock_lmstudio_model

            # Update config for faster retries in test
            embedding_config.max_retries = 1
            embedding_config.retry_delay = 0.01
            generator = EmbeddingGenerator(config=embedding_config)
            embeddings = list(generator.embed_batch(["t1", "t2", "t3"], show_progress=False))

            assert len(embeddings) == 3
            assert embeddings[0] is not None
            assert embeddings[1] is None  # Failed
            assert embeddings[2] is not None

    def test_get_embedding_dimension(self, embedding_generator):
        """Test getting embedding dimension."""
        dim = embedding_generator.get_embedding_dimension()
        assert dim == 768

    def test_validate_embedding_success(self, embedding_generator):
        """Test valid embedding passes validation."""
        valid_embedding = [0.1] * 768
        # Should not raise
        embedding_generator._validate_embedding(valid_embedding)

    def test_validate_embedding_empty(self, embedding_generator):
        """Test empty embedding fails validation."""
        with pytest.raises(ValueError, match="Embedding is empty"):
            embedding_generator._validate_embedding([])

    def test_validate_embedding_wrong_type(self, embedding_generator):
        """Test wrong type fails validation."""
        with pytest.raises(ValueError, match="must be a list"):
            embedding_generator._validate_embedding("not a list")

    def test_validate_embedding_nan_values(self, embedding_generator):
        """Test NaN/Inf values fail validation."""
        invalid_embedding = [float("inf")] + [0.1] * 767

        with pytest.raises(ValueError, match="NaN or Inf"):
            embedding_generator._validate_embedding(invalid_embedding)

    def test_validate_embedding_dimension_mismatch(self, embedding_generator):
        """Test dimension mismatch fails validation."""
        # First establish the dimension
        embedding_generator.get_embedding_dimension()

        # Then try with wrong dimension
        wrong_dim_embedding = [0.1] * 512

        with pytest.raises(ValueError, match="dimension mismatch"):
            embedding_generator._validate_embedding(wrong_dim_embedding)


class TestEmbeddingCache:
    """Test EmbeddingCache class."""

    def test_initialization(self, tmp_path):
        """Test cache initialization."""
        cache_dir = tmp_path / "cache"
        cache = EmbeddingCache(cache_dir)

        assert cache.cache_path == cache_dir
        assert cache_dir.exists()

    def test_cache_set_and_get(self, embedding_cache):
        """Test caching and retrieving an embedding."""
        text = "Test text for caching"
        embedding = [0.1, 0.2, 0.3]

        # Set embedding
        embedding_cache.set(text, embedding)

        # Get embedding
        retrieved = embedding_cache.get(text)

        assert retrieved == embedding

    def test_cache_get_nonexistent(self, embedding_cache):
        """Test getting non-existent cache entry."""
        result = embedding_cache.get("nonexistent text")
        assert result is None

    def test_cache_different_texts(self, embedding_cache):
        """Test caching different texts."""
        text1 = "First text"
        text2 = "Second text"
        emb1 = [0.1] * 768
        emb2 = [0.2] * 768

        embedding_cache.set(text1, emb1)
        embedding_cache.set(text2, emb2)

        assert embedding_cache.get(text1) == emb1
        assert embedding_cache.get(text2) == emb2

    def test_cache_clear(self, embedding_cache):
        """Test clearing cache."""
        embedding_cache.set("text1", [0.1] * 768)
        embedding_cache.set("text2", [0.2] * 768)

        assert embedding_cache.get_cache_size() == 2

        embedding_cache.clear()

        assert embedding_cache.get_cache_size() == 0
        assert embedding_cache.get("text1") is None

    def test_cache_size(self, embedding_cache):
        """Test getting cache size."""
        assert embedding_cache.get_cache_size() == 0

        embedding_cache.set("text1", [0.1] * 768)
        assert embedding_cache.get_cache_size() == 1

        embedding_cache.set("text2", [0.2] * 768)
        assert embedding_cache.get_cache_size() == 2

    def test_cache_file_structure(self, embedding_cache):
        """Test that cache files are organized in subdirectories."""
        embedding_cache.set("test", [0.1] * 768)

        # Check that subdirectory was created
        subdirs = list(embedding_cache.cache_path.iterdir())
        assert len(subdirs) > 0
        assert any(d.is_dir() for d in subdirs)

    def test_cache_corrupted_file_handling(self, embedding_cache):
        """Test handling of corrupted cache files."""
        text = "test"
        text_hash = embedding_cache._get_text_hash(text)
        cache_file = embedding_cache._get_cache_file(text_hash)

        # Write invalid JSON
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("invalid json")

        # Should return None for corrupted file
        result = embedding_cache.get(text)
        assert result is None


class TestCachedEmbeddingGenerator:
    """Test CachedEmbeddingGenerator class."""

    def test_initialization(self, embedding_generator, embedding_cache):
        """Test cached generator initialization."""
        cached_gen = CachedEmbeddingGenerator(embedding_generator, embedding_cache)

        assert cached_gen.generator == embedding_generator
        assert cached_gen.cache == embedding_cache
        assert cached_gen.cache_enabled is True
        assert cached_gen.cache_hits == 0
        assert cached_gen.cache_misses == 0

    def test_embed_single_cache_miss(self, embedding_generator, embedding_cache):
        """Test cache miss on first embedding."""
        cached_gen = CachedEmbeddingGenerator(embedding_generator, embedding_cache)

        text = "New text"
        embedding = cached_gen.embed_single(text)

        assert len(embedding) == 768
        assert cached_gen.cache_hits == 0
        assert cached_gen.cache_misses == 1

    def test_embed_single_cache_hit(self, embedding_generator, embedding_cache):
        """Test cache hit on second embedding."""
        cached_gen = CachedEmbeddingGenerator(embedding_generator, embedding_cache)

        text = "Repeated text"

        # First call - cache miss
        emb1 = cached_gen.embed_single(text)

        # Second call - cache hit
        emb2 = cached_gen.embed_single(text)

        assert emb1 == emb2
        assert cached_gen.cache_hits == 1
        assert cached_gen.cache_misses == 1

    def test_embed_single_cache_disabled(self, embedding_generator, embedding_cache):
        """Test behavior with cache disabled."""
        cached_gen = CachedEmbeddingGenerator(
            embedding_generator, embedding_cache, cache_enabled=False
        )

        text = "Test text"

        # Call twice
        cached_gen.embed_single(text)
        cached_gen.embed_single(text)

        # Both should be cache misses (cache disabled)
        assert cached_gen.cache_hits == 0
        assert cached_gen.cache_misses == 0  # Not tracked when disabled

    def test_embed_batch_with_cache(self, embedding_generator, embedding_cache):
        """Test batch embedding with cache."""
        cached_gen = CachedEmbeddingGenerator(embedding_generator, embedding_cache)

        texts = ["text1", "text2", "text1"]  # text1 repeated

        embeddings = list(cached_gen.embed_batch(texts, show_progress=False))

        assert len(embeddings) == 3
        assert embeddings[0] == embeddings[2]  # Same text = same embedding
        assert cached_gen.cache_hits == 1  # One hit for repeated text
        assert cached_gen.cache_misses == 2  # Two misses for unique texts

    def test_get_cache_stats(self, embedding_generator, embedding_cache):
        """Test getting cache statistics."""
        cached_gen = CachedEmbeddingGenerator(embedding_generator, embedding_cache)

        # Generate some cache activity
        cached_gen.embed_single("text1")
        cached_gen.embed_single("text2")
        cached_gen.embed_single("text1")  # Cache hit

        stats = cached_gen.get_cache_stats()

        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["cache_size"] == 2
        assert 0 < stats["hit_rate"] < 1

    def test_batch_with_failures(self, mock_lmstudio_model, embedding_cache, embedding_config):
        """Test batch embedding with some failures."""
        with patch("src.embedding_service.lms.embedding_model") as mock_lms:
            mock_lmstudio_model.embed.side_effect = [
                [0.1] * 768,
                Exception("Error"),
                [0.2] * 768,
            ]
            mock_lms.return_value = mock_lmstudio_model

            # Update config for faster retries in test
            embedding_config.max_retries = 1
            embedding_config.retry_delay = 0.01
            generator = EmbeddingGenerator(config=embedding_config)
            cached_gen = CachedEmbeddingGenerator(generator, embedding_cache)

            embeddings = list(cached_gen.embed_batch(["t1", "t2", "t3"], show_progress=False))

            assert len(embeddings) == 3
            assert embeddings[0] is not None
            assert embeddings[1] is None
            assert embeddings[2] is not None


class TestEmbeddingIntegration:
    """Integration tests for embedding components."""

    def test_full_workflow(self, embedding_generator, embedding_cache):
        """Test complete workflow with caching."""
        cached_gen = CachedEmbeddingGenerator(embedding_generator, embedding_cache)

        # Simulate processing chunks from an article
        chunks = [
            "Introduction to machine learning",
            "Supervised learning algorithms",
            "Unsupervised learning methods",
            "Introduction to machine learning",  # Duplicate
        ]

        embeddings = list(cached_gen.embed_batch(chunks, show_progress=False))

        # All embeddings generated
        assert len(embeddings) == 4
        assert all(e is not None for e in embeddings)

        # Duplicate should use cache
        assert cached_gen.cache_hits == 1
        assert cached_gen.cache_misses == 3

        # Embeddings should be valid
        assert all(len(e) == 768 for e in embeddings)

    def test_dimension_consistency(self, embedding_generator):
        """Test that all embeddings have consistent dimensions."""
        texts = ["short", "a longer text here", "another text with more words"]

        embeddings = [embedding_generator.embed_single(t) for t in texts]

        # All should have same dimension
        dimensions = [len(e) for e in embeddings]
        assert len(set(dimensions)) == 1
        assert dimensions[0] == 768
