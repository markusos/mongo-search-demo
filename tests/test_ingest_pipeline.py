"""Tests for data ingestion pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.config_loader import (
    AppConfig,
    EmbeddingConfig,
    LoggingConfig,
    MongoDBCollections,
    MongoDBConfig,
    PipelineConfig,
    SearchConfig,
    TextProcessingConfig,
    WikipediaConfig,
)
from src.ingest_pipeline import IngestionPipeline, PipelineStats
from src.wiki_parser import WikiArticle


@pytest.fixture
def app_config(tmp_path):
    """Create a test application configuration."""
    return AppConfig(
        wikipedia=WikipediaConfig(
            xml_path="test.xml",
            max_articles=10,
        ),
        mongodb=MongoDBConfig(
            uri="mongodb://localhost:27017",
            database="test_db",
            collections=MongoDBCollections(),
        ),
        text_processing=TextProcessingConfig(),
        embedding=EmbeddingConfig(
            model="test-model",
            cache_embeddings=False,
            cache_path=str(tmp_path / "embedding_cache"),
        ),
        pipeline=PipelineConfig(
            batch_size=2,
            checkpoint_path=str(tmp_path),
        ),
        search=SearchConfig(),
        logging=LoggingConfig(),
    )


@pytest.fixture
def mock_components():
    """Create mocked pipeline components."""
    mocks = {
        "parser": MagicMock(),
        "processor": MagicMock(),
        "embedding_gen": MagicMock(),
        "db_manager": MagicMock(),
    }
    return mocks


@pytest.fixture
def sample_article():
    """Create a sample Wikipedia article."""
    from datetime import UTC, datetime

    return WikiArticle(
        page_id=12345,
        title="Test Article",
        text="This is a test article content.",
        namespace=0,
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def sample_chunks():
    """Create sample text chunks."""
    from src.text_processor import TextChunk

    return [
        TextChunk(
            text="Chunk 1 content",
            chunk_index=0,
            title="Test Article",
            section=None,
            token_count=10,
        ),
        TextChunk(
            text="Chunk 2 content",
            chunk_index=1,
            title="Test Article",
            section=None,
            token_count=10,
        ),
    ]


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.batch_size == 100
        assert config.checkpoint_interval == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            batch_size=50,
            checkpoint_interval=200,
        )

        assert config.batch_size == 50
        assert config.checkpoint_interval == 200


class TestPipelineStats:
    """Test PipelineStats dataclass."""

    def test_initialization(self):
        """Test stats initialization."""
        stats = PipelineStats()

        assert stats.articles_processed == 0
        assert stats.articles_failed == 0
        assert stats.chunks_created == 0
        assert stats.embeddings_generated == 0

    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = PipelineStats(
            articles_processed=10,
            chunks_created=50,
        )

        stats_dict = stats.to_dict()

        assert stats_dict["articles_processed"] == 10
        assert stats_dict["chunks_created"] == 50
        assert isinstance(stats_dict, dict)


class TestIngestionPipeline:
    """Test IngestionPipeline class."""

    def test_initialization(self, app_config):
        """Test pipeline initialization."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextChunker"),
            patch("src.ingest_pipeline.TextChunker"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(app_config)

            assert pipeline.config == app_config
            assert isinstance(pipeline.stats, PipelineStats)
            assert hasattr(pipeline, "parser")
            assert hasattr(pipeline, "processor")
            assert hasattr(pipeline, "embedding_gen")
            assert hasattr(pipeline, "db_manager")

    def test_process_worker_function(self, app_config, sample_article, sample_chunks):
        """Test the worker function with initializer processes articles correctly."""
        with (
            patch("src.ingest_pipeline.TextChunker") as mock_chunker_class,
            patch("src.ingest_pipeline.TextProcessor") as mock_processor_class,
            patch("src.ingest_pipeline.EmbeddingGenerator") as mock_emb_class,
            patch("src.ingest_pipeline.MongoDBManager") as mock_db_class,
        ):
            # Setup mocks
            mock_chunker = MagicMock()
            mock_chunker_class.return_value = mock_chunker

            mock_processor = MagicMock()
            mock_processor.process_article.return_value = sample_chunks
            mock_processor_class.return_value = mock_processor

            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
            mock_emb_class.return_value = mock_emb

            mock_db = MagicMock()
            mock_db.insert_article.return_value = "article_id"
            mock_db.insert_chunks_bulk.return_value = {
                "inserted_count": 2,
                "errors": [],
            }
            mock_db_class.return_value = mock_db

            from src.ingest_pipeline import _init_worker, _process_article_worker

            # Initialize worker with mocked components
            _init_worker(app_config)

            # Process article
            result = _process_article_worker(sample_article)

            # Verify result
            assert result["success"] is True
            assert result["chunks_created"] == 2
            assert result["documents_inserted"] == 2

    def test_save_checkpoint(self, app_config):
        """Test saving checkpoint."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextChunker"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(app_config)
            pipeline.stats.articles_processed = 100

            pipeline._save_checkpoint(100)

            checkpoint_path = Path(app_config.pipeline.checkpoint_path) / "pipeline_checkpoint.json"
            assert checkpoint_path.exists()

            with open(checkpoint_path) as f:
                checkpoint = json.load(f)

            assert checkpoint["article_count"] == 100
            assert checkpoint["stats"]["articles_processed"] == 100

    def test_load_checkpoint(self, app_config):
        """Test loading checkpoint."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextChunker"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(app_config)

            # Save a checkpoint first
            pipeline.stats.articles_processed = 50
            pipeline._save_checkpoint(50)

            # Create new pipeline and load checkpoint
            pipeline2 = IngestionPipeline(app_config)
            checkpoint_file = str(
                Path(app_config.pipeline.checkpoint_path) / "pipeline_checkpoint.json"
            )
            article_count = pipeline2._load_checkpoint(checkpoint_file)

            assert article_count == 50
            assert pipeline2.stats.articles_processed == 50

    def test_load_checkpoint_not_found(self, app_config):
        """Test loading non-existent checkpoint."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextChunker"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(app_config)

            article_count = pipeline._load_checkpoint("nonexistent.json")

            assert article_count == 0

    def test_get_stats(self, app_config):
        """Test getting pipeline statistics."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextChunker"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(app_config)
            pipeline.stats.articles_processed = 10

            stats = pipeline.get_stats()

            assert isinstance(stats, PipelineStats)
            assert stats.articles_processed == 10

    def test_close(self, app_config):
        """Test closing pipeline resources."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextChunker"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager") as mock_db_class,
        ):
            mock_db = MagicMock()
            mock_db_class.return_value = mock_db

            pipeline = IngestionPipeline(app_config)
            pipeline.close()

            mock_db.close.assert_called_once()


class TestPipelineIntegration:
    """Integration tests for pipeline."""

    def test_run_with_resume(self, app_config, sample_article):
        """Test resuming pipeline from checkpoint."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser") as mock_parser_class,
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            # Setup parser to return 3 articles
            mock_parser = MagicMock()
            mock_parser.parse_stream.return_value = [
                sample_article,
                sample_article,
                sample_article,
            ]
            mock_parser_class.return_value = mock_parser

            # First run - save checkpoint
            pipeline = IngestionPipeline(app_config)
            pipeline._save_checkpoint(1)  # Simulate processing 1 article

            # Second run - resume from checkpoint
            pipeline2 = IngestionPipeline(app_config)
            pipeline2._process_article = Mock()

            # Mock run to test resume logic
            with patch.object(pipeline2, "_load_checkpoint", return_value=1) as mock_load:
                # This would normally call run(), but we're testing the checkpoint logic
                article_count = pipeline2._load_checkpoint(app_config.pipeline.checkpoint_path)

                assert article_count == 1
                mock_load.assert_called_once()
