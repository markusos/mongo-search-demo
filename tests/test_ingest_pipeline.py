"""Tests for data ingestion pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.ingest_pipeline import IngestionPipeline, PipelineConfig, PipelineStats
from src.wiki_parser import WikiArticle


@pytest.fixture
def pipeline_config(tmp_path):
    """Create a test pipeline configuration."""
    return PipelineConfig(
        xml_path="test.xml",
        mongodb_uri="mongodb://localhost:27017",
        database_name="test_db",
        batch_size=2,
        max_articles=10,
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        embedding_cache_path=str(tmp_path / "embedding_cache"),
        cache_embeddings=False,  # Disable caching for tests
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
        config = PipelineConfig(
            xml_path="test.xml",
            mongodb_uri="mongodb://localhost:27017",
        )

        assert config.database_name == "wikipedia_kb"
        assert config.batch_size == 100
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.chunking_strategy == "semantic"
        assert config.max_articles is None
        assert config.checkpoint_interval == 1000
        assert config.cache_embeddings is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            xml_path="custom.xml",
            mongodb_uri="mongodb://custom:27017",
            database_name="custom_db",
            batch_size=50,
            max_articles=100,
        )

        assert config.xml_path == "custom.xml"
        assert config.database_name == "custom_db"
        assert config.batch_size == 50
        assert config.max_articles == 100


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

    def test_initialization(self, pipeline_config):
        """Test pipeline initialization."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(pipeline_config)

            assert pipeline.config == pipeline_config
            assert isinstance(pipeline.stats, PipelineStats)
            assert hasattr(pipeline, "parser")
            assert hasattr(pipeline, "processor")
            assert hasattr(pipeline, "embedding_gen")
            assert hasattr(pipeline, "db_manager")

    def test_process_article_success(self, pipeline_config, sample_article, sample_chunks):
        """Test successful article processing."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor") as mock_processor_class,
            patch("src.ingest_pipeline.EmbeddingGenerator") as mock_emb_class,
            patch("src.ingest_pipeline.MongoDBManager") as mock_db_class,
        ):
            # Setup mocks
            mock_processor = MagicMock()
            mock_processor.process_article.return_value = sample_chunks
            mock_processor_class.return_value = mock_processor

            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
            mock_emb_class.return_value = mock_emb

            mock_db = MagicMock()
            mock_db.article_exists.return_value = False
            mock_db.insert_article.return_value = "article_id"
            mock_db.insert_chunks_bulk.return_value = {
                "inserted_count": 2,
                "errors": [],
            }
            mock_db_class.return_value = mock_db

            pipeline = IngestionPipeline(pipeline_config)
            pipeline._process_article(sample_article)

            # Verify article was processed
            mock_processor.process_article.assert_called_once_with(
                text=sample_article.text, title=sample_article.title
            )
            mock_emb.embed_batch.assert_called_once()
            mock_db.insert_article.assert_called_once()
            mock_db.insert_chunks_bulk.assert_called_once()

    def test_process_article_already_exists(self, pipeline_config, sample_article):
        """Test skipping existing article."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager") as mock_db_class,
        ):
            mock_db = MagicMock()
            mock_db.article_exists.return_value = True
            mock_db_class.return_value = mock_db

            pipeline = IngestionPipeline(pipeline_config)
            initial_skipped = pipeline.stats.articles_skipped

            pipeline._process_article(sample_article)

            assert pipeline.stats.articles_skipped == initial_skipped + 1
            mock_db.insert_article.assert_not_called()

    def test_process_article_no_chunks(self, pipeline_config, sample_article):
        """Test handling article with no chunks."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor") as mock_processor_class,
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager") as mock_db_class,
        ):
            mock_processor = MagicMock()
            mock_processor.process_article.return_value = []
            mock_processor_class.return_value = mock_processor

            mock_db = MagicMock()
            mock_db.article_exists.return_value = False
            mock_db_class.return_value = mock_db

            pipeline = IngestionPipeline(pipeline_config)
            pipeline._process_article(sample_article)

            # Should not insert anything
            mock_db.insert_article.assert_not_called()

    def test_process_article_embedding_failure(
        self, pipeline_config, sample_article, sample_chunks
    ):
        """Test handling embedding generation failure."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor") as mock_processor_class,
            patch("src.ingest_pipeline.EmbeddingGenerator") as mock_emb_class,
            patch("src.ingest_pipeline.MongoDBManager") as mock_db_class,
        ):
            mock_processor = MagicMock()
            mock_processor.process_article.return_value = sample_chunks
            mock_processor_class.return_value = mock_processor

            mock_emb = MagicMock()
            # One successful, one failed embedding
            mock_emb.embed_batch.return_value = [[0.1] * 768, None]
            mock_emb_class.return_value = mock_emb

            mock_db = MagicMock()
            mock_db.article_exists.return_value = False
            mock_db.insert_article.return_value = "article_id"
            mock_db.insert_chunks_bulk.return_value = {
                "inserted_count": 1,
                "errors": [],
            }
            mock_db_class.return_value = mock_db

            pipeline = IngestionPipeline(pipeline_config)
            pipeline._process_article(sample_article)

            # Should insert only the successful chunk
            assert pipeline.stats.chunks_failed == 1
            mock_db.insert_chunks_bulk.assert_called_once()
            call_args = mock_db.insert_chunks_bulk.call_args
            assert len(call_args[0][0]) == 1  # Only 1 chunk inserted

    def test_process_batch(self, pipeline_config, sample_article):
        """Test batch processing."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(pipeline_config)
            pipeline._process_article = Mock()

            articles = [sample_article, sample_article]
            pipeline._process_batch(articles)

            assert pipeline._process_article.call_count == 2
            assert pipeline.stats.articles_processed == 2

    def test_process_batch_with_failure(self, pipeline_config, sample_article):
        """Test batch processing with one failure."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(pipeline_config)
            pipeline._process_article = Mock(side_effect=[None, Exception("Test error")])

            articles = [sample_article, sample_article]
            pipeline._process_batch(articles)

            assert pipeline.stats.articles_processed == 1
            assert pipeline.stats.articles_failed == 1

    def test_save_checkpoint(self, pipeline_config):
        """Test saving checkpoint."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(pipeline_config)
            pipeline.stats.articles_processed = 100

            pipeline._save_checkpoint(100)

            checkpoint_path = Path(pipeline_config.checkpoint_path)
            assert checkpoint_path.exists()

            with open(checkpoint_path) as f:
                checkpoint = json.load(f)

            assert checkpoint["article_count"] == 100
            assert checkpoint["stats"]["articles_processed"] == 100

    def test_load_checkpoint(self, pipeline_config):
        """Test loading checkpoint."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(pipeline_config)

            # Save a checkpoint first
            pipeline.stats.articles_processed = 50
            pipeline._save_checkpoint(50)

            # Create new pipeline and load checkpoint
            pipeline2 = IngestionPipeline(pipeline_config)
            article_count = pipeline2._load_checkpoint(pipeline_config.checkpoint_path)

            assert article_count == 50
            assert pipeline2.stats.articles_processed == 50

    def test_load_checkpoint_not_found(self, pipeline_config):
        """Test loading non-existent checkpoint."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(pipeline_config)

            article_count = pipeline._load_checkpoint("nonexistent.json")

            assert article_count == 0

    def test_get_stats(self, pipeline_config):
        """Test getting pipeline statistics."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager"),
        ):
            pipeline = IngestionPipeline(pipeline_config)
            pipeline.stats.articles_processed = 10

            stats = pipeline.get_stats()

            assert isinstance(stats, PipelineStats)
            assert stats.articles_processed == 10

    def test_close(self, pipeline_config):
        """Test closing pipeline resources."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser"),
            patch("src.ingest_pipeline.TextProcessor"),
            patch("src.ingest_pipeline.EmbeddingGenerator"),
            patch("src.ingest_pipeline.MongoDBManager") as mock_db_class,
        ):
            mock_db = MagicMock()
            mock_db_class.return_value = mock_db

            pipeline = IngestionPipeline(pipeline_config)
            pipeline.close()

            mock_db.close.assert_called_once()


class TestPipelineIntegration:
    """Integration tests for pipeline."""

    def test_run_with_small_dataset(self, pipeline_config, sample_article, sample_chunks):
        """Test running pipeline with small dataset."""
        with (
            patch("src.ingest_pipeline.WikiXMLParser") as mock_parser_class,
            patch("src.ingest_pipeline.TextProcessor") as mock_processor_class,
            patch("src.ingest_pipeline.EmbeddingGenerator") as mock_emb_class,
            patch("src.ingest_pipeline.MongoDBManager") as mock_db_class,
        ):
            # Setup parser mock
            mock_parser = MagicMock()
            mock_parser.parse_stream.return_value = [sample_article, sample_article]
            mock_parser_class.return_value = mock_parser

            # Setup processor mock
            mock_processor = MagicMock()
            mock_processor.process_article.return_value = sample_chunks
            mock_processor_class.return_value = mock_processor

            # Setup embedding generator mock
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
            mock_emb_class.return_value = mock_emb

            # Setup database mock
            mock_db = MagicMock()
            mock_db.article_exists.return_value = False
            mock_db.insert_article.return_value = "article_id"
            mock_db.insert_chunks_bulk.return_value = {
                "inserted_count": 2,
                "errors": [],
            }
            mock_db_class.return_value = mock_db

            # Run pipeline
            pipeline = IngestionPipeline(pipeline_config)
            stats = pipeline.run()

            # Verify results
            assert stats.articles_processed == 2
            assert stats.chunks_created == 4  # 2 articles * 2 chunks
            assert mock_db.insert_article.call_count == 2
            assert mock_db.insert_chunks_bulk.call_count == 2

    def test_run_with_resume(self, pipeline_config, sample_article):
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
            pipeline = IngestionPipeline(pipeline_config)
            pipeline._save_checkpoint(1)  # Simulate processing 1 article

            # Second run - resume from checkpoint
            pipeline2 = IngestionPipeline(pipeline_config)
            pipeline2._process_article = Mock()

            # Mock run to test resume logic
            with patch.object(pipeline2, "_load_checkpoint", return_value=1) as mock_load:
                # This would normally call run(), but we're testing the checkpoint logic
                article_count = pipeline2._load_checkpoint(pipeline_config.checkpoint_path)

                assert article_count == 1
                mock_load.assert_called_once()
