"""Configuration loader for the application.

This module provides a centralized way to load configuration from config/config.yaml
and exposes it through a structured Pydantic model.
"""

import os
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    SEMANTIC = "semantic"
    FIXED = "fixed"
    HYBRID = "hybrid"


class WikipediaConfig(BaseModel):
    """Wikipedia parsing configuration."""

    xml_path: str
    max_articles: int | None = None


class MongoDBCollections(BaseModel):
    """MongoDB collection names."""

    articles: str = "wiki_articles"
    chunks: str = "wiki_chunks"


class MongoDBConfig(BaseModel):
    """MongoDB configuration."""

    uri: str
    database: str
    collections: MongoDBCollections = MongoDBCollections()


class TextProcessingConfig(BaseModel):
    """Text processing configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    min_chunk_length: int = 100
    max_chunk_length: int = 1000


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str
    dimension: int = 768
    batch_size: int = 32
    lmstudio_url: str = "http://localhost:1234"
    cache_embeddings: bool = True
    cache_path: str = "./embedding_cache"
    max_retries: int = 3
    retry_delay: float = 1.0


class PipelineConfig(BaseModel):
    """Ingestion pipeline configuration."""

    batch_size: int = 100
    checkpoint_interval: int = 100
    checkpoint_path: str = "./checkpoints"
    log_level: str = "INFO"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    embedding_verbose: bool = False
    cache_stats_interval: int = 100


class SearchConfig(BaseModel):
    """Search configuration."""

    default_limit: int = 10
    vector_candidates_multiplier: int = 10
    hybrid_vector_weight: float = 0.7
    similarity_metric: str = "cosine"


class AppConfig(BaseModel):
    """Application configuration."""

    wikipedia: WikipediaConfig
    mongodb: MongoDBConfig
    text_processing: TextProcessingConfig = TextProcessingConfig()
    embedding: EmbeddingConfig
    pipeline: PipelineConfig = PipelineConfig()
    search: SearchConfig = SearchConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Optional path to config file. If None, uses config/config.yaml
                    relative to project root.

    Returns:
        AppConfig instance with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If config validation fails
    """
    if config_path is None:
        # Default to config/config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Allow environment variable overrides for sensitive values
    if "MONGODB_URI" in os.environ:
        config_dict["mongodb"]["uri"] = os.environ["MONGODB_URI"]

    if "LMSTUDIO_URL" in os.environ:
        config_dict["embedding"]["lmstudio_url"] = os.environ["LMSTUDIO_URL"]

    return AppConfig(**config_dict)


def get_config() -> AppConfig:
    """Get the application configuration.

    This is a convenience function that loads the default config.

    Returns:
        AppConfig instance
    """
    return load_config()
