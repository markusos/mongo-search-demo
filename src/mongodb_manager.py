"""MongoDB manager for Wikipedia knowledge base."""

from datetime import UTC, datetime
from typing import Any

from loguru import logger
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError, ConnectionFailure

from src.config_loader import MongoDBConfig


class MongoDBManager:
    """Manage MongoDB operations for Wikipedia knowledge base."""

    def __init__(self, config: MongoDBConfig):
        """Initialize MongoDB connection.

        Args:
            config: MongoDB configuration

        Raises:
            ConnectionFailure: If unable to connect to MongoDB
        """
        self.config = config
        self.connection_string = config.uri
        self.database_name = config.database
        self.articles_collection_name = config.collections.articles
        self.chunks_collection_name = config.collections.chunks

        try:
            self.client: MongoClient = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
            )
            # Test connection
            self.client.admin.command("ping")
            logger.info(f"Connected to MongoDB database: {self.database_name}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

        self.db: Database = self.client[self.database_name]
        self.articles_collection: Collection = self.db[self.articles_collection_name]
        self.chunks_collection: Collection = self.db[self.chunks_collection_name]

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def is_connected(self) -> bool:
        """Check if connected to MongoDB.

        Returns:
            True if connected, False otherwise
        """
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def setup_collections(self) -> None:
        """Create collections and indexes."""
        logger.info("Setting up collections and indexes...")

        # Create articles collection if it doesn't exist
        if self.articles_collection_name not in self.db.list_collection_names():
            self.db.create_collection(self.articles_collection_name)
            logger.info(f"Created collection: {self.articles_collection_name}")

        # Create chunks collection if it doesn't exist
        if self.chunks_collection_name not in self.db.list_collection_names():
            self.db.create_collection(self.chunks_collection_name)
            logger.info(f"Created collection: {self.chunks_collection_name}")

        self._create_articles_indexes()
        self._create_chunks_indexes()

        # Create search indexes for MongoDB 8.2+
        self.create_vector_search_index()
        self.create_text_search_index()

        logger.info("Collections and indexes setup complete")

    def _create_articles_indexes(self) -> None:
        """Create indexes for articles collection."""
        # Index on page_id for fast lookups
        self.articles_collection.create_index(
            [("page_id", ASCENDING)],
            unique=True,
            name="page_id_unique",
        )

        # Index on title for text search
        self.articles_collection.create_index(
            [("title", "text")],
            name="title_text_search",
        )

        # Index on created_at for sorting
        self.articles_collection.create_index(
            [("created_at", DESCENDING)],
            name="created_at_desc",
        )

        logger.info("Created indexes for articles collection")

    def _create_chunks_indexes(self) -> None:
        """Create indexes for chunks collection."""
        # Index on page_id for filtering chunks by article
        self.chunks_collection.create_index(
            [("page_id", ASCENDING)],
            name="page_id_asc",
        )

        # Compound index on page_id and chunk_index
        self.chunks_collection.create_index(
            [("page_id", ASCENDING), ("chunk_index", ASCENDING)],
            name="page_id_chunk_index",
        )

        # Text index on content for full-text search
        self.chunks_collection.create_index(
            [("content", "text")],
            name="content_text_search",
        )

        logger.info("Created indexes for chunks collection")

    def create_vector_search_index(
        self,
        index_name: str = "vector_index",
        vector_field: str = "embedding",
        num_dimensions: int = 768,
        similarity: str = "cosine",
    ) -> None:
        """Create Vector Search index (MongoDB 8.2+ Community Edition).

        Args:
            index_name: Name of the vector search index
            vector_field: Field containing the embedding vector
            num_dimensions: Number of dimensions in the embedding
            similarity: Similarity metric (cosine, euclidean, or dotProduct)
        """
        try:
            # Check if index already exists
            existing_indexes = list(self.chunks_collection.aggregate([{"$listSearchIndexes": {}}]))

            if any(idx.get("name") == index_name for idx in existing_indexes):
                logger.info(f"Vector search index '{index_name}' already exists")
                return

            # Create the vector search index using MongoDB 8.2+ syntax
            # db.collection.createSearchIndex(name, type, definition)
            definition = {
                "fields": [
                    {
                        "type": "vector",
                        "path": vector_field,
                        "numDimensions": num_dimensions,
                        "similarity": similarity,
                    },
                    {
                        "type": "filter",
                        "path": "page_id",
                    },
                ]
            }

            # PyMongo's create_search_index takes a model dict
            from pymongo.operations import SearchIndexModel

            model = SearchIndexModel(definition=definition, name=index_name, type="vectorSearch")

            self.chunks_collection.create_search_indexes([model])

            logger.info(
                f"Created vector search index '{index_name}' on {self.chunks_collection_name}"
            )
            logger.info(f"  - Vector field: {vector_field}")
            logger.info(f"  - Dimensions: {num_dimensions}")
            logger.info(f"  - Similarity: {similarity}")

            # Return the config for testing purposes
            return {"name": index_name, "type": "vectorSearch", "definition": definition}

        except Exception as e:
            logger.warning(f"Failed to create vector search index: {e}")
            logger.warning("Vector search may not work properly without this index")
            return None

    def create_text_search_index(
        self,
        index_name: str = "text_search_index",
    ) -> None:
        """Create Text Search index (MongoDB 8.2+ Community Edition).

        Args:
            index_name: Name of the text search index
        """
        try:
            # Check if index already exists
            existing_indexes = list(self.chunks_collection.aggregate([{"$listSearchIndexes": {}}]))

            if any(idx.get("name") == index_name for idx in existing_indexes):
                logger.info(f"Text search index '{index_name}' already exists")
                return

            # Create text search index with dynamic mapping
            definition = {"mappings": {"dynamic": True}}

            # PyMongo's create_search_index for text search
            from pymongo.operations import SearchIndexModel

            model = SearchIndexModel(definition=definition, name=index_name, type="search")

            self.chunks_collection.create_search_indexes([model])

            logger.info(
                f"Created text search index '{index_name}' on {self.chunks_collection_name}"
            )
            logger.info("  - Dynamic field mapping enabled")

        except Exception as e:
            logger.warning(f"Failed to create text search index: {e}")
            logger.warning("Text search may not work properly without this index")

    def insert_article(self, article: dict[str, Any]) -> str:
        """Insert or update a single article using upsert.

        Args:
            article: Article document to insert/update

        Returns:
            Inserted or updated document ID

        Raises:
            OperationFailure: If operation fails
        """
        page_id = article.get("page_id")
        if not page_id:
            raise ValueError("Article must have a page_id")

        if "created_at" not in article:
            article["created_at"] = datetime.now(UTC)

        # Use replace_one with upsert to handle duplicates efficiently
        # This replaces the entire document if it exists, or inserts if new
        article["updated_at"] = datetime.now(UTC)
        result = self.articles_collection.replace_one({"page_id": page_id}, article, upsert=True)
        return str(result.upserted_id) if result.upserted_id else str(page_id)

    def insert_articles_bulk(
        self,
        articles: list[dict[str, Any]],
        ordered: bool = False,
    ) -> dict[str, Any]:
        """Insert multiple articles in bulk.

        Args:
            articles: List of article documents
            ordered: Whether to perform ordered inserts

        Returns:
            Dictionary with insert statistics

        Raises:
            BulkWriteError: If bulk insert fails
        """
        if not articles:
            return {"inserted_count": 0, "errors": []}

        # Add created_at to all articles
        now = datetime.now(UTC)
        for article in articles:
            article["created_at"] = now

        try:
            result = self.articles_collection.insert_many(articles, ordered=ordered)
            return {
                "inserted_count": len(result.inserted_ids),
                "errors": [],
            }
        except BulkWriteError as e:
            write_errors = e.details.get("writeErrors", [])
            return {
                "inserted_count": e.details.get("nInserted", 0),
                "errors": write_errors,
            }

    def insert_chunk(self, chunk: dict[str, Any]) -> str:
        """Insert a single chunk.

        Args:
            chunk: Chunk document to insert

        Returns:
            Inserted document ID

        Raises:
            OperationFailure: If insert fails
        """
        chunk["created_at"] = datetime.now(UTC)
        result = self.chunks_collection.insert_one(chunk)
        return str(result.inserted_id)

    def insert_chunks_bulk(
        self,
        chunks: list[dict[str, Any]],
        ordered: bool = False,
        replace_existing: bool = True,
    ) -> dict[str, Any]:
        """Insert multiple chunks in bulk, optionally replacing existing chunks.

        Args:
            chunks: List of chunk documents
            ordered: Whether to perform ordered inserts
            replace_existing: If True, delete existing chunks for the page_id before inserting

        Returns:
            Dictionary with insert statistics

        Raises:
            BulkWriteError: If bulk insert fails
        """
        if not chunks:
            return {"inserted_count": 0, "errors": []}

        # If replacing, delete existing chunks for this page_id
        if replace_existing and chunks:
            page_id = chunks[0].get("page_id")
            if page_id:
                # Delete existing chunks for this article in one operation
                self.chunks_collection.delete_many({"page_id": page_id})

        # Add created_at to all chunks (only if not present)
        now = datetime.now(UTC)
        for chunk in chunks:
            if "created_at" not in chunk:
                chunk["created_at"] = now

        try:
            result = self.chunks_collection.insert_many(chunks, ordered=ordered)
            return {
                "inserted_count": len(result.inserted_ids),
                "errors": [],
            }
        except BulkWriteError as e:
            write_errors = e.details.get("writeErrors", [])
            return {
                "inserted_count": e.details.get("nInserted", 0),
                "errors": write_errors,
            }

    def get_article_by_page_id(self, page_id: int) -> dict[str, Any] | None:
        """Get article by page ID.

        Args:
            page_id: Wikipedia page ID

        Returns:
            Article document or None if not found
        """
        return self.articles_collection.find_one({"page_id": page_id})

    def get_chunks_by_page_id(
        self,
        page_id: int,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get all chunks for an article.

        Args:
            page_id: Wikipedia page ID
            limit: Maximum number of chunks to return

        Returns:
            List of chunk documents
        """
        query = {"page_id": page_id}
        cursor = self.chunks_collection.find(query).sort("chunk_index", ASCENDING)

        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)

    def article_exists(self, page_id: int) -> bool:
        """Check if an article exists.

        Args:
            page_id: Wikipedia page ID

        Returns:
            True if article exists, False otherwise
        """
        return self.articles_collection.count_documents({"page_id": page_id}, limit=1) > 0

    def delete_article(self, page_id: int) -> int:
        """Delete an article and its chunks.

        Args:
            page_id: Wikipedia page ID

        Returns:
            Number of documents deleted
        """
        articles_deleted = self.articles_collection.delete_one({"page_id": page_id}).deleted_count
        chunks_deleted = self.chunks_collection.delete_many({"page_id": page_id}).deleted_count

        total_deleted = articles_deleted + chunks_deleted
        logger.info(f"Deleted {articles_deleted} article(s) and {chunks_deleted} chunk(s)")

        return total_deleted

    def delete_all_articles(self) -> int:
        """Delete all articles and chunks.

        Returns:
            Total number of documents deleted
        """
        articles_deleted = self.articles_collection.delete_many({}).deleted_count
        chunks_deleted = self.chunks_collection.delete_many({}).deleted_count

        total_deleted = articles_deleted + chunks_deleted
        logger.info(
            f"Deleted all data: {articles_deleted} article(s) and {chunks_deleted} chunk(s)"
        )

        return total_deleted

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collections.

        Returns:
            Dictionary with collection statistics
        """
        articles_count = self.articles_collection.count_documents({})
        chunks_count = self.chunks_collection.count_documents({})

        # Get sample article to calculate average chunks per article
        avg_chunks = chunks_count / articles_count if articles_count > 0 else 0

        # Get collection sizes
        articles_stats = self.db.command("collStats", self.articles_collection_name)
        chunks_stats = self.db.command("collStats", self.chunks_collection_name)

        return {
            "articles": {
                "count": articles_count,
                "size_bytes": articles_stats.get("size", 0),
                "avg_size_bytes": (
                    articles_stats.get("size", 0) / articles_count if articles_count > 0 else 0
                ),
            },
            "chunks": {
                "count": chunks_count,
                "size_bytes": chunks_stats.get("size", 0),
                "avg_size_bytes": (
                    chunks_stats.get("size", 0) / chunks_count if chunks_count > 0 else 0
                ),
            },
            "avg_chunks_per_article": avg_chunks,
            "total_size_bytes": articles_stats.get("size", 0) + chunks_stats.get("size", 0),
        }

    def get_indexes_info(self) -> dict[str, list[dict[str, Any]]]:
        """Get information about all indexes.

        Returns:
            Dictionary with index information for each collection
        """
        return {
            "articles": list(self.articles_collection.list_indexes()),
            "chunks": list(self.chunks_collection.list_indexes()),
        }
