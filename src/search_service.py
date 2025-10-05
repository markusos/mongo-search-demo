"""Search service for querying Wikipedia knowledge base."""

from collections import defaultdict
from dataclasses import dataclass

from loguru import logger
from pymongo.collection import Collection

from src.embedding_service import EmbeddingGenerator
from src.mongodb_manager import MongoDBManager


@dataclass(frozen=True)
class SearchResult:
    """Represents a single search result."""

    chunk_id: str
    article_id: str | None
    page_id: int
    title: str
    text: str
    section: str | None
    score: float
    rank: int
    search_type: str  # 'vector', 'text', or 'hybrid'

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "page_id": self.page_id,
            "title": self.title,
            "text": self.text,
            "section": self.section,
            "score": self.score,
            "rank": self.rank,
            "search_type": self.search_type,
        }


class SearchService:
    """Service for searching Wikipedia knowledge base."""

    def __init__(
        self,
        db_manager: MongoDBManager,
        embedding_generator: EmbeddingGenerator,
        vector_index_name: str = "vector_index",
        text_index_name: str = "text_search_index",
    ):
        """Initialize search service.

        Args:
            db_manager: MongoDB manager instance
            embedding_generator: Embedding generator for query vectors
            vector_index_name: Name of the vector search index
            text_index_name: Name of the text search index
        """
        self.db_manager = db_manager
        self.embedding_gen = embedding_generator
        self.vector_index_name = vector_index_name
        self.text_index_name = text_index_name
        self.chunks_collection: Collection = db_manager.chunks_collection

    def vector_search(
        self,
        query: str,
        limit: int = 10,
        num_candidates: int | None = None,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Perform vector similarity search.

        Args:
            query: Search query text
            limit: Maximum number of results
            num_candidates: Number of candidates to consider (default: limit * 10)
            filters: Optional filters to apply

        Returns:
            List of search results sorted by similarity score
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Generate query embedding
        logger.info(f"Generating embedding for query: '{query}'")
        query_embedding = self.embedding_gen.embed_single(query)

        # Set num_candidates if not provided
        if num_candidates is None:
            num_candidates = limit * 10

        # Build vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": num_candidates,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "page_id": 1,
                    "title": 1,
                    "text": 1,
                    "section": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        # Add filters if provided
        if filters:
            pipeline.insert(1, {"$match": filters})

        # Execute search
        logger.debug(f"Executing vector search with limit={limit}")
        results = list(self.chunks_collection.aggregate(pipeline))

        # Convert to SearchResult objects
        search_results = []
        for rank, result in enumerate(results, 1):
            search_results.append(
                SearchResult(
                    chunk_id=str(result["_id"]),
                    article_id=None,  # Not fetched in this query
                    page_id=result["page_id"],
                    title=result["title"],
                    text=result["text"],
                    section=result.get("section"),
                    score=result["score"],
                    rank=rank,
                    search_type="vector",
                )
            )

        logger.info(f"Vector search returned {len(search_results)} results")
        return search_results

    def text_search(
        self,
        query: str,
        limit: int = 10,
        fuzzy: bool = True,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Perform full-text search.

        Args:
            query: Search query text
            limit: Maximum number of results
            fuzzy: Enable fuzzy matching
            filters: Optional filters to apply

        Returns:
            List of search results sorted by text relevance score
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Build text search pipeline
        search_config = {
            "index": self.text_index_name,
            "text": {
                "query": query,
                "path": ["text", "title"],
            },
        }

        # Add fuzzy matching if enabled
        if fuzzy:
            search_config["text"]["fuzzy"] = {"maxEdits": 2}

        pipeline = [
            {"$search": search_config},
            {"$limit": limit},
            {
                "$project": {
                    "_id": 1,
                    "page_id": 1,
                    "title": 1,
                    "text": 1,
                    "section": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]

        # Add filters if provided
        if filters:
            pipeline.insert(1, {"$match": filters})

        # Execute search
        logger.debug(f"Executing text search with limit={limit}")
        results = list(self.chunks_collection.aggregate(pipeline))

        # Convert to SearchResult objects
        search_results = []
        for rank, result in enumerate(results, 1):
            search_results.append(
                SearchResult(
                    chunk_id=str(result["_id"]),
                    article_id=None,
                    page_id=result["page_id"],
                    title=result["title"],
                    text=result["text"],
                    section=result.get("section"),
                    score=result["score"],
                    rank=rank,
                    search_type="text",
                )
            )

        logger.info(f"Text search returned {len(search_results)} results")
        return search_results

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        rrf_k: int = 60,
        use_rrf: bool = True,
    ) -> list[SearchResult]:
        """Perform hybrid search combining vector and text search.

        Args:
            query: Search query text
            limit: Maximum number of results
            vector_weight: Weight for vector search scores (if not using RRF)
            text_weight: Weight for text search scores (if not using RRF)
            rrf_k: Constant for Reciprocal Rank Fusion (default: 60)
            use_rrf: Use RRF algorithm instead of weighted scores

        Returns:
            List of search results sorted by combined score
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Perform both searches
        logger.info(f"Executing hybrid search for: '{query}'")
        vector_results = self.vector_search(query, limit=limit * 2)
        text_results = self.text_search(query, limit=limit * 2)

        if use_rrf:
            # Use Reciprocal Rank Fusion
            combined_results = self._reciprocal_rank_fusion(vector_results, text_results, k=rrf_k)
        else:
            # Use weighted score combination
            combined_results = self._weighted_score_combination(
                vector_results, text_results, vector_weight, text_weight
            )

        # Sort by score and limit
        combined_results.sort(key=lambda x: x.score, reverse=True)
        final_results = combined_results[:limit]

        # Create new results with updated ranks and search type
        ranked_results = []
        for rank, result in enumerate(final_results, 1):
            ranked_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    article_id=result.article_id,
                    page_id=result.page_id,
                    title=result.title,
                    text=result.text,
                    section=result.section,
                    score=result.score,
                    rank=rank,
                    search_type="hybrid",
                )
            )

        logger.info(f"Hybrid search returned {len(ranked_results)} results")
        return ranked_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        text_results: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """Combine results using Reciprocal Rank Fusion (RRF).

        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            k: Constant for RRF calculation (default: 60)

        Returns:
            Combined and deduplicated results
        """
        # Calculate RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        results_map: dict[str, SearchResult] = {}

        # Add vector search scores
        for result in vector_results:
            rrf_score = 1.0 / (k + result.rank)
            rrf_scores[result.chunk_id] += rrf_score
            results_map[result.chunk_id] = result

        # Add text search scores
        for result in text_results:
            rrf_score = 1.0 / (k + result.rank)
            rrf_scores[result.chunk_id] += rrf_score
            if result.chunk_id not in results_map:
                results_map[result.chunk_id] = result

        # Create combined results with RRF scores
        combined = []
        for chunk_id, rrf_score in rrf_scores.items():
            result = results_map[chunk_id]
            # Create new result with RRF score
            combined.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    article_id=result.article_id,
                    page_id=result.page_id,
                    title=result.title,
                    text=result.text,
                    section=result.section,
                    score=rrf_score,
                    rank=0,  # Will be set later
                    search_type="hybrid",
                )
            )

        return combined

    def _weighted_score_combination(
        self,
        vector_results: list[SearchResult],
        text_results: list[SearchResult],
        vector_weight: float,
        text_weight: float,
    ) -> list[SearchResult]:
        """Combine results using weighted score combination.

        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            vector_weight: Weight for vector scores
            text_weight: Weight for text scores

        Returns:
            Combined and deduplicated results
        """
        # Normalize scores to [0, 1]
        vector_normalized = self._normalize_scores(vector_results)
        text_normalized = self._normalize_scores(text_results)

        # Combine scores
        combined_scores: dict[str, float] = {}
        results_map: dict[str, SearchResult] = {}

        for result, norm_score in vector_normalized.items():
            combined_scores[result.chunk_id] = vector_weight * norm_score
            results_map[result.chunk_id] = result

        for result, norm_score in text_normalized.items():
            if result.chunk_id in combined_scores:
                combined_scores[result.chunk_id] += text_weight * norm_score
            else:
                combined_scores[result.chunk_id] = text_weight * norm_score
                results_map[result.chunk_id] = result

        # Create combined results
        combined = []
        for chunk_id, score in combined_scores.items():
            result = results_map[chunk_id]
            combined.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    article_id=result.article_id,
                    page_id=result.page_id,
                    title=result.title,
                    text=result.text,
                    section=result.section,
                    score=score,
                    rank=0,  # Will be set later
                    search_type="hybrid",
                )
            )

        return combined

    def _normalize_scores(self, results: list[SearchResult]) -> dict[SearchResult, float]:
        """Normalize scores to [0, 1] range.

        Args:
            results: List of search results

        Returns:
            Dictionary mapping results to normalized scores
        """
        if not results:
            return {}

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            return {r: 1.0 for r in results}

        normalized = {}
        for result in results:
            norm_score = (result.score - min_score) / (max_score - min_score)
            normalized[result] = norm_score

        return normalized

    def get_article_by_id(self, page_id: int) -> dict | None:
        """Get full article by page ID.

        Args:
            page_id: Wikipedia page ID

        Returns:
            Article document or None if not found
        """
        return self.db_manager.get_article_by_page_id(page_id)

    def get_chunks_by_article(self, page_id: int, limit: int | None = None) -> list[dict]:
        """Get all chunks for an article.

        Args:
            page_id: Wikipedia page ID
            limit: Maximum number of chunks to return

        Returns:
            List of chunk documents
        """
        return self.db_manager.get_chunks_by_page_id(page_id, limit)
