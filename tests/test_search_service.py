"""Tests for search service."""

from unittest.mock import MagicMock

import pytest

from src.search_service import SearchResult, SearchService


@pytest.fixture
def mock_db_manager():
    """Create a mock MongoDB manager."""
    manager = MagicMock()
    manager.chunks_collection = MagicMock()
    return manager


@pytest.fixture
def mock_embedding_gen():
    """Create a mock embedding generator."""
    gen = MagicMock()
    gen.embed_single.return_value = [0.1] * 768
    return gen


@pytest.fixture
def search_service(mock_db_manager, mock_embedding_gen):
    """Create a search service instance."""
    return SearchService(mock_db_manager, mock_embedding_gen)


@pytest.fixture
def sample_vector_results():
    """Create sample vector search results."""
    return [
        {
            "_id": "id1",
            "page_id": 1,
            "title": "Article 1",
            "text": "Content 1",
            "section": None,
            "score": 0.95,
        },
        {
            "_id": "id2",
            "page_id": 2,
            "title": "Article 2",
            "text": "Content 2",
            "section": "Section A",
            "score": 0.85,
        },
    ]


@pytest.fixture
def sample_text_results():
    """Create sample text search results."""
    return [
        {
            "_id": "id3",
            "page_id": 3,
            "title": "Article 3",
            "text": "Content 3",
            "section": None,
            "score": 10.5,
        },
        {
            "_id": "id1",
            "page_id": 1,
            "title": "Article 1",
            "text": "Content 1",
            "section": None,
            "score": 8.2,
        },
    ]


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_initialization(self):
        """Test creating a search result."""
        result = SearchResult(
            chunk_id="123",
            article_id="456",
            page_id=789,
            title="Test Article",
            text="Test content",
            section="Test Section",
            score=0.95,
            rank=1,
            search_type="vector",
        )

        assert result.chunk_id == "123"
        assert result.page_id == 789
        assert result.title == "Test Article"
        assert result.score == 0.95
        assert result.search_type == "vector"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = SearchResult(
            chunk_id="123",
            article_id=None,
            page_id=789,
            title="Test",
            text="Content",
            section=None,
            score=0.5,
            rank=1,
            search_type="text",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["chunk_id"] == "123"
        assert result_dict["score"] == 0.5


class TestSearchService:
    """Test SearchService class."""

    def test_initialization(self, mock_db_manager, mock_embedding_gen):
        """Test service initialization."""
        service = SearchService(
            mock_db_manager,
            mock_embedding_gen,
            vector_index_name="custom_vector",
            text_index_name="custom_text",
        )

        assert service.db_manager == mock_db_manager
        assert service.embedding_gen == mock_embedding_gen
        assert service.vector_index_name == "custom_vector"
        assert service.text_index_name == "custom_text"

    def test_vector_search_empty_query(self, search_service):
        """Test vector search with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_service.vector_search("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_service.vector_search("   ")

    def test_vector_search_success(self, search_service, sample_vector_results):
        """Test successful vector search."""
        search_service.chunks_collection.aggregate.return_value = sample_vector_results

        results = search_service.vector_search("test query", limit=10)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].chunk_id == "id1"
        assert results[0].score == 0.95
        assert results[0].search_type == "vector"
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_vector_search_with_filters(self, search_service, sample_vector_results):
        """Test vector search with filters."""
        search_service.chunks_collection.aggregate.return_value = sample_vector_results

        filters = {"page_id": 1}
        search_service.vector_search("test", limit=5, filters=filters)

        # Verify aggregate was called
        search_service.chunks_collection.aggregate.assert_called_once()
        pipeline = search_service.chunks_collection.aggregate.call_args[0][0]

        # Check that filters were added to pipeline
        assert any("$match" in stage for stage in pipeline)

    def test_vector_search_custom_candidates(self, search_service, sample_vector_results):
        """Test vector search with custom num_candidates."""
        search_service.chunks_collection.aggregate.return_value = sample_vector_results

        search_service.vector_search("test", limit=5, num_candidates=100)

        # Verify the pipeline includes custom num_candidates
        pipeline = search_service.chunks_collection.aggregate.call_args[0][0]
        vector_search_stage = pipeline[0]["$vectorSearch"]
        assert vector_search_stage["numCandidates"] == 100

    def test_text_search_empty_query(self, search_service):
        """Test text search with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_service.text_search("")

    def test_text_search_success(self, search_service, sample_text_results):
        """Test successful text search."""
        search_service.chunks_collection.aggregate.return_value = sample_text_results

        results = search_service.text_search("test query", limit=10)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].chunk_id == "id3"
        assert results[0].search_type == "text"

    def test_text_search_fuzzy_disabled(self, search_service, sample_text_results):
        """Test text search without fuzzy matching."""
        search_service.chunks_collection.aggregate.return_value = sample_text_results

        search_service.text_search("test", fuzzy=False)

        # Verify pipeline doesn't include fuzzy config
        pipeline = search_service.chunks_collection.aggregate.call_args[0][0]
        search_stage = pipeline[0]["$search"]
        assert "fuzzy" not in search_stage.get("text", {})

    def test_text_search_with_filters(self, search_service, sample_text_results):
        """Test text search with filters."""
        search_service.chunks_collection.aggregate.return_value = sample_text_results

        filters = {"title": "Article 1"}
        search_service.text_search("test", filters=filters)

        # Verify filters were added
        pipeline = search_service.chunks_collection.aggregate.call_args[0][0]
        assert any("$match" in stage for stage in pipeline)

    def test_hybrid_search_empty_query(self, search_service):
        """Test hybrid search with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_service.hybrid_search("")

    def test_hybrid_search_rrf(self, search_service, sample_vector_results, sample_text_results):
        """Test hybrid search with RRF."""
        # Mock both search methods
        search_service.chunks_collection.aggregate.side_effect = [
            sample_vector_results,  # vector search
            sample_text_results,  # text search
        ]

        results = search_service.hybrid_search("test query", limit=5, use_rrf=True)

        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.search_type == "hybrid" for r in results)
        # Verify ranks are sequential
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_hybrid_search_weighted(
        self, search_service, sample_vector_results, sample_text_results
    ):
        """Test hybrid search with weighted scores."""
        search_service.chunks_collection.aggregate.side_effect = [
            sample_vector_results,
            sample_text_results,
        ]

        results = search_service.hybrid_search(
            "test query",
            limit=5,
            use_rrf=False,
            vector_weight=0.6,
            text_weight=0.4,
        )

        assert len(results) <= 5
        assert all(r.search_type == "hybrid" for r in results)

    def test_reciprocal_rank_fusion(self, search_service):
        """Test RRF algorithm."""
        vector_results = [
            SearchResult("id1", None, 1, "T1", "C1", None, 0.9, 1, "vector"),
            SearchResult("id2", None, 2, "T2", "C2", None, 0.8, 2, "vector"),
        ]

        text_results = [
            SearchResult("id3", None, 3, "T3", "C3", None, 10.0, 1, "text"),
            SearchResult("id1", None, 1, "T1", "C1", None, 8.0, 2, "text"),
        ]

        combined = search_service._reciprocal_rank_fusion(vector_results, text_results, k=60)

        # id1 appears in both, so should have higher RRF score
        chunk_ids = [r.chunk_id for r in combined]
        assert "id1" in chunk_ids
        assert "id2" in chunk_ids
        assert "id3" in chunk_ids

        # Find id1 result
        id1_result = next(r for r in combined if r.chunk_id == "id1")
        id2_result = next(r for r in combined if r.chunk_id == "id2")

        # id1 should have higher score (appears in both result sets)
        assert id1_result.score > id2_result.score

    def test_weighted_score_combination(self, search_service):
        """Test weighted score combination."""
        vector_results = [
            SearchResult("id1", None, 1, "T1", "C1", None, 0.9, 1, "vector"),
            SearchResult("id2", None, 2, "T2", "C2", None, 0.8, 2, "vector"),
        ]

        text_results = [
            SearchResult("id3", None, 3, "T3", "C3", None, 10.0, 1, "text"),
            SearchResult("id1", None, 1, "T1", "C1", None, 8.0, 2, "text"),
        ]

        combined = search_service._weighted_score_combination(
            vector_results, text_results, vector_weight=0.7, text_weight=0.3
        )

        assert len(combined) == 3
        # All results should have scores in [0, 1] range (normalized and weighted)
        assert all(0 <= r.score <= 1 for r in combined)

    def test_normalize_scores(self, search_service):
        """Test score normalization."""
        results = [
            SearchResult("id1", None, 1, "T1", "C1", None, 10.0, 1, "vector"),
            SearchResult("id2", None, 2, "T2", "C2", None, 5.0, 2, "vector"),
            SearchResult("id3", None, 3, "T3", "C3", None, 0.0, 3, "vector"),
        ]

        normalized = search_service._normalize_scores(results)

        assert len(normalized) == 3
        assert normalized[results[0]] == 1.0  # Max score
        assert normalized[results[2]] == 0.0  # Min score
        assert 0 < normalized[results[1]] < 1  # Middle score

    def test_normalize_scores_empty(self, search_service):
        """Test normalizing empty results."""
        normalized = search_service._normalize_scores([])
        assert normalized == {}

    def test_normalize_scores_same_values(self, search_service):
        """Test normalizing when all scores are the same."""
        results = [
            SearchResult("id1", None, 1, "T1", "C1", None, 5.0, 1, "vector"),
            SearchResult("id2", None, 2, "T2", "C2", None, 5.0, 2, "vector"),
        ]

        normalized = search_service._normalize_scores(results)

        # All should be 1.0 when scores are identical
        assert all(score == 1.0 for score in normalized.values())

    def test_get_article_by_id(self, search_service, mock_db_manager):
        """Test getting article by ID."""
        expected_article = {
            "page_id": 123,
            "title": "Test Article",
            "full_text": "Content",
        }
        mock_db_manager.get_article_by_page_id.return_value = expected_article

        article = search_service.get_article_by_id(123)

        assert article == expected_article
        mock_db_manager.get_article_by_page_id.assert_called_once_with(123)

    def test_get_chunks_by_article(self, search_service, mock_db_manager):
        """Test getting chunks by article."""
        expected_chunks = [
            {"chunk_index": 0, "text": "Chunk 1"},
            {"chunk_index": 1, "text": "Chunk 2"},
        ]
        mock_db_manager.get_chunks_by_page_id.return_value = expected_chunks

        chunks = search_service.get_chunks_by_article(123, limit=10)

        assert chunks == expected_chunks
        mock_db_manager.get_chunks_by_page_id.assert_called_once_with(123, 10)


class TestSearchIntegration:
    """Integration tests for search service."""

    def test_full_search_workflow(self, search_service, sample_vector_results, sample_text_results):
        """Test complete search workflow."""
        # Test vector search
        search_service.chunks_collection.aggregate.return_value = sample_vector_results
        vector_results = search_service.vector_search("neural networks")
        assert len(vector_results) > 0
        assert vector_results[0].search_type == "vector"

        # Test text search
        search_service.chunks_collection.aggregate.return_value = sample_text_results
        text_results = search_service.text_search("neural networks")
        assert len(text_results) > 0
        assert text_results[0].search_type == "text"

        # Test hybrid search
        search_service.chunks_collection.aggregate.side_effect = [
            sample_vector_results,
            sample_text_results,
        ]
        hybrid_results = search_service.hybrid_search("neural networks")
        assert len(hybrid_results) > 0
        assert all(r.search_type == "hybrid" for r in hybrid_results)

    def test_deduplication_in_hybrid_search(
        self, search_service, sample_vector_results, sample_text_results
    ):
        """Test that hybrid search deduplicates results."""
        # Both result sets contain id1
        search_service.chunks_collection.aggregate.side_effect = [
            sample_vector_results,
            sample_text_results,
        ]

        results = search_service.hybrid_search("test")

        # Check no duplicate chunk_ids
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_ranking_consistency(self, search_service, sample_vector_results):
        """Test that rankings are consistent and sequential."""
        search_service.chunks_collection.aggregate.return_value = sample_vector_results

        results = search_service.vector_search("test")

        # Verify ranks are 1, 2, 3, ...
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

        # Verify sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
