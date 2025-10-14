"""Tests for search TUI."""

from src.search_service import SearchResult
from src.search_tui import SearchApp, SearchResultWidget


class TestSearchResultWidget:
    """Tests for SearchResultWidget."""

    def test_initialization(self):
        """Test widget initialization."""
        result = SearchResult(
            chunk_id="chunk1",
            article_id="art1",
            page_id=123,
            title="Test Article",
            text="This is test content.",
            section="Introduction",
            score=0.95,
            rank=1,
            search_type="vector",
        )

        widget = SearchResultWidget(result, 1)
        assert widget.result == result
        assert widget.index == 1

    def test_render_collapsed(self):
        """Test rendering a compact single-line result."""
        result = SearchResult(
            chunk_id="chunk1",
            article_id="art1",
            page_id=123,
            title="Test Article",
            text="This is test content that is longer than 100 characters. " * 5,
            section="Introduction",
            score=0.95,
            rank=1,
            search_type="vector",
        )

        widget = SearchResultWidget(result, 1)
        rendered = widget.render()

        assert "Test Article" in rendered
        assert "Introduction" in rendered
        assert "0.9500" in rendered
        assert "chunk1" in rendered  # Chunk ID shown
        assert "\n" in rendered  # Two lines: title+score, chunk ID

    def test_render_expanded(self):
        """Test rendering a compact result without section."""
        result = SearchResult(
            chunk_id="chunk1",
            article_id="art1",
            page_id=123,
            title="Test Article",
            text="This is test content.",
            section=None,
            score=0.85,
            rank=2,
            search_type="text",
        )

        widget = SearchResultWidget(result, 2)
        rendered = widget.render()

        assert "Test Article" in rendered
        assert "0.8500" in rendered
        assert "chunk1" in rendered  # Chunk ID shown
        # No preview text in list view
        assert "This is test content." not in rendered


class TestSearchApp:
    """Tests for SearchApp TUI class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock services
        from unittest.mock import Mock

        self.mock_search_service = Mock()
        self.mock_db_manager = Mock()
        self.app = SearchApp(self.mock_search_service, self.mock_db_manager)

    def test_init(self):
        """Test SearchApp initialization."""
        assert self.app.results == []
        assert self.app.last_query == ""
        assert self.app.search_type == "hybrid"
        assert self.app.last_search_time == 0.0
        assert self.app.selected_index == 0
        assert self.app.search_service is not None
        assert self.app.db_manager is not None

    def test_title_and_subtitle(self):
        """Test app has proper title and subtitle."""
        assert self.app.TITLE == "Wikipedia Vector Search"
        assert "tab" in self.app.SUB_TITLE.lower()
        assert "vector" in self.app.SUB_TITLE.lower()

    def test_bindings(self):
        """Test app has necessary key bindings."""
        binding_keys = [b.key for b in self.app.BINDINGS]
        assert "q" in binding_keys
        assert "?" in binding_keys  # Help
        assert "tab" in binding_keys  # Cycle focus
        assert "v" in binding_keys  # Vector search
        assert "t" in binding_keys  # Text search
        assert "h" in binding_keys  # Hybrid search
        assert "c" in binding_keys  # Compare binding
        assert "s" in binding_keys  # Stats

    def test_update_status(self):
        """Test status update method exists."""
        assert hasattr(self.app, "update_status")
        assert callable(self.app.update_status)


class TestIntegration:
    """Integration tests for the demo script."""

    def test_result_ordering(self):
        """Test that results maintain proper ordering."""
        results = [
            SearchResult(
                chunk_id=f"chunk{i}",
                article_id=f"art{i}",
                page_id=i,
                title=f"Article {i}",
                text=f"Content {i}",
                section=None,
                score=1.0 - (i * 0.1),
                rank=i + 1,
                search_type="vector",
            )
            for i in range(5)
        ]

        # Verify ordering is preserved
        for i, result in enumerate(results):
            assert result.rank == i + 1
            assert result.score >= results[i + 1].score if i < len(results) - 1 else True
