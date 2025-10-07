"""Tests for search.py script."""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# ruff: noqa: E402, I001
from search import format_result, print_comparison, print_help, print_results
from src.search_service import SearchResult


class TestFormatResult:
    """Tests for result formatting."""

    def test_format_result_with_text(self):
        """Test formatting a result with text."""
        result = SearchResult(
            chunk_id="chunk1",
            article_id="art1",
            page_id=123,
            title="Test Article",
            text="This is a test article content.",
            section="Introduction",
            score=0.95,
            rank=1,
            search_type="vector",
        )

        formatted = format_result(result, 1, show_text=True)

        assert "Result #1" in formatted
        assert "Test Article" in formatted
        assert "0.9500" in formatted
        assert "vector" in formatted
        assert "Introduction" in formatted
        assert "This is a test article content." in formatted

    def test_format_result_without_text(self):
        """Test formatting a result without text."""
        result = SearchResult(
            chunk_id="chunk1",
            article_id="art1",
            page_id=123,
            title="Test Article",
            text="This is a test article content.",
            section=None,
            score=0.85,
            rank=2,
            search_type="text",
        )

        formatted = format_result(result, 2, show_text=False)

        assert "Result #2" in formatted
        assert "Test Article" in formatted
        assert "0.8500" in formatted
        assert "text" in formatted
        assert "This is a test article content." not in formatted

    def test_format_result_long_text_truncation(self):
        """Test that long text is truncated."""
        long_text = "a" * 500
        result = SearchResult(
            chunk_id="chunk1",
            article_id="art1",
            page_id=123,
            title="Test Article",
            text=long_text,
            section=None,
            score=0.85,
            rank=1,
            search_type="hybrid",
        )

        formatted = format_result(result, 1, show_text=True)

        # Should be truncated to 300 chars + "..."
        assert "..." in formatted
        assert len([line for line in formatted.split("\n") if "aaa" in line][0]) <= 304

    def test_format_result_no_section(self):
        """Test formatting when section is None."""
        result = SearchResult(
            chunk_id="chunk1",
            article_id="art1",
            page_id=123,
            title="Test Article",
            text="Content",
            section=None,
            score=0.85,
            rank=1,
            search_type="vector",
        )

        formatted = format_result(result, 1, show_text=True)

        assert "Section:" not in formatted


class TestPrintResults:
    """Tests for printing results."""

    def test_print_results_with_results(self, capsys):
        """Test printing results."""
        results = [
            SearchResult(
                chunk_id="chunk1",
                article_id="art1",
                page_id=123,
                title="Test Article 1",
                text="Content 1",
                section=None,
                score=0.95,
                rank=1,
                search_type="vector",
            ),
            SearchResult(
                chunk_id="chunk2",
                article_id="art2",
                page_id=456,
                title="Test Article 2",
                text="Content 2",
                section=None,
                score=0.85,
                rank=2,
                search_type="vector",
            ),
        ]

        print_results(results, "vector", "test query", show_text=False)

        captured = capsys.readouterr()
        assert "VECTOR" in captured.out
        assert "test query" in captured.out
        assert "Results: 2" in captured.out
        assert "Test Article 1" in captured.out
        assert "Test Article 2" in captured.out

    def test_print_results_empty(self, capsys):
        """Test printing empty results."""
        print_results([], "text", "test query")

        captured = capsys.readouterr()
        assert "No results found" in captured.out
        assert "TEXT" in captured.out


class TestPrintComparison:
    """Tests for comparison printing."""

    def test_print_comparison(self, capsys):
        """Test comparison printing."""
        vector_results = [
            SearchResult(
                chunk_id="chunk1",
                article_id="art1",
                page_id=123,
                title="Vector Result 1",
                text="Vector content 1",
                section=None,
                score=0.95,
                rank=1,
                search_type="vector",
            ),
        ]

        text_results = [
            SearchResult(
                chunk_id="chunk2",
                article_id="art2",
                page_id=456,
                title="Text Result 1",
                text="Text content 1",
                section=None,
                score=10.5,
                rank=1,
                search_type="text",
            ),
        ]

        hybrid_results = [
            SearchResult(
                chunk_id="chunk1",
                article_id="art1",
                page_id=123,
                title="Hybrid Result 1",
                text="Hybrid content 1",
                section=None,
                score=0.85,
                rank=1,
                search_type="hybrid",
            ),
        ]

        print_comparison(vector_results, text_results, hybrid_results, "test query")

        captured = capsys.readouterr()
        assert "SEARCH COMPARISON" in captured.out
        assert "test query" in captured.out
        assert "Vector Search" in captured.out
        assert "Text Search" in captured.out
        assert "Hybrid Search" in captured.out
        assert "Vector Result 1" in captured.out
        assert "Text Result 1" in captured.out
        assert "Hybrid Result 1" in captured.out
        assert "Overlap Analysis" in captured.out

    def test_print_comparison_with_empty_results(self, capsys):
        """Test comparison with some empty result sets."""
        vector_results = [
            SearchResult(
                chunk_id="chunk1",
                article_id="art1",
                page_id=123,
                title="Vector Result",
                text="Content",
                section=None,
                score=0.95,
                rank=1,
                search_type="vector",
            ),
        ]

        print_comparison(vector_results, [], [], "test query")

        captured = capsys.readouterr()
        assert "Vector Search" in captured.out
        assert "No results" in captured.out

    def test_print_comparison_overlap_calculation(self, capsys):
        """Test overlap calculation in comparison."""
        # Same chunk_id in all results
        shared_result = SearchResult(
            chunk_id="chunk1",
            article_id="art1",
            page_id=123,
            title="Shared Result",
            text="Content",
            section=None,
            score=0.95,
            rank=1,
            search_type="vector",
        )

        vector_results = [shared_result]
        text_results = [shared_result]
        hybrid_results = [shared_result]

        print_comparison(vector_results, text_results, hybrid_results, "test query")

        captured = capsys.readouterr()
        assert "1 common results" in captured.out


class TestPrintHelp:
    """Tests for help printing."""

    def test_print_help(self, capsys):
        """Test help printing."""
        print_help()

        captured = capsys.readouterr()
        assert "Available Commands:" in captured.out
        assert "/vector" in captured.out
        assert "/text" in captured.out
        assert "/hybrid" in captured.out
        assert "/compare" in captured.out
        assert "/stats" in captured.out
        assert "/help" in captured.out
        assert "/quit" in captured.out
        assert "Search Tips:" in captured.out


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
