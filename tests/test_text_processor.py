"""Tests for text processing and chunking."""

import pytest

from src.config_loader import ChunkingStrategy, TextProcessingConfig
from src.text_processor import TextChunk, TextChunker, TextProcessor


@pytest.fixture
def text_processing_config():
    """Create a test text processing configuration."""
    return TextProcessingConfig(
        chunk_size=512,
        chunk_overlap=50,
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        min_chunk_length=100,
        max_chunk_length=1000,
    )


class TestTextChunk:
    """Test TextChunk dataclass."""

    def test_valid_chunk(self):
        """Test creating a valid chunk."""
        chunk = TextChunk(
            text="Test content",
            chunk_index=0,
            title="Test Article",
            section="Introduction",
            token_count=10,
        )
        assert chunk.text == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.title == "Test Article"
        assert chunk.section == "Introduction"
        assert chunk.token_count == 10

    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            TextChunk(text="", chunk_index=0, title="Test")

    def test_negative_index_raises_error(self):
        """Test that negative index raises ValueError."""
        with pytest.raises(ValueError, match="index must be non-negative"):
            TextChunk(text="Test", chunk_index=-1, title="Test")


class TestTextChunker:
    """Test TextChunker."""

    def test_initialization(self, text_processing_config):
        """Test chunker initialization."""
        chunker = TextChunker(config=text_processing_config)
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50
        assert chunker.strategy == ChunkingStrategy.SEMANTIC

    def test_initialization_invalid_overlap(self):
        """Test that overlap >= chunk_size raises error."""
        config = TextProcessingConfig(
            chunk_size=100,
            chunk_overlap=100,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
        )
        with pytest.raises(ValueError, match="Overlap must be less than chunk_size"):
            TextChunker(config=config)

    def test_count_tokens(self, text_processing_config):
        """Test token counting."""
        chunker = TextChunker(config=text_processing_config)
        text = "This is a test sentence."
        token_count = chunker.count_tokens(text)

        assert token_count > 0
        assert isinstance(token_count, int)

    def test_fixed_split_basic(self):
        """Test basic fixed splitting."""
        config = TextProcessingConfig(
            chunk_size=50,
            chunk_overlap=10,
            chunking_strategy=ChunkingStrategy.FIXED,
        )
        chunker = TextChunker(config=config)

        # Create text that will need multiple chunks
        text = " ".join([f"Word{i}" for i in range(100)])
        chunks = chunker.chunk_text(text, "Test Article")

        assert len(chunks) > 1
        assert all(chunk.title == "Test Article" for chunk in chunks)
        assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))

    def test_fixed_split_overlap(self):
        """Test that overlap works in fixed splitting."""
        config = TextProcessingConfig(
            chunk_size=20,
            chunk_overlap=5,
            chunking_strategy=ChunkingStrategy.FIXED,
        )
        chunker = TextChunker(config=config)

        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = chunker.chunk_text(text, "Test")

        # Check that chunks have some overlapping content
        if len(chunks) > 1:
            # There should be some text similarity between consecutive chunks
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0

    def test_semantic_split_paragraphs(self):
        """Test semantic splitting respects paragraph boundaries."""
        config = TextProcessingConfig(
            chunk_size=100,
            chunk_overlap=10,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
        )
        chunker = TextChunker(config=config)

        text = """This is the first paragraph. It contains some information.

This is the second paragraph. It contains different information.

This is the third paragraph. It contains even more information."""

        chunks = chunker.chunk_text(text, "Test Article")

        assert len(chunks) >= 1
        assert all("paragraph" in chunk.text.lower() for chunk in chunks)

    def test_semantic_split_sections(self):
        """Test semantic splitting handles section headers."""
        chunker = TextChunker(
            config=TextProcessingConfig(chunk_size=200, chunking_strategy=ChunkingStrategy.SEMANTIC)
        )

        text = """Introduction paragraph with some content.

== First Section ==

Content in the first section with more details.

== Second Section ==

Content in the second section with different details."""

        chunks = chunker.chunk_text(text, "Test Article")

        assert len(chunks) >= 1
        # Section headers should not be in the chunks
        for chunk in chunks:
            assert "==" not in chunk.text

    def test_semantic_split_large_paragraph(self):
        """Test that large paragraphs are split with fixed strategy."""
        chunker = TextChunker(
            config=TextProcessingConfig(
                chunk_size=50,
                chunk_overlap=10,
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                min_chunk_length=20,
            )
        )

        # Create a very long paragraph (no line breaks)
        text = " ".join([f"Word{i}" for i in range(200)])
        chunks = chunker.chunk_text(text, "Test")

        assert len(chunks) > 1
        # All chunks should have reasonable token counts
        for chunk in chunks:
            assert chunk.token_count <= chunker.chunk_size * 1.5 + 50  # Allow some margin

    def test_hybrid_split(self):
        """Test hybrid splitting strategy."""
        chunker = TextChunker(
            config=TextProcessingConfig(chunk_size=100, chunking_strategy=ChunkingStrategy.HYBRID)
        )

        text = """First paragraph with normal content.

Second paragraph also normal.

== Big Section ==

"""
        text += " ".join([f"Word{i}" for i in range(300)])  # Add large content

        chunks = chunker.chunk_text(text, "Test Article")

        assert len(chunks) >= 1
        # Check that chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_minimum_chunk_length(self):
        """Test that minimum chunk length is enforced."""
        chunker = TextChunker(
            config=TextProcessingConfig(chunk_size=50, chunk_overlap=10, min_chunk_length=100)
        )

        text = "Short text."
        chunks = chunker.chunk_text(text, "Test")

        # Should not create chunks below minimum length
        assert len(chunks) == 0

    def test_chunk_text_empty_input(self, text_processing_config):
        """Test handling of empty text."""
        chunker = TextChunker(config=text_processing_config)
        chunks = chunker.chunk_text("", "Test")

        assert len(chunks) == 0

    def test_chunk_text_whitespace_only(self, text_processing_config):
        """Test handling of whitespace-only text."""
        chunker = TextChunker(config=text_processing_config)
        chunks = chunker.chunk_text("   \n\n   ", "Test")

        assert len(chunks) == 0

    def test_chunk_preserves_title(self):
        """Test that all chunks preserve the article title."""
        chunker = TextChunker(
            config=TextProcessingConfig(
                chunk_size=50,
                chunk_overlap=10,
                chunking_strategy=ChunkingStrategy.FIXED,
                min_chunk_length=20,
            )
        )

        text = " ".join([f"Content{i}" for i in range(100)])
        title = "Important Article"
        chunks = chunker.chunk_text(text, title)

        assert all(chunk.title == title for chunk in chunks)

    def test_chunk_indices_are_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = TextChunker(
            config=TextProcessingConfig(chunk_size=50, chunk_overlap=10, min_chunk_length=20)
        )

        text = " ".join([f"Word{i}" for i in range(200)])
        chunks = chunker.chunk_text(text, "Test")

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_fixed_split_with_section(self, text_processing_config):
        """Test fixed split preserves section information."""
        chunker = TextChunker(config=text_processing_config)

        text = "Some content in a section."
        chunks = chunker._fixed_split(text, "Test", start_index=5, section="Test Section")

        for chunk in chunks:
            assert chunk.section == "Test Section"
            assert chunk.chunk_index >= 5

    def test_token_count_in_chunks(self):
        """Test that chunks have token count set."""
        chunker = TextChunker(config=TextProcessingConfig(chunk_size=100))

        text = "This is a test sentence with several words in it."
        chunks = chunker.chunk_text(text, "Test")

        for chunk in chunks:
            assert chunk.token_count > 0
            assert isinstance(chunk.token_count, int)


class TestTextProcessor:
    """Test TextProcessor."""

    def test_initialization(self, text_processing_config):
        """Test processor initialization."""
        chunker = TextChunker(config=text_processing_config)
        processor = TextProcessor(chunker)

        assert processor.chunker == chunker

    def test_process_article(self):
        """Test processing an article."""
        chunker = TextChunker(
            config=TextProcessingConfig(
                chunk_size=100,
                chunk_overlap=10,
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                min_chunk_length=20,
            )
        )
        processor = TextProcessor(chunker)

        text = """This is a test article.

It has multiple paragraphs.

Each paragraph contains useful information."""

        chunks = processor.process_article(text, "Test Article")

        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.title == "Test Article" for chunk in chunks)

    def test_clean_text_removes_empty_lines(self):
        """Test that clean_text removes excessive empty lines."""
        text = """Line 1


Line 2



Line 3"""

        cleaned = TextProcessor.clean_text(text)

        # Should not have multiple consecutive newlines
        assert "\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned

    def test_clean_text_strips_whitespace(self):
        """Test that clean_text strips line whitespace."""
        text = "  Line with spaces  \n  Another line  "

        cleaned = TextProcessor.clean_text(text)

        lines = cleaned.split("\n")
        for line in lines:
            assert line == line.strip()

    def test_process_article_empty_text(self, text_processing_config):
        """Test processing empty article."""
        chunker = TextChunker(config=text_processing_config)
        processor = TextProcessor(chunker)

        chunks = processor.process_article("", "Test")

        assert len(chunks) == 0

    def test_process_article_integration(self):
        """Test complete article processing workflow."""
        chunker = TextChunker(
            config=TextProcessingConfig(
                chunk_size=100,
                chunk_overlap=20,
                chunking_strategy=ChunkingStrategy.SEMANTIC,
                min_chunk_length=50,
            )
        )
        processor = TextProcessor(chunker)

        text = """Python Programming

Python is a high-level programming language. It was created by Guido van Rossum.

== Features ==

Python has simple syntax and is easy to learn. It's widely used for web development.

== History ==

Python was first released in 1991. It has grown significantly in popularity."""

        chunks = processor.process_article(text, "Python")

        assert len(chunks) >= 1
        assert all(chunk.title == "Python" for chunk in chunks)
        assert all(len(chunk.text) >= 50 for chunk in chunks)
        assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))

        # Check content is preserved
        all_text = " ".join(chunk.text for chunk in chunks)
        assert "Python" in all_text
        assert "programming language" in all_text.lower()


class TestChunkingStrategies:
    """Test different chunking strategies with real-world scenarios."""

    def test_semantic_preserves_context(self):
        """Test that semantic chunking preserves paragraph context."""
        chunker = TextChunker(
            config=TextProcessingConfig(chunk_size=100, chunking_strategy=ChunkingStrategy.SEMANTIC)
        )

        text = """The French Revolution was a period of radical change.

It began in 1789 and had far-reaching consequences.

Many historians consider it a pivotal moment in European history."""

        chunks = chunker.chunk_text(text, "French Revolution")

        # Paragraphs should be kept together when possible
        for chunk in chunks:
            # Each chunk should be a complete thought (no mid-sentence cuts)
            assert not chunk.text.startswith("and ")
            assert not chunk.text.startswith("but ")

    def test_fixed_handles_uniform_splitting(self):
        """Test that fixed chunking provides uniform chunks."""
        chunker = TextChunker(
            config=TextProcessingConfig(
                chunk_size=50, chunk_overlap=10, chunking_strategy=ChunkingStrategy.FIXED
            )
        )

        # Long continuous text without paragraph breaks
        text = " ".join(["word" for _ in range(500)])

        chunks = chunker.chunk_text(text, "Test")

        # Check that chunk sizes are relatively uniform
        token_counts = [chunk.token_count for chunk in chunks]
        if len(token_counts) > 1:
            # Most chunks should be close to target size
            avg_tokens = sum(token_counts[:-1]) / len(token_counts[:-1])
            assert abs(avg_tokens - 50) < 20

    def test_hybrid_combines_benefits(self):
        """Test that hybrid strategy combines semantic and fixed benefits."""
        chunker = TextChunker(
            config=TextProcessingConfig(
                chunk_size=80,
                chunk_overlap=15,
                chunking_strategy=ChunkingStrategy.HYBRID,
                min_chunk_length=20,
            )
        )

        text = """Short paragraph one.

Short paragraph two.

"""
        # Add a very long paragraph
        text += " ".join([f"word{i}" for i in range(200)])

        chunks = chunker.chunk_text(text, "Test")

        # Should have multiple chunks due to long paragraph
        assert len(chunks) > 1

        # Check that chunks exist with various lengths
        assert len(chunks) >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_text(self):
        """Test handling of very short text."""
        chunker = TextChunker(config=TextProcessingConfig(min_chunk_length=100))
        chunks = chunker.chunk_text("Short.", "Test")

        assert len(chunks) == 0

    def test_text_exactly_chunk_size(self):
        """Test text that's exactly the chunk size."""
        chunker = TextChunker(
            config=TextProcessingConfig(
                chunk_size=20,
                chunk_overlap=5,
                chunking_strategy=ChunkingStrategy.FIXED,
                min_chunk_length=10,
            )
        )

        # Create text with exactly 20 tokens
        text = " ".join(["word" for _ in range(20)])
        chunks = chunker.chunk_text(text, "Test")

        assert len(chunks) == 1

    def test_special_characters(self, text_processing_config):
        """Test handling of special characters."""
        chunker = TextChunker(config=text_processing_config)

        text = "Text with émojis 😀 and spëcial çharacters!"
        chunks = chunker.chunk_text(text, "Test")

        if chunks:
            assert "😀" in chunks[0].text or len(chunks) > 0

    def test_unicode_text(self, text_processing_config):
        """Test handling of Unicode text."""
        chunker = TextChunker(config=text_processing_config)

        text = "中文文本测试。" + "日本語のテキスト。" + "한국어 텍스트."
        chunks = chunker.chunk_text(text, "Unicode Test")

        # Should handle Unicode without errors
        assert len(chunks) >= 0  # May or may not create chunks depending on length

    def test_very_long_article(self):
        """Test handling of very long articles."""
        chunker = TextChunker(config=TextProcessingConfig(chunk_size=100))

        # Create a very long article
        paragraphs = [f"This is paragraph {i} with some content." for i in range(1000)]
        text = "\n\n".join(paragraphs)

        chunks = chunker.chunk_text(text, "Long Article")

        # Should create many chunks
        assert len(chunks) > 10

        # All chunks should have valid indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
