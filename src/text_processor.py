"""Text processing and chunking utilities for Wikipedia articles."""

from dataclasses import dataclass

import tiktoken
from loguru import logger

from src.config_loader import ChunkingStrategy, TextProcessingConfig


@dataclass
class TextChunk:
    """Represents a chunk of text from an article."""

    text: str
    chunk_index: int
    title: str
    section: str | None = None
    token_count: int = 0

    def __post_init__(self):
        """Validate chunk data."""
        if not self.text:
            raise ValueError("Chunk text cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("Chunk index must be non-negative")


class TextChunker:
    """Handles text chunking with various strategies."""

    def __init__(
        self,
        config: TextProcessingConfig,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize the text chunker.

        Args:
            config: Text processing configuration
            encoding_name: Tiktoken encoding to use for token counting
        """
        self.config = config
        self.chunk_size = config.chunk_size
        self.overlap = config.chunk_overlap
        self.strategy = ChunkingStrategy(config.chunking_strategy)
        self.min_chunk_length = config.min_chunk_length
        self.max_chunk_length = config.max_chunk_length

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load encoding {encoding_name}: {e}")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be less than chunk_size")

    def chunk_text(self, text: str, title: str) -> list[TextChunk]:
        """
        Chunk text according to the configured strategy.

        Args:
            text: Text to chunk
            title: Article title (for context)

        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []

        if self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_split(text, title)
        elif self.strategy == ChunkingStrategy.FIXED:
            return self._fixed_split(text, title)
        elif self.strategy == ChunkingStrategy.HYBRID:
            return self._hybrid_split(text, title)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _semantic_split(self, text: str, title: str) -> list[TextChunk]:
        """
        Split text by semantic boundaries (paragraphs and sections).

        Args:
            text: Text to split
            title: Article title

        Returns:
            List of TextChunk objects
        """
        chunks = []
        current_section = None

        # Split by double newlines (paragraphs) or section headers
        paragraphs = []
        current_para = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append("\n".join(current_para))
                    current_para = []
                continue

            # Check if line is a section header (starts with =)
            if line.startswith("==") and line.endswith("=="):
                # Save previous paragraph
                if current_para:
                    paragraphs.append("\n".join(current_para))
                    current_para = []

                # Extract section name
                current_section = line.strip("= ").strip()

                # Add section header as context (but don't count as separate chunk)
                continue

            current_para.append(line)

        # Add last paragraph
        if current_para:
            paragraphs.append("\n".join(current_para))

        # Now group paragraphs into chunks based on token count
        current_chunk_text = []
        current_tokens = 0
        chunk_index = 0

        for paragraph in paragraphs:
            para_tokens = self.count_tokens(paragraph)

            # If single paragraph exceeds max, split it with fixed strategy
            if para_tokens > self.chunk_size * 1.5:
                # Save current chunk if exists
                if current_chunk_text:
                    chunk_text = "\n\n".join(current_chunk_text)
                    if len(chunk_text) >= self.min_chunk_length:
                        chunks.append(
                            TextChunk(
                                text=chunk_text,
                                chunk_index=chunk_index,
                                title=title,
                                section=current_section,
                                token_count=current_tokens,
                            )
                        )
                        chunk_index += 1
                    current_chunk_text = []
                    current_tokens = 0

                # Split large paragraph with fixed strategy
                large_para_chunks = self._fixed_split(paragraph, title, start_index=chunk_index)
                chunks.extend(large_para_chunks)
                chunk_index += len(large_para_chunks)
                continue

            # Check if adding this paragraph exceeds chunk size
            if current_tokens + para_tokens > self.chunk_size and current_chunk_text:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk_text)
                if len(chunk_text) >= self.min_chunk_length:
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            chunk_index=chunk_index,
                            title=title,
                            section=current_section,
                            token_count=current_tokens,
                        )
                    )
                    chunk_index += 1

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk_text:
                    # Keep last paragraph as overlap
                    current_chunk_text = [current_chunk_text[-1]]
                    current_tokens = self.count_tokens(current_chunk_text[0])
                else:
                    current_chunk_text = []
                    current_tokens = 0

            # Add paragraph to current chunk
            current_chunk_text.append(paragraph)
            current_tokens += para_tokens

        # Add final chunk
        if current_chunk_text:
            chunk_text = "\n\n".join(current_chunk_text)
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        title=title,
                        section=current_section,
                        token_count=current_tokens,
                    )
                )

        return chunks

    def _fixed_split(
        self, text: str, title: str, start_index: int = 0, section: str | None = None
    ) -> list[TextChunk]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Text to split
            title: Article title
            start_index: Starting chunk index
            section: Section name (optional)

        Returns:
            List of TextChunk objects
        """
        chunks = []
        tokens = self.encoding.encode(text)

        chunk_index = start_index
        start = 0

        while start < len(tokens):
            # Get chunk of tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Only add if meets minimum length
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        title=title,
                        section=section,
                        token_count=len(chunk_tokens),
                    )
                )
                chunk_index += 1

            # Move start position with overlap
            start = end - self.overlap if end < len(tokens) else end

        return chunks

    def _hybrid_split(self, text: str, title: str) -> list[TextChunk]:
        """
        Hybrid chunking: semantic split first, then fixed split for large chunks.

        Args:
            text: Text to split
            title: Article title

        Returns:
            List of TextChunk objects
        """
        # First do semantic split
        semantic_chunks = self._semantic_split(text, title)

        # Then check each chunk and split large ones with fixed strategy
        final_chunks = []
        for chunk in semantic_chunks:
            if chunk.token_count > self.chunk_size * 1.5:
                # Split this chunk further with fixed strategy
                sub_chunks = self._fixed_split(
                    chunk.text,
                    chunk.title,
                    start_index=len(final_chunks),
                    section=chunk.section,
                )
                final_chunks.extend(sub_chunks)
            else:
                # Re-index chunk
                chunk.chunk_index = len(final_chunks)
                final_chunks.append(chunk)

        return final_chunks

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Fallback to rough estimate (1 token ≈ 4 chars)
            return len(text) // 4


class TextProcessor:
    """High-level text processor for Wikipedia articles."""

    def __init__(self, chunker: TextChunker):
        """
        Initialize the text processor.

        Args:
            chunker: TextChunker instance to use
        """
        self.chunker = chunker

    def process_article(self, text: str, title: str) -> list[TextChunk]:
        """
        Process an article into chunks.

        Args:
            text: Article text
            title: Article title

        Returns:
            List of TextChunk objects
        """
        # Clean and normalize text
        cleaned_text = self.clean_text(text)

        # Chunk the text
        chunks = self.chunker.chunk_text(cleaned_text, title)

        logger.debug(f"Processed article '{title}' into {len(chunks)} chunks")

        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split("\n")]
        cleaned = "\n".join(line for line in lines if line)

        return cleaned
