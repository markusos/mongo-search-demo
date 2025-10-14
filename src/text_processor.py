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
        # Store paragraphs with their associated section
        paragraphs = []  # List of (paragraph_text, section_name) tuples
        current_para = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append(("\n".join(current_para), current_section))
                    current_para = []
                continue

            # Check if line is a section header (starts with =)
            if line.startswith("==") and line.endswith("=="):
                # Save previous paragraph
                if current_para:
                    paragraphs.append(("\n".join(current_para), current_section))
                    current_para = []

                # Extract section name
                current_section = line.strip("= ").strip()

                # Add section header as context (but don't count as separate chunk)
                continue

            current_para.append(line)

        # Add last paragraph
        if current_para:
            paragraphs.append(("\n".join(current_para), current_section))

        # Now group paragraphs into chunks based on token count
        current_chunk_text = []
        current_tokens = 0
        chunk_index = 0
        current_chunk_section = None

        for paragraph, para_section in paragraphs:
            para_tokens = self.count_tokens(paragraph)

            # Initialize section on first paragraph
            if current_chunk_section is None and not current_chunk_text:
                current_chunk_section = para_section

            # If section changes and current chunk is large enough, save it and start new one
            if para_section != current_chunk_section and current_chunk_text:
                chunk_text = "\n\n".join(current_chunk_text)
                # Only split on section boundary if we have enough content
                if len(chunk_text) >= self.min_chunk_length:
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            chunk_index=chunk_index,
                            title=title,
                            section=current_chunk_section,
                            token_count=current_tokens,
                        )
                    )
                    chunk_index += 1
                    current_chunk_text = []
                    current_tokens = 0
                    current_chunk_section = para_section
                # If chunk is too small, continue adding to it despite section change

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
                                section=current_chunk_section,
                                token_count=current_tokens,
                            )
                        )
                        chunk_index += 1
                    current_chunk_text = []
                    current_tokens = 0

                # Split large paragraph with fixed strategy
                large_para_chunks = self._fixed_split(
                    paragraph, title, start_index=chunk_index, section=para_section
                )
                chunks.extend(large_para_chunks)
                chunk_index += len(large_para_chunks)
                current_chunk_section = para_section
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
                            section=current_chunk_section,
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
                    current_chunk_section = para_section

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
                        section=current_chunk_section,
                        token_count=current_tokens,
                    )
                )

        return chunks

    def _fixed_split(
        self, text: str, title: str, start_index: int = 0, section: str | None = None
    ) -> list[TextChunk]:
        """
        Split text into fixed-size chunks with overlap, respecting semantic boundaries.

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

            # If this is not the last chunk, try to break at semantic boundaries
            adjusted_chunk_tokens = None
            if end < len(tokens):
                adjusted_text = self._find_semantic_break(chunk_text)
                # Only use adjusted text if it's not too short
                if (
                    len(adjusted_text) >= self.min_chunk_length
                    and len(adjusted_text) >= len(chunk_text) * 0.7
                ):
                    chunk_text = adjusted_text
                    # Re-encode to get the actual token count for positioning
                    adjusted_chunk_tokens = self.encoding.encode(chunk_text)

            # Only add if meets minimum length
            if len(chunk_text) >= self.min_chunk_length:
                # Recalculate token count for the actual text used
                actual_token_count = self.count_tokens(chunk_text)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        title=title,
                        section=section,
                        token_count=actual_token_count,
                    )
                )
                chunk_index += 1

            # Move start position with overlap
            # If we adjusted the chunk, use the adjusted length; otherwise use original
            if end < len(tokens):
                if adjusted_chunk_tokens is not None:
                    # Use actual chunk length for next position
                    start = start + len(adjusted_chunk_tokens) - self.overlap
                else:
                    # Use original token boundaries
                    start = end - self.overlap
            else:
                break

        return chunks

    def _find_semantic_break(self, chunk_text: str) -> str:
        """
        Find a good semantic breaking point for a chunk.

        Tries to break at:
        1. Paragraph boundaries (double newline)
        2. Sentence boundaries (. ! ? followed by space/newline)
        3. Clause boundaries (, ; : followed by space)
        4. Word boundaries (whitespace)

        Args:
            chunk_text: The initial chunk text

        Returns:
            Adjusted chunk text with a clean break point
        """
        # Don't try to break very short chunks
        if len(chunk_text) < self.min_chunk_length:
            logger.debug(
                f"_find_semantic_break: chunk too short ({len(chunk_text)}), returning original"
            )
            return chunk_text

        # Start searching from last 20% of chunk, but expand if needed
        search_start = int(len(chunk_text) * 0.8)

        # Priority 1: Try sentence/paragraph boundaries in last 20%
        search_text = chunk_text[search_start:]

        # Paragraph break (double newline)
        para_break = search_text.rfind("\n\n")
        if para_break != -1:
            return chunk_text[: search_start + para_break + 2].rstrip()

        # Sentence break (. ! ? followed by space or newline, optionally with quotes)
        for punct in [
            '. "',
            '." ',
            ". ",
            ".\n",
            "!' ",
            '! "',
            "! ",
            "!\n",
            "?' ",
            '? "',
            "? ",
            "?\n",
        ]:
            sent_break = search_text.rfind(punct)
            if sent_break != -1:
                return chunk_text[: search_start + sent_break + 1].rstrip()

        # Priority 2: If no sentence break in last 20%, search last 40%
        search_start = int(len(chunk_text) * 0.6)
        search_text = chunk_text[search_start:]

        # Try sentence breaks again in wider range
        for punct in [
            '. "',
            '." ',
            ". ",
            ".\n",
            "!' ",
            '! "',
            "! ",
            "!\n",
            "?' ",
            '? "',
            "? ",
            "?\n",
        ]:
            sent_break = search_text.rfind(punct)
            if sent_break != -1:
                return chunk_text[: search_start + sent_break + 1].rstrip()

        # Priority 3: Look for clause boundaries in last 20%
        search_start = int(len(chunk_text) * 0.8)
        search_text = chunk_text[search_start:]

        for punct in [", ", "; ", ": ", ",\n", ";\n", ":\n"]:
            clause_break = search_text.rfind(punct)
            if clause_break != -1:
                return chunk_text[: search_start + clause_break + 1].rstrip()

        # Priority 4: Word boundary - MUST find one to avoid breaking mid-word
        # Search from the end backwards to find ANY space
        # Start from 90% and work backwards
        for search_pct in [0.9, 0.8, 0.7, 0.6, 0.5]:
            search_start = int(len(chunk_text) * search_pct)
            search_text = chunk_text[search_start:]

            space_break = search_text.rfind(" ")
            if space_break != -1:
                # Break right before the last word (at the space position)
                break_pos = search_start + space_break
                return chunk_text[:break_pos]

            newline_break = search_text.rfind("\n")
            if newline_break != -1:
                return chunk_text[: search_start + newline_break]

        # Last resort: find LAST space in the entire chunk
        last_space = chunk_text.rfind(" ")
        if last_space != -1 and last_space > self.min_chunk_length:
            # Break right before the last word
            return chunk_text[:last_space]

        # If absolutely no space found (very rare), return original
        return chunk_text

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
