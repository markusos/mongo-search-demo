"""Wikipedia XML parser for extracting and cleaning articles."""

import bz2
import re
import xml.sax
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger


@dataclass
class WikiArticle:
    """Represents a Wikipedia article."""

    title: str
    text: str
    page_id: str
    timestamp: datetime
    namespace: int = 0

    def __post_init__(self):
        """Validate article data."""
        if not self.title:
            raise ValueError("Article title cannot be empty")
        if not self.page_id:
            raise ValueError("Article page_id cannot be empty")


class WikiXMLHandler(xml.sax.ContentHandler):
    """SAX handler for parsing Wikipedia XML dumps."""

    def __init__(self, filter_redirects: bool = True, filter_namespaces: list[int] = None):
        """
        Initialize the XML handler.

        Args:
            filter_redirects: If True, skip redirect pages
            filter_namespaces: List of namespaces to keep (default: [0] for main articles)
        """
        super().__init__()
        self.filter_redirects = filter_redirects
        self.filter_namespaces = filter_namespaces or [0]

        # Current element tracking
        self.current_element = []
        self.in_page = False

        # Current article data
        self.current_title = ""
        self.current_text = ""
        self.current_id = ""
        self.current_timestamp = ""
        self.current_namespace = 0
        self.is_redirect = False

        # Article callback
        self.article_callback = None
        self.article_count = 0

    def set_article_callback(self, callback):
        """Set callback function to process articles as they're parsed."""
        self.article_callback = callback

    def startElement(self, name, attrs):  # noqa: N802
        """Handle opening XML tags."""
        self.current_element.append(name)

        if name == "page":
            self.in_page = True
            self._reset_article_data()

        # Check for redirect
        if name == "redirect":
            self.is_redirect = True

    def characters(self, content):
        """Handle character data within XML tags."""
        if not self.in_page:
            return

        current = self.current_element[-1] if self.current_element else ""

        if current == "title":
            # Strip whitespace from titles
            if content.strip():
                self.current_title += content
        elif current == "text":
            # Preserve ALL whitespace in article text including newlines!
            self.current_text += content
        elif current == "id":
            # Only get the page ID (first ID tag under page), not revision ID
            # Check if parent is "page" (not "revision")
            if len(self.current_element) >= 2 and self.current_element[-2] == "page":
                if not self.current_id and content.strip():
                    self.current_id += content
        elif current == "timestamp":
            if content.strip():
                self.current_timestamp += content
        elif current == "ns":
            self.current_namespace = int(content) if content.strip() else 0

    def endElement(self, name):  # noqa: N802
        """Handle closing XML tags."""
        if name == "page":
            self.in_page = False
            self._process_article()

        if self.current_element and self.current_element[-1] == name:
            self.current_element.pop()

    def _reset_article_data(self):
        """Reset article data for next article."""
        self.current_title = ""
        self.current_text = ""
        self.current_id = ""
        self.current_timestamp = ""
        self.current_namespace = 0
        self.is_redirect = False

    def _process_article(self):
        """Process the completed article."""
        # Filter out unwanted articles
        if self.is_redirect and self.filter_redirects:
            return

        if self.current_namespace not in self.filter_namespaces:
            return

        if not self.current_title or not self.current_text:
            return

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(self.current_timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now()

        # Create article object
        try:
            article = WikiArticle(
                title=self.current_title.strip(),
                text=self.current_text.strip(),
                page_id=self.current_id.strip(),
                timestamp=timestamp,
                namespace=self.current_namespace,
            )

            if self.article_callback:
                self.article_callback(article)

            self.article_count += 1

        except ValueError as e:
            logger.warning(f"Invalid article data: {e}")


class WikiXMLParser:
    """Parser for Wikipedia XML dumps."""

    def __init__(
        self,
        xml_path: str | Path,
        filter_redirects: bool = True,
        filter_namespaces: list[int] = None,
        clean_markup: bool = True,
    ):
        """
        Initialize the parser.

        Args:
            xml_path: Path to Wikipedia XML dump (can be .bz2 compressed)
            filter_redirects: If True, skip redirect pages
            filter_namespaces: List of namespaces to keep (default: [0])
            clean_markup: If True, clean Wikipedia markup from text
        """
        self.xml_path = Path(xml_path)
        self.filter_redirects = filter_redirects
        self.filter_namespaces = filter_namespaces or [0]
        self.clean_markup = clean_markup

        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")

    def parse_stream(self, max_articles: int | None = None) -> Generator[WikiArticle]:
        """
        Parse XML file and yield WikiArticle objects one at a time.

        Args:
            max_articles: Maximum number of articles to parse (None = all)

        Yields:
            WikiArticle objects
        """
        articles_yielded = 0
        pending_article = deque()  # Use deque for O(1) popleft operations

        def article_callback(article: WikiArticle):
            """Yield articles immediately as they're parsed."""
            nonlocal articles_yielded
            if max_articles and articles_yielded >= max_articles:
                raise StopParsingError()

            # Clean markup if requested
            if self.clean_markup:
                article.text = self.clean_wiki_markup(article.text)

            # Filter by article quality
            if self.filter_article(article):
                pending_article.append(article)
                articles_yielded += 1

        # Create SAX parser
        parser = xml.sax.make_parser()
        handler = WikiXMLHandler(
            filter_redirects=self.filter_redirects,
            filter_namespaces=self.filter_namespaces,
        )
        handler.set_article_callback(article_callback)
        parser.setContentHandler(handler)

        # Open file (handle both compressed and uncompressed)
        try:
            if self.xml_path.suffix == ".bz2":
                logger.info(f"Opening compressed file: {self.xml_path}")
                file_handle = bz2.open(self.xml_path, "rt", encoding="utf-8")
            else:
                logger.info(f"Opening file: {self.xml_path}")
                file_handle = open(self.xml_path, encoding="utf-8")

            with file_handle:
                # Parse in chunks for better performance (64KB chunks)
                chunk_size = 64 * 1024  # 64KB chunks
                try:
                    while True:
                        chunk = file_handle.read(chunk_size)
                        if not chunk:
                            break

                        parser.feed(chunk)

                        # Yield any pending articles
                        while pending_article:
                            yield pending_article.popleft()

                    # Close the parser to ensure all data is processed
                    parser.close()

                except StopParsingError:
                    logger.info(f"Stopped parsing after {articles_yielded} articles")
                finally:
                    # Yield any remaining articles
                    while pending_article:
                        yield pending_article.popleft()

        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            raise

        logger.info(f"Total articles parsed: {handler.article_count}")
        logger.info(f"Articles yielded after filtering: {articles_yielded}")

    def filter_article(self, article: WikiArticle) -> bool:
        """
        Filter articles based on quality criteria.

        Args:
            article: WikiArticle to filter

        Returns:
            True if article should be kept
        """
        # Skip very short articles (stubs) - minimum 50 characters
        if len(article.text) < 50:
            return False

        # Skip disambiguation pages
        if "disambiguation" in article.title.lower():
            return False

        # Skip list pages (often not useful for Q&A)
        if article.title.lower().startswith("list of"):
            return False

        return True

    @staticmethod
    def clean_wiki_markup(text: str) -> str:
        """
        Clean Wikipedia markup from text while preserving paragraph structure.

        Args:
            text: Raw Wikipedia markup text

        Returns:
            Cleaned plain text with preserved paragraph breaks
        """
        # Use regex-based cleaning to preserve whitespace structure
        # mwparserfromhell.strip_code() collapses all whitespace, which breaks paragraph detection
        return WikiXMLParser._fallback_clean_markup(text)

    @staticmethod
    def _fallback_clean_markup(text: str) -> str:
        """
        Fallback regex-based markup cleaning for when mwparserfromhell fails.

        Args:
            text: Raw Wikipedia markup text

        Returns:
            Cleaned plain text
        """
        # Remove templates (anything in double braces)
        # Use a non-greedy approach to handle nested templates better
        text = re.sub(r"\{\{[^}]+\}\}", "", text)
        # Handle leftover template artifacts
        text = re.sub(r"\{\{|\}\}", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # Remove wiki links but keep the display text
        # [[link|display]] -> display
        text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)

        # Remove external links but keep the display text
        # [http://url display] -> display
        text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
        # Remove bare external links
        text = re.sub(r"\[https?://[^\]]+\]", "", text)

        # Remove wiki formatting
        text = re.sub(r"'''([^']+)'''", r"\1", text)  # Bold
        text = re.sub(r"''([^']+)''", r"\1", text)  # Italic

        # Remove file/image references
        text = re.sub(r"\[\[(?:File|Image):[^\]]+\]\]", "", text, flags=re.IGNORECASE)

        # Remove category tags
        text = re.sub(r"\[\[Category:[^\]]+\]\]", "", text, flags=re.IGNORECASE)

        # Remove references
        text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<ref[^>]*/>", "", text, flags=re.IGNORECASE)

        # Clean up whitespace while preserving paragraph structure
        # Strip each line but keep empty lines for paragraph breaks
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse multiple empty lines into double newlines (paragraph breaks)
        text = re.sub(r"\n\n+", "\n\n", text)

        # Remove multiple spaces on the same line (but NOT newlines)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()


class StopParsingError(Exception):
    """Exception to stop parsing early."""

    pass
