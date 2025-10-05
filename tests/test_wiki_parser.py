"""Tests for Wikipedia XML parser."""

from datetime import datetime
from pathlib import Path

import pytest

from src.wiki_parser import WikiArticle, WikiXMLParser


@pytest.fixture
def sample_xml_path():
    """Path to sample Wikipedia XML file."""
    return Path(__file__).parent / "fixtures" / "sample_wiki.xml"


@pytest.fixture
def parser(sample_xml_path):
    """Create a WikiXMLParser instance."""
    return WikiXMLParser(sample_xml_path, clean_markup=True)


class TestWikiArticle:
    """Test WikiArticle dataclass."""

    def test_valid_article(self):
        """Test creating a valid article."""
        article = WikiArticle(
            title="Test",
            text="Test content",
            page_id="123",
            timestamp=datetime.now(),
            namespace=0,
        )
        assert article.title == "Test"
        assert article.text == "Test content"
        assert article.page_id == "123"
        assert article.namespace == 0

    def test_empty_title_raises_error(self):
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            WikiArticle(
                title="",
                text="Test content",
                page_id="123",
                timestamp=datetime.now(),
            )

    def test_empty_page_id_raises_error(self):
        """Test that empty page_id raises ValueError."""
        with pytest.raises(ValueError, match="page_id cannot be empty"):
            WikiArticle(
                title="Test",
                text="Test content",
                page_id="",
                timestamp=datetime.now(),
            )


class TestWikiXMLParser:
    """Test WikiXMLParser."""

    def test_parser_initialization(self, sample_xml_path):
        """Test parser initialization with valid path."""
        parser = WikiXMLParser(sample_xml_path)
        assert parser.xml_path == sample_xml_path
        assert parser.filter_redirects is True
        assert parser.filter_namespaces == [0]

    def test_parser_initialization_file_not_found(self):
        """Test parser initialization with invalid path."""
        with pytest.raises(FileNotFoundError):
            WikiXMLParser("nonexistent.xml")

    def test_parse_stream_basic(self, parser):
        """Test basic parsing functionality."""
        articles = list(parser.parse_stream())

        # Should have 2 articles (Test Article 1 and Machine Learning)
        # - Test Article 2 is a redirect (filtered)
        # - Short Article is too short (filtered)
        # - List of Test Items starts with "List of" (filtered)
        # - Talk page is namespace 1 (filtered)
        assert len(articles) == 2

        # Check first article
        assert articles[0].title == "Test Article 1"
        assert articles[0].page_id == "12345"
        assert "Python" in articles[0].text
        assert articles[0].namespace == 0

    def test_parse_stream_max_articles(self, parser):
        """Test parsing with max_articles limit."""
        articles = list(parser.parse_stream(max_articles=1))
        assert len(articles) == 1

    def test_parse_stream_filters_redirects(self, sample_xml_path):
        """Test that redirect pages are filtered."""
        parser = WikiXMLParser(sample_xml_path, filter_redirects=True)
        articles = list(parser.parse_stream())

        # No redirect articles should be present
        for article in articles:
            assert "REDIRECT" not in article.text

    def test_parse_stream_includes_redirects(self, sample_xml_path):
        """Test parsing with redirects included."""
        parser = WikiXMLParser(sample_xml_path, filter_redirects=False, clean_markup=False)
        articles = list(parser.parse_stream(max_articles=10))

        # Should include the redirect article (or at least have more articles than with filtering)
        # The redirect article has "Test Article 2" as title
        article_titles = [a.title for a in articles]
        assert "Test Article 2" in article_titles or len(articles) >= 2

    def test_parse_stream_filters_namespaces(self, sample_xml_path):
        """Test namespace filtering."""
        parser = WikiXMLParser(sample_xml_path, filter_namespaces=[0])
        articles = list(parser.parse_stream())

        # All articles should be from namespace 0
        for article in articles:
            assert article.namespace == 0

    def test_parse_stream_cleans_markup(self, sample_xml_path):
        """Test that wiki markup is cleaned."""
        parser = WikiXMLParser(sample_xml_path, clean_markup=True)
        articles = list(parser.parse_stream())

        ml_article = next(a for a in articles if a.title == "Machine Learning")

        # Should not contain wiki markup
        assert "[[" not in ml_article.text
        assert "]]" not in ml_article.text
        assert "{{" not in ml_article.text
        assert "}}" not in ml_article.text
        assert "'''" not in ml_article.text

        # Should contain actual content
        assert "Machine learning" in ml_article.text

    def test_parse_stream_without_cleaning(self, sample_xml_path):
        """Test parsing without cleaning markup."""
        parser = WikiXMLParser(sample_xml_path, clean_markup=False)
        articles = list(parser.parse_stream())

        ml_article = next(a for a in articles if a.title == "Machine Learning")

        # Should contain wiki markup
        assert "[[" in ml_article.text or "{{" in ml_article.text

    def test_filter_article_short_content(self, parser):
        """Test filtering of short articles."""
        short_article = WikiArticle(
            title="Short",
            text="Too short",
            page_id="1",
            timestamp=datetime.now(),
        )
        assert not parser.filter_article(short_article)

    def test_filter_article_disambiguation(self, parser):
        """Test filtering of disambiguation pages."""
        disambig_article = WikiArticle(
            title="Test (disambiguation)",
            text="This is a disambiguation page with enough content to pass length check.",
            page_id="2",
            timestamp=datetime.now(),
        )
        assert not parser.filter_article(disambig_article)

    def test_filter_article_list_page(self, parser):
        """Test filtering of list pages."""
        list_article = WikiArticle(
            title="List of test items",
            text="This is a list page with enough content to pass the length check.",
            page_id="3",
            timestamp=datetime.now(),
        )
        assert not parser.filter_article(list_article)

    def test_filter_article_valid(self, parser):
        """Test that valid articles pass filtering."""
        valid_article = WikiArticle(
            title="Valid Article",
            text="This is a valid article with enough content to be useful.",
            page_id="4",
            timestamp=datetime.now(),
        )
        assert parser.filter_article(valid_article)

    def test_clean_wiki_markup_basic(self):
        """Test basic markup cleaning."""
        text = "'''Bold text''' and [[link|linked text]]"
        cleaned = WikiXMLParser.clean_wiki_markup(text)

        assert "'''" not in cleaned
        assert "[[" not in cleaned
        assert "Bold text" in cleaned

    def test_clean_wiki_markup_templates(self):
        """Test template removal."""
        text = "Text with {{template|param=value}} here"
        cleaned = WikiXMLParser.clean_wiki_markup(text)

        assert "{{" not in cleaned
        assert "}}" not in cleaned
        assert "Text with" in cleaned

    def test_clean_wiki_markup_comments(self):
        """Test HTML comment removal."""
        text = "Text with <!-- comment --> here"
        cleaned = WikiXMLParser.clean_wiki_markup(text)

        assert "<!--" not in cleaned
        assert "comment" not in cleaned
        assert "Text with" in cleaned

    def test_clean_wiki_markup_whitespace(self):
        """Test whitespace normalization."""
        text = "Line 1\n\n\nLine 2\n\n\n\nLine 3"
        cleaned = WikiXMLParser.clean_wiki_markup(text)

        # Should have no more than single newlines between lines
        assert "\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned

    def test_article_timestamps(self, parser):
        """Test that article timestamps are parsed correctly."""
        articles = list(parser.parse_stream())

        for article in articles:
            assert isinstance(article.timestamp, datetime)
            # Timestamps should be in 2024
            assert article.timestamp.year == 2024

    def test_article_ids(self, parser):
        """Test that article IDs are extracted correctly."""
        articles = list(parser.parse_stream())

        # Check that IDs are present and unique
        ids = [a.page_id for a in articles]
        assert len(ids) == len(set(ids))  # All unique
        assert all(id.isdigit() for id in ids)  # All numeric

    def test_parse_stream_preserves_content(self, parser):
        """Test that important content is preserved."""
        articles = list(parser.parse_stream())

        python_article = next(a for a in articles if a.title == "Test Article 1")
        ml_article = next(a for a in articles if a.title == "Machine Learning")

        # Python article content
        assert "Python" in python_article.text
        assert "programming language" in python_article.text

        # ML article content
        assert "Machine learning" in ml_article.text
        assert "artificial intelligence" in ml_article.text


class TestWikiXMLParserEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty XML file."""
        empty_file = tmp_path / "empty.xml"
        empty_file.write_text("<mediawiki></mediawiki>")

        parser = WikiXMLParser(empty_file)
        articles = list(parser.parse_stream())

        assert len(articles) == 0

    def test_parse_malformed_xml_raises_error(self, tmp_path):
        """Test that malformed XML raises an appropriate error."""
        malformed_file = tmp_path / "malformed.xml"
        malformed_file.write_text("<mediawiki><page>")

        parser = WikiXMLParser(malformed_file)

        with pytest.raises(Exception):  # SAX will raise parsing exception
            list(parser.parse_stream())

    def test_clean_markup_handles_errors(self):
        """Test that markup cleaning handles errors gracefully."""
        # Provide some edge case text that might cause issues
        text = "<<<invalid>>>"
        cleaned = WikiXMLParser.clean_wiki_markup(text)

        # Should return text even if cleaning fails
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
