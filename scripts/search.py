#!/usr/bin/env -S uv run python
"""Interactive demo for Wikipedia vector search knowledge base.

This script provides an interactive CLI for searching the Wikipedia knowledge base
using vector search, text search, or hybrid search methods.
"""

import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from loguru import logger

from src.config_loader import load_config
from src.embedding_service import EmbeddingGenerator
from src.mongodb_manager import MongoDBManager
from src.search_service import SearchResult, SearchService


def get_terminal_width() -> int:
    """Get the current terminal width.

    Returns:
        Terminal width in characters (minimum 70, maximum 150)
    """
    try:
        width = shutil.get_terminal_size().columns
        # Clamp between reasonable values
        return max(70, min(width - 2, 150))
    except Exception:
        return 100  # Default fallback


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Args:
        verbose: Enable verbose logging
    """
    logger.remove()  # Remove default handler
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


def print_banner() -> None:
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    Wikipedia Vector Search Knowledge Base - Demo             ║
║                                                              ║
║    Search using Vector, Text, or Hybrid methods              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help() -> None:
    """Print help information."""
    help_text = """
Available Commands:
  /vector <query>     - Semantic search using embeddings
  /text <query>       - Keyword search using MongoDB text search
  /hybrid <query>     - Combined search (RRF algorithm)
  /weighted <query>   - Combined search (weighted scores)
  /compare <query>    - Compare all search methods
  /stats              - Show database statistics
  /help               - Show this help
  /quit or /exit      - Exit the program

Search Tips:
  - Vector search: Best for semantic/conceptual queries
    Example: "How does photosynthesis convert light to energy?"

  - Text search: Best for keyword/exact match queries
    Example: "Albert Einstein relativity"

  - Hybrid search: Combines both approaches
    Example: "machine learning neural networks"

Press Enter without a command to use hybrid search (default).
    """
    print(help_text)


def pad_line(text: str, width: int) -> str:
    """Pad a line to the full width and add closing border.

    Args:
        text: Line of text (should start with │)
        width: Total terminal width

    Returns:
        Padded line with closing │
    """
    # Calculate how much padding we need
    current_length = len(text)
    padding_needed = width - current_length - 1  # -1 for closing │
    if padding_needed > 0:
        return text + " " * padding_needed + "│"
    else:
        # Line is too long, truncate and add │
        return text[:width - 1] + "│"


def wrap_text(text: str, width: int, prefix: str = "│   │ ") -> list[str]:
    """Wrap text to fit within terminal width.

    Args:
        text: Text to wrap
        width: Maximum width including prefix
        prefix: Line prefix (e.g., table border)

    Returns:
        List of wrapped lines with proper padding
    """
    # Reserve space for: prefix + ending │ + at least 1 space before │
    available_width = width - len(prefix) - 2  # -2 for ending " │"
    if available_width <= 20:
        available_width = 40  # Minimum reasonable width

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word) + (1 if current_line else 0)  # +1 for space
        if current_length + word_length > available_width and current_line:
            line_text = " ".join(current_line)
            lines.append(pad_line(f"{prefix}{line_text}", width))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += word_length

    if current_line:
        line_text = " ".join(current_line)
        lines.append(pad_line(f"{prefix}{line_text}", width))

    return lines


def format_result(result: SearchResult, index: int, show_text: bool = True, width: int = 100) -> str:
    """Format a search result for display.

    Args:
        result: Search result to format
        index: Result index (1-based)
        show_text: Whether to show the full text
        width: Terminal width for formatting

    Returns:
        Formatted result string
    """
    # Table-style format with metadata row and text row
    section_info = f" › {result.section}" if result.section else ""

    lines = [
        pad_line(f"│ {index} │ {result.title}{section_info}", width),
        pad_line(f"│   │ Score: {result.score:.4f}", width),
    ]

    if show_text:
        # Show text sample
        text_preview = result.text[:1000] + "..." if len(result.text) > 1000 else result.text
        # Wrap text to fit terminal width (wrap_text now returns padded lines)
        wrapped_lines = wrap_text(text_preview, width)
        lines.extend(wrapped_lines)

    return "\n".join(lines)


def print_results(
    results: list[SearchResult], search_type: str, query: str, show_text: bool = True
) -> None:
    """Print search results.

    Args:
        results: List of search results
        search_type: Type of search performed
        query: The search query
        show_text: Whether to show full text
    """
    width = get_terminal_width()
    line_width = width - 2  # Account for border characters

    # Table header
    print(f"\n┌{'─' * line_width}┐")
    print(pad_line(f"│ {search_type.upper()}: \"{query}\"", width))
    print(f"├{'─' * line_width}┤")

    if not results:
        print(pad_line("│ No results found.", width))
        print(f"└{'─' * line_width}┘")
        return

    # Only show top 3 results
    for i, result in enumerate(results[:3], 1):
        if i > 1:
            print(f"├{'─' * line_width}┤")
        print(format_result(result, i, show_text, width))

    print(f"└{'─' * line_width}┘")


def print_comparison(
    vector_results: list[SearchResult],
    text_results: list[SearchResult],
    hybrid_results: list[SearchResult],
    query: str,
) -> None:
    """Print comparison of all search methods.

    Args:
        vector_results: Results from vector search
        text_results: Results from text search
        hybrid_results: Results from hybrid search
        query: The search query
    """
    width = get_terminal_width()
    line_width = width - 2  # Account for border characters

    # Table header
    print(f"\n┌{'─' * line_width}┐")
    print(pad_line(f"│ COMPARISON: \"{query}\"", width))
    print(f"├{'─' * line_width}┤")

    # Show top 3 results from each method
    methods = [
        ("VECTOR", vector_results),
        ("TEXT", text_results),
        ("HYBRID", hybrid_results),
    ]

    for method_idx, (method_name, results) in enumerate(methods):
        if method_idx > 0:
            print(f"├{'─' * line_width}┤")

        print(pad_line(f"│ {method_name}", width))
        print(f"├{'─' * line_width}┤")

        if not results:
            print(pad_line("│ No results", width))
            continue

        for i, result in enumerate(results[:3], 1):
            if i > 1:
                print(f"├{'·' * line_width}┤")

            section_info = f" › {result.section}" if result.section else ""
            print(pad_line(f"│ {i} │ {result.title}{section_info}", width))
            print(pad_line(f"│   │ Score: {result.score:.4f}", width))
            # Show bigger preview (300 chars for comparison)
            preview = result.text[:300] + "..." if len(result.text) > 300 else result.text
            # Wrap text to fit terminal width (wrap_text now returns padded lines)
            wrapped_lines = wrap_text(preview, width)
            for line in wrapped_lines:
                print(line)

    # Overlap analysis
    vector_ids = {r.chunk_id for r in vector_results[:3]}
    text_ids = {r.chunk_id for r in text_results[:3]}
    hybrid_ids = {r.chunk_id for r in hybrid_results[:3]}

    v_t = len(vector_ids & text_ids)
    v_h = len(vector_ids & hybrid_ids)
    t_h = len(text_ids & hybrid_ids)

    print(f"├{'─' * line_width}┤")
    print(pad_line(f"│ Overlap: V∩T={v_t} │ V∩H={v_h} │ T∩H={t_h}", width))
    print(f"└{'─' * line_width}┘")


def print_stats(db_manager: MongoDBManager) -> None:
    """Print database statistics.

    Args:
        db_manager: MongoDB manager instance
    """
    print(f"\n{'=' * 70}")
    print("Database Statistics")
    print(f"{'=' * 70}\n")

    # Get collection stats
    stats = db_manager.get_collection_stats()

    print("Articles Collection:")
    print(f"  Documents: {stats['articles']['count']:,}")
    print(f"  Size:      {stats['articles']['size_bytes'] / (1024 * 1024):.2f} MB")
    if stats["articles"]["count"] > 0:
        print(f"  Avg Size:  {stats['articles']['avg_size_bytes'] / 1024:.2f} KB")

    print("\nChunks Collection:")
    print(f"  Documents: {stats['chunks']['count']:,}")
    print(f"  Size:      {stats['chunks']['size_bytes'] / (1024 * 1024):.2f} MB")
    if stats["chunks"]["count"] > 0:
        print(f"  Avg Size:  {stats['chunks']['avg_size_bytes'] / 1024:.2f} KB")

    # Show relationship stats
    if stats["articles"]["count"] > 0:
        print(f"\nAverage chunks per article: {stats['avg_chunks_per_article']:.1f}")


def run_interactive_search() -> None:
    """Run interactive search CLI."""
    setup_logging(verbose=False)
    print_banner()

    # Load configuration from config.yaml
    try:
        config = load_config()
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}")
        print("Please ensure config/config.yaml exists and is valid")
        return

    print("\nConnecting to MongoDB...")
    print(f"LMStudio URL: {config.embedding.lmstudio_url}")
    print(f"Embedding Model: {config.embedding.model}")

    try:
        # Initialize services
        db_manager = MongoDBManager(config=config.mongodb)
        embedding_gen = EmbeddingGenerator(config=config.embedding)
        search_service = SearchService(db_manager, embedding_gen)

        print("✓ Connected successfully!\n")
        print_help()

        # Main loop
        while True:
            try:
                user_input = input("\n🔍 Enter query (or /help): ").strip()

                if not user_input:
                    continue

                # Parse command
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    query = parts[1] if len(parts) > 1 else ""

                    if command in ["/quit", "/exit"]:
                        print("\nGoodbye!")
                        break

                    if command == "/help":
                        print_help()
                        continue

                    if command == "/stats":
                        print_stats(db_manager)
                        continue

                    if not query:
                        print("Error: Query required for search commands")
                        continue

                    # Execute search based on command
                    start_time = datetime.now()

                    if command == "/vector":
                        results = search_service.vector_search(query, limit=3)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_results(results, "Vector", query)
                        print(f"⏱️  {elapsed:.2f}s")

                    elif command == "/text":
                        results = search_service.text_search(query, limit=3)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_results(results, "Text", query)
                        print(f"⏱️  {elapsed:.2f}s")

                    elif command == "/hybrid":
                        results = search_service.hybrid_search(query, limit=3, use_rrf=True)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_results(results, "Hybrid (RRF)", query)
                        print(f"⏱️  {elapsed:.2f}s")

                    elif command == "/weighted":
                        results = search_service.hybrid_search(
                            query, limit=3, use_rrf=False, vector_weight=0.6, text_weight=0.4
                        )
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_results(results, "Hybrid (Weighted)", query)
                        print(f"⏱️  {elapsed:.2f}s")

                    elif command == "/compare":
                        vector_results = search_service.vector_search(query, limit=3)
                        text_results = search_service.text_search(query, limit=3)
                        hybrid_results = search_service.hybrid_search(query, limit=3)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_comparison(vector_results, text_results, hybrid_results, query)
                        print(f"⏱️  {elapsed:.2f}s")

                    else:
                        print(f"Unknown command: {command}")
                        print("Type /help for available commands")

                else:
                    # Default: hybrid search
                    start_time = datetime.now()
                    results = search_service.hybrid_search(user_input, limit=3)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print_results(results, "Hybrid (RRF)", user_input)
                    print(f"⏱️  {elapsed:.2f}s")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit or continue searching.")
                continue
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\n❌ Error: {e}")
                print("Please try again or type /help for assistance.")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check your configuration and try again.")
    finally:
        if "db_manager" in locals():
            db_manager.close()


def main() -> None:
    """Main entry point."""
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)

    run_interactive_search()


if __name__ == "__main__":
    main()
