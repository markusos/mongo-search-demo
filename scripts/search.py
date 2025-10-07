#!/usr/bin/env -S uv run python
"""Interactive demo for Wikipedia vector search knowledge base.

This script provides an interactive CLI for searching the Wikipedia knowledge base
using vector search, text search, or hybrid search methods.
"""

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


def format_result(result: SearchResult, index: int, show_text: bool = True) -> str:
    """Format a search result for display.

    Args:
        result: Search result to format
        index: Result index (1-based)
        show_text: Whether to show the full text

    Returns:
        Formatted result string
    """
    lines = [
        f"\n{'=' * 70}",
        f"Result #{index}",
        f"{'=' * 70}",
        f"Title:   {result.title}",
        f"Score:   {result.score:.4f}",
        f"Rank:    {result.rank}",
        f"Type:    {result.search_type}",
    ]

    if result.section:
        lines.append(f"Section: {result.section}")

    lines.append(f"Page ID: {result.page_id}")
    lines.append(f"Chunk:   {result.chunk_id}")

    if show_text:
        lines.append("\nContent Preview:")
        lines.append("-" * 70)
        # Limit text preview to 300 characters
        text_preview = result.text[:300] + "..." if len(result.text) > 300 else result.text
        lines.append(text_preview)

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
    print(f"\n{'=' * 70}")
    print(f"Search Type: {search_type.upper()}")
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print(f"{'=' * 70}")

    if not results:
        print("\nNo results found.")
        return

    for i, result in enumerate(results, 1):
        print(format_result(result, i, show_text))


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
    print(f"\n{'=' * 70}")
    print(f"SEARCH COMPARISON: {query}")
    print(f"{'=' * 70}\n")

    # Show top 3 results from each method
    methods = [
        ("Vector Search", vector_results),
        ("Text Search", text_results),
        ("Hybrid Search", hybrid_results),
    ]

    for method_name, results in methods:
        print(f"\n{method_name} (Top 3):")
        print("-" * 70)
        if not results:
            print("  No results")
            continue

        for i, result in enumerate(results[:3], 1):
            print(f"\n  {i}. {result.title}")
            print(f"     Score: {result.score:.4f} | Rank: {result.rank}")
            if result.section:
                print(f"     Section: {result.section}")
            # Show first 100 chars of text
            preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
            print(f"     {preview}")

    # Show overlap analysis
    print(f"\n{'=' * 70}")
    print("Overlap Analysis:")
    print("-" * 70)

    vector_ids = {r.chunk_id for r in vector_results[:5]}
    text_ids = {r.chunk_id for r in text_results[:5]}
    hybrid_ids = {r.chunk_id for r in hybrid_results[:5]}

    v_t_overlap = len(vector_ids & text_ids)
    v_h_overlap = len(vector_ids & hybrid_ids)
    t_h_overlap = len(text_ids & hybrid_ids)

    print(f"Vector ∩ Text:   {v_t_overlap} common results (top 5)")
    print(f"Vector ∩ Hybrid: {v_h_overlap} common results (top 5)")
    print(f"Text ∩ Hybrid:   {t_h_overlap} common results (top 5)")


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
        embedding_gen = EmbeddingGenerator(config=config.embedding, max_retries=3)
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
                        results = search_service.vector_search(query, limit=10)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_results(results, "Vector", query)
                        print(f"\n⏱️  Search completed in {elapsed:.2f}s")

                    elif command == "/text":
                        results = search_service.text_search(query, limit=10)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_results(results, "Text", query)
                        print(f"\n⏱️  Search completed in {elapsed:.2f}s")

                    elif command == "/hybrid":
                        results = search_service.hybrid_search(query, limit=10, use_rrf=True)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_results(results, "Hybrid (RRF)", query)
                        print(f"\n⏱️  Search completed in {elapsed:.2f}s")

                    elif command == "/weighted":
                        results = search_service.hybrid_search(
                            query, limit=10, use_rrf=False, vector_weight=0.6, text_weight=0.4
                        )
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_results(results, "Hybrid (Weighted)", query)
                        print(f"\n⏱️  Search completed in {elapsed:.2f}s")

                    elif command == "/compare":
                        vector_results = search_service.vector_search(query, limit=10)
                        text_results = search_service.text_search(query, limit=10)
                        hybrid_results = search_service.hybrid_search(query, limit=10)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print_comparison(vector_results, text_results, hybrid_results, query)
                        print(f"\n⏱️  All searches completed in {elapsed:.2f}s")

                    else:
                        print(f"Unknown command: {command}")
                        print("Type /help for available commands")

                else:
                    # Default: hybrid search
                    start_time = datetime.now()
                    results = search_service.hybrid_search(user_input, limit=10)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print_results(results, "Hybrid (RRF)", user_input)
                    print(f"\n⏱️  Search completed in {elapsed:.2f}s")

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
