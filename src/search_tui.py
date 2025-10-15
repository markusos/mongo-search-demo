"""TUI components for Wikipedia vector search application."""

from datetime import datetime

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Static

from src.search_service import SearchResult, SearchService


class SearchResultWidget(Static):
    """Widget to display a single search result."""

    DEFAULT_CSS = """
    SearchResultWidget {
        height: auto;
        padding: 0 1;
        margin: 0;
    }

    SearchResultWidget.selected {
        background: $boost;
        text-style: bold;
    }

    SearchResultWidget .title {
        text-style: bold;
        color: $text;
    }

    SearchResultWidget .score {
        color: $warning;
    }

    SearchResultWidget .text {
        color: $text-muted;
        margin: 1 0;
    }

    SearchResultWidget .metadata {
        color: $text-disabled;
        text-style: italic;
    }
    """

    def __init__(self, result: SearchResult, index: int):
        super().__init__()
        self.result = result
        self.index = index

    def render(self) -> str:
        """Render the search result as a compact two-line item."""
        section_info = f" › {self.result.section}" if self.result.section else ""
        title = f"[bold]{self.index}. {self.result.title}{section_info}[/bold]"
        score = f"[yellow]{self.result.score:.4f}[/yellow]"
        chunk_id = f"[dim]{self.result.chunk_id}[/dim]"

        # Two lines: title+score, then chunk ID
        return f"{title} {score}\n  {chunk_id}"


class SearchApp(App):
    """Wikipedia Search TUI Application."""

    CSS = """
    #search-container {
        dock: top;
        height: 5;
        background: $panel;
    }

    #search-box {
        height: 3;
        margin: 1;
    }

    #search-type-label {
        width: 15;
        content-align: center middle;
        background: $primary;
        color: $text;
        text-style: bold;
        border: solid $primary;
        margin-right: 1;
    }

    #search-input {
        width: 1fr;
    }

    #search-controls {
        height: 3;
        align: center middle;
    }

    #main-content {
        height: 1fr;
    }

    #results-list {
        width: 40%;
        border-right: solid $primary;
        padding: 1;
    }

    #detail-pane {
        width: 60%;
        border: solid $primary;
        padding: 2;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        background: $panel;
        padding: 1;
    }

    .search-button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("?", "help", "Help", show=True),
        Binding("c", "compare", "Compare", show=True),
        Binding("s", "show_stats", "Stats", show=True),
        Binding("v", "set_vector", "Vector", show=True),
        Binding("t", "set_text", "Text", show=True),
        Binding("h", "set_hybrid", "Hybrid", show=True),
        Binding("tab", "cycle_focus", "Tab=Focus", show=True, priority=True),
        Binding("escape", "escape", "Cancel", show=False),
    ]

    TITLE = "Wikipedia Vector Search"
    SUB_TITLE = "v=Vector t=Text h=Hybrid | Tab=Cycle Focus | ?=Help q=Quit"

    selected_index: reactive[int] = reactive(0)
    search_type: reactive[str] = reactive("hybrid")
    current_focus_index: int = 0  # 0=input, 1=results, 2=detail

    def __init__(self, search_service: SearchService, db_manager):
        super().__init__()
        self.results: list[SearchResult] = []
        self.last_query = ""
        self.last_search_time = 0.0
        self.search_service = search_service
        self.db_manager = db_manager

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with Container(id="search-container"):
            with Horizontal(id="search-box"):
                label = Static("HYBRID", id="search-type-label")
                label.can_focus = False
                yield label
                yield Input(placeholder="Enter search query...", id="search-input")
            with Horizontal(id="search-controls"):
                vector_btn = Button("Vector", id="vector-btn", classes="search-button")
                vector_btn.can_focus = False
                yield vector_btn
                text_btn = Button("Text", id="text-btn", classes="search-button")
                text_btn.can_focus = False
                yield text_btn
                hybrid_btn = Button(
                    "Hybrid", id="hybrid-btn", classes="search-button", variant="primary"
                )
                hybrid_btn.can_focus = False
                yield hybrid_btn
                compare_btn = Button("Compare", id="compare-btn", classes="search-button")
                compare_btn.can_focus = False
                yield compare_btn

        with Horizontal(id="main-content"):
            yield VerticalScroll(id="results-list")
            yield VerticalScroll(
                Static("Select a result to view details", id="detail-text"), id="detail-pane"
            )

        status_bar = Static("Ready. Press ? for help, q to quit.", id="status-bar")
        status_bar.can_focus = False
        yield status_bar
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the application."""
        try:
            # Get stats
            stats = self.db_manager.get_collection_stats()
            article_count = stats["articles"]["count"]
            chunk_count = stats["chunks"]["count"]

            self.update_status(
                f"✓ Connected | Articles: {article_count:,} | Chunks: {chunk_count:,} | Press ? for help, q to quit"
            )
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            self.update_status(f"Error: {e}")

    def update_status(self, message: str) -> None:
        """Update the status bar."""
        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(message)

    def update_button_states(self) -> None:
        """Update button visual states based on selected search type."""
        vector_btn = self.query_one("#vector-btn", Button)
        text_btn = self.query_one("#text-btn", Button)
        hybrid_btn = self.query_one("#hybrid-btn", Button)
        type_label = self.query_one("#search-type-label", Static)

        # Reset all buttons to default variant
        vector_btn.variant = "primary" if self.search_type == "vector" else "default"
        text_btn.variant = "primary" if self.search_type == "text" else "default"
        hybrid_btn.variant = "primary" if self.search_type == "hybrid" else "default"

        # Update the search type label
        type_label.update(self.search_type.upper())

    @on(Button.Pressed, "#vector-btn")
    async def on_vector_button(self) -> None:
        """Handle vector search button."""
        self.search_type = "vector"
        self.update_button_states()
        await self.perform_search()

    @on(Button.Pressed, "#text-btn")
    async def on_text_button(self) -> None:
        """Handle text search button."""
        self.search_type = "text"
        self.update_button_states()
        await self.perform_search()

    @on(Button.Pressed, "#hybrid-btn")
    async def on_hybrid_button(self) -> None:
        """Handle hybrid search button."""
        self.search_type = "hybrid"
        self.update_button_states()
        await self.perform_search()

    @on(Button.Pressed, "#compare-btn")
    async def on_compare_button(self) -> None:
        """Handle compare button."""
        await self.show_comparison()

    @on(Input.Submitted, "#search-input")
    async def on_search_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        await self.perform_search()

    async def perform_search(self) -> None:
        """Perform a search with the current query."""
        search_input = self.query_one("#search-input", Input)
        query = search_input.value.strip()

        if not query:
            self.update_status("Please enter a search query")
            return

        if not self.search_service:
            self.update_status("Search service not initialized")
            return

        self.update_status(f"Searching ({self.search_type})...")
        self.last_query = query

        try:
            start_time = datetime.now()

            if self.search_type == "vector":
                self.results = self.search_service.vector_search(query, limit=10)
            elif self.search_type == "text":
                self.results = self.search_service.text_search(query, limit=10)
            else:  # hybrid
                self.results = self.search_service.hybrid_search(query, limit=10, use_rrf=True)

            elapsed = (datetime.now() - start_time).total_seconds()
            self.last_search_time = elapsed

            # Reset selection
            self.selected_index = 0

            # Update results display
            await self.update_results_display()

            # Auto-focus results list so user can immediately navigate
            results_list = self.query_one("#results-list")
            results_list.focus()
            self.current_focus_index = 1

            # Show first result in detail pane
            if self.results:
                await self.update_detail_pane()

            self.update_status(
                f"Found {len(self.results)} results in {elapsed:.2f}s | "
                f"Type: {self.search_type.upper()} | Focus: Results (↑↓ navigate, Tab cycle)"
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.update_status(f"Search error: {e}")

    async def update_results_display(self) -> None:
        """Update the results list display."""
        results_list = self.query_one("#results-list", VerticalScroll)
        await results_list.remove_children()

        if not self.results:
            await results_list.mount(Static("No results found.", classes="no-results"))
            return

        selected_widget = None
        for i, result in enumerate(self.results, 1):
            widget = SearchResultWidget(result, i)

            if i == self.selected_index + 1:
                widget.add_class("selected")
                selected_widget = widget

            await results_list.mount(widget)

        # Scroll selected widget into view
        if selected_widget is not None:
            results_list.scroll_to_widget(selected_widget, animate=False)

    async def update_detail_pane(self) -> None:
        """Update the detail pane with the selected result."""
        if not self.results or self.selected_index >= len(self.results):
            return

        result = self.results[self.selected_index]
        detail_pane = self.query_one("#detail-pane", VerticalScroll)
        await detail_pane.remove_children()

        # Create detailed view using Rich Text to safely handle special characters
        text = Text()

        # Add title (bold cyan)
        section_info = f" › {result.section}" if result.section else ""
        text.append(f"{result.title}{section_info}", style="bold cyan")
        text.append("\n")

        # Add metadata
        text.append(f"Score: {result.score:.4f}", style="yellow")
        text.append(" | ")
        text.append(f"Search Type: {result.search_type}", style="dim")
        text.append(" | ")
        text.append(f"Chunk ID: {result.chunk_id}", style="dim")
        text.append("\n\n")

        # Add content text (plain, no markup parsing)
        text.append(result.text)

        await detail_pane.mount(Static(text, id="detail-text"))

    def action_cycle_focus(self) -> None:
        """Cycle focus between search input, results list, and detail pane."""
        self.current_focus_index = (self.current_focus_index + 1) % 3

        if self.current_focus_index == 0:
            # Focus search input
            search_input = self.query_one("#search-input", Input)
            search_input.focus()
            self.update_status("Focus: Search Input (type query, Enter to search, Tab to cycle)")
        elif self.current_focus_index == 1:
            # Focus results list
            results_list = self.query_one("#results-list", VerticalScroll)
            results_list.focus()
            self.update_status("Focus: Results List (↑↓ to navigate, Tab to cycle)")
        else:
            # Focus detail pane
            detail_pane = self.query_one("#detail-pane", VerticalScroll)
            detail_pane.focus()
            self.update_status("Focus: Detail Pane (↑↓ to scroll, Tab to cycle)")

    async def action_help(self) -> None:
        """Show help information."""
        help_text = """[bold cyan]Wikipedia Vector Search - Help[/bold cyan]

[yellow]Search Methods:[/yellow]
  [cyan]v[/cyan]       Vector search (semantic/embeddings)
  [cyan]t[/cyan]       Text search (keyword/MongoDB)
  [cyan]h[/cyan]       Hybrid search (RRF algorithm)
  [dim]Buttons[/dim]  Click Vector/Text/Hybrid buttons

[yellow]Focus & Navigation:[/yellow]
  [cyan]Tab[/cyan]     Cycle focus: Input → Results → Detail → Input...
  [cyan]↑/↓[/cyan]     Navigate results (when Results focused) or scroll (when Detail focused)
  [cyan]Enter[/cyan]   Submit search (when Input focused)

[yellow]Actions:[/yellow]
  [cyan]c[/cyan]       Compare all search methods
  [cyan]s[/cyan]       Show database statistics
  [cyan]?[/cyan]       Show this help
  [cyan]q[/cyan]       Quit

[yellow]Layout:[/yellow]
  • Top: Search input with mode label (VECTOR/TEXT/HYBRID)
  • Left pane: Results list (rank, title, score, chunk ID)
  • Right pane: Full text of selected result (auto-updates)

[yellow]Tips:[/yellow]
  • Press v/t/h to switch search modes (only when NOT in input field)
  • Use Tab to cycle through input, results, and detail panes
  • Detail pane updates automatically as you navigate results
  • Press ? again or Esc to return to normal view
"""
        # Display help in the detail pane
        detail_pane = self.query_one("#detail-pane", VerticalScroll)
        await detail_pane.remove_children()
        await detail_pane.mount(Static(help_text, id="help-text"))

        # Focus the detail pane so user can scroll if needed
        detail_pane.focus()
        self.current_focus_index = 2

        self.update_status("Help displayed in detail pane | Press ? again or Esc to return")

    def action_show_stats(self) -> None:
        """Show database statistics."""
        if not self.db_manager:
            self.update_status("Database not connected")
            return

        try:
            stats = self.db_manager.get_collection_stats()
            stats_text = (
                f"Articles: {stats['articles']['count']:,} "
                f"({stats['articles']['size_bytes'] / (1024 * 1024):.1f} MB) | "
                f"Chunks: {stats['chunks']['count']:,} "
                f"({stats['chunks']['size_bytes'] / (1024 * 1024):.1f} MB) | "
                f"Avg chunks/article: {stats['avg_chunks_per_article']:.1f}"
            )
            self.update_status(stats_text)
        except Exception as e:
            self.update_status(f"Error getting stats: {e}")

    async def show_comparison(self) -> None:
        """Show comparison of all search methods."""
        search_input = self.query_one("#search-input", Input)
        query = search_input.value.strip()

        if not query:
            self.update_status("Please enter a search query for comparison")
            return

        if not self.search_service:
            self.update_status("Search service not initialized")
            return

        self.update_status("Running comparison (this may take a moment)...")

        try:
            start_time = datetime.now()

            vector_results = self.search_service.vector_search(query, limit=10)
            text_results = self.search_service.text_search(query, limit=10)
            hybrid_results = self.search_service.hybrid_search(query, limit=10)

            elapsed = (datetime.now() - start_time).total_seconds()

            # Calculate overlap
            vector_ids = {r.chunk_id for r in vector_results[:5]}
            text_ids = {r.chunk_id for r in text_results[:5]}
            hybrid_ids = {r.chunk_id for r in hybrid_results[:5]}

            v_t = len(vector_ids & text_ids)
            v_h = len(vector_ids & hybrid_ids)
            t_h = len(text_ids & hybrid_ids)

            self.update_status(
                f"Comparison complete ({elapsed:.2f}s) | "
                f"Results: V={len(vector_results)} T={len(text_results)} H={len(hybrid_results)} | "
                f"Overlap: V∩T={v_t} V∩H={v_h} T∩H={t_h}"
            )

            # Show hybrid results by default
            self.search_type = "hybrid"
            self.update_button_states()
            self.results = hybrid_results
            self.selected_index = 0
            await self.update_results_display()
            await self.update_detail_pane()

        except Exception as e:
            logger.error(f"Comparison error: {e}")
            self.update_status(f"Comparison error: {e}")

    async def action_compare(self) -> None:
        """Trigger comparison action."""
        await self.show_comparison()

    async def action_escape(self) -> None:
        """Handle escape key."""
        search_input = self.query_one("#search-input", Input)
        detail_pane = self.query_one("#detail-pane")
        results_list = self.query_one("#results-list")

        # Check if help is showing
        try:
            help_widget = self.query_one("#help-text")
            if help_widget:
                # Help is showing, restore previous view
                if self.results:
                    await self.update_detail_pane()
                    results_list.focus()
                    self.current_focus_index = 1
                    self.update_status(
                        "Help closed | Focus: Results List (↑↓ to navigate, Tab to cycle)"
                    )
                else:
                    await detail_pane.remove_children()
                    await detail_pane.mount(
                        Static("Select a result to view details", id="detail-text")
                    )
                    search_input.focus()
                    self.current_focus_index = 0
                    self.update_status("Help closed | Focus: Search Input")
                return
        except Exception:
            pass  # Help not showing

        if search_input.has_focus:
            self.screen.set_focus(None)
        elif detail_pane.has_focus:
            results_list.focus()
        else:
            self.update_status("Press q to quit, ? for help")

    async def action_set_vector(self) -> None:
        """Set search type to vector."""
        self.search_type = "vector"
        self.update_button_states()
        # Focus search input
        search_input = self.query_one("#search-input", Input)
        search_input.focus()
        self.current_focus_index = 0
        # Trigger search if there's a query
        if search_input.value.strip():
            await self.perform_search()
        else:
            self.update_status("Search mode: VECTOR | Enter a query to search")

    async def action_set_text(self) -> None:
        """Set search type to text."""
        self.search_type = "text"
        self.update_button_states()
        # Focus search input
        search_input = self.query_one("#search-input", Input)
        search_input.focus()
        self.current_focus_index = 0
        # Trigger search if there's a query
        if search_input.value.strip():
            await self.perform_search()
        else:
            self.update_status("Search mode: TEXT | Enter a query to search")

    async def action_set_hybrid(self) -> None:
        """Set search type to hybrid."""
        self.search_type = "hybrid"
        self.update_button_states()
        # Focus search input
        search_input = self.query_one("#search-input", Input)
        search_input.focus()
        self.current_focus_index = 0
        # Trigger search if there's a query
        if search_input.value.strip():
            await self.perform_search()
        else:
            self.update_status("Search mode: HYBRID | Enter a query to search")

    async def refresh_results(self) -> None:
        """Refresh the results display without re-fetching."""
        await self.update_results_display()
        await self.update_detail_pane()

    async def on_key(self, event) -> None:
        """Handle key presses for navigation."""
        search_input = self.query_one("#search-input", Input)
        results_list = self.query_one("#results-list", VerticalScroll)

        # Handle tab for cycling - intercept to prevent default tab behavior
        if event.key == "tab":
            self.action_cycle_focus()
            event.prevent_default()
            event.stop()
            return

        # When in input mode, don't intercept v/t/h (let user type them)
        if search_input.has_focus:
            return

        # Handle v/t/h for search mode switching (only when NOT in input)
        if event.key == "v":
            await self.action_set_vector()
            event.prevent_default()
            event.stop()
            return
        elif event.key == "t":
            await self.action_set_text()
            event.prevent_default()
            event.stop()
            return
        elif event.key == "h":
            await self.action_set_hybrid()
            event.prevent_default()
            event.stop()
            return

        # Handle arrow keys in results list for result selection
        if results_list.has_focus and self.results:
            if event.key == "up":
                if self.selected_index > 0:
                    self.selected_index -= 1
                    await self.refresh_results()
                event.prevent_default()
                event.stop()
            elif event.key == "down":
                if self.selected_index < len(self.results) - 1:
                    self.selected_index += 1
                    await self.refresh_results()
                event.prevent_default()
                event.stop()

        # Allow normal scrolling in detail pane
        # (Textual handles this by default, we just don't intercept)

    def action_quit(self) -> None:
        """Quit the application."""
        if self.db_manager:
            self.db_manager.close()
        self.exit()
