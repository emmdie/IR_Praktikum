import sys
import os
from pathlib import Path
from typing import Dict, List

from rich import color
from frontend import fake_results_generator, df_prepocessing
import pandas as pd

from textual import on
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textual.widget import Widget
from textual.widgets import Select
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Button, Digits, Input, Label, Static, RadioSet, RadioButton
from textual.containers import HorizontalGroup, VerticalScroll, Vertical, Horizontal, Container
from textual.reactive import reactive
from rich.text import Text
from textual.message import Message

from src.dynamic_approach.SBERT_HDBSCAN import the_function

class SearchTriggered(Message):
    def __init__(self, sender: Widget, query: str) -> None:
        super().__init__()
        self.sender = sender
        self.query = query

class SearchEngineFrontend(App):
    CSS_PATH = Path(__file__).parent / "style.css"  # Commented out for testing
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield SearchBar()
        yield VerticalScroll(id="results_container")

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light")

    def populate_results_from_dict(self, result_df) -> None:
        """Populate the UI with reranked results"""
        self.clear_results()
        results_container: Widget = self.query_one("#results_container")

        # Convert DataFrame to list of dicts if needed
        if isinstance(result_df, pd.DataFrame):
            result_list = result_df.reset_index().to_dict('records')
        else:
            result_list = result_df

        for entry in result_list:
            rf = ResultField(
                original_ranking=entry.get("init_ranking", 0),
                new_ranking=entry.get("new_ranking", 0),
                cluster=str(entry.get("cluster", "N/A")),
                doc_id=str(entry.get("doc_id", "")),
                label=str(entry.get("label", "")),
                text=entry.get("text", "No text available"),
                similarity_score=entry.get("similarity_score", 0.0)
            )
            results_container.mount(rf)


    def on_search_triggered(self, message: SearchTriggered) -> None:
        """Handle search trigger and apply reranking using the_function"""
        if not message.query.strip():
            return

        # Get reranking method from UI
        # reranking_controls = self.query_one("#reranking_controls", RerankingControls)
        # method = reranking_controls.get_selected_method()
        method = 'hdbscan'

        try:
            # Use the_function to get reranked results
            results = the_function(message.query, k=5, method=method)
            # self.populate_results_from_dict(results)
        except Exception as e:
            # Show error in results
            self.clear_results()
            results_container: Widget = self.query_one("#results_container")
            error_field = Static(f"Error: {str(e)}", classes="error-message")
            results_container.mount(error_field)
            
            
        # results_df = fake_results_generator.generate_fake_results_df()
        results_df = df_prepocessing.assign_cluster_colors(results)
        self.populate_results_from_dict(results_df)
        
        
    def clear_results(self) -> None:
        """Clear all results from the UI"""
        results_container: Widget = self.query_one("#results_container")
        for child in results_container.children:
            child.remove()

LINES = """Lama
SBERT
Colbert""".splitlines()

class SearchBar(HorizontalGroup):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter search term...", id="search_input", classes="search_input")
        yield Button("Search 🔎", id="start", variant="success")
        yield Select((line, line) for line in LINES)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            query_input = self.query_one("#search_input", Input)
            self.post_message(SearchTriggered(self, query_input.value))
    
    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        self.title = str(event.value)

# class RerankingControls(Vertical):
#     def __init__(self):
#         super().__init__()
#         self.add_class("reranking-controls")

#     def compose(self) -> ComposeResult:
#         yield Label("Reranking Method:", classes="control-label")
#         yield RadioSet(
#             RadioButton("Original (No Reranking)", value=True, id="original"),
#             RadioButton("K-Means Clustering", id="kmeans"),
#             RadioButton("HDBSCAN Clustering", id="hdbscan"),
#             id="reranking_method"
#         )

#     def get_selected_method(self) -> str:
#         """Get the currently selected reranking method"""
#         radio_set = self.query_one("#reranking_method", RadioSet)
#         if radio_set.pressed_button:
#             return radio_set.pressed_button.id
#         return "original"


class ResultField(Vertical):
    def __init__(self, original_ranking: int, new_ranking: int, cluster: str, doc_id: str, label: str, text: str, similarity_score: float = 0.0) -> None:
        super().__init__()
        self.original_ranking = original_ranking
        self.new_ranking = new_ranking
        self.cluster = cluster
        self.doc_id = doc_id
        self.label = label
        self.text = text or "No text available"  # Ensure text is not None/empty
        self.similarity_score = similarity_score
        self.color = color

    def compose(self):
        # Create header with ranking information
        header_text = f"#{self.new_ranking} {self.label}"
        if self.new_ranking != self.original_ranking:
            header_text += f" (was #{self.original_ranking})"
        header_text += f" | Cluster: {self.cluster} | ID: {self.doc_id}"
        if self.similarity_score > 0:
            header_text += f" | Score: {self.similarity_score:.4f}"

        # Truncate very long text for display
        display_text = self.text[:500] + "..." if len(self.text) > 500 else self.text

        yield Container(
            Vertical(
                Label(header_text, classes="result-header"),
                Static(display_text, classes="result-body"),  # Use plain text instead of markup
            ),
            classes="result-frame"
        )
        
