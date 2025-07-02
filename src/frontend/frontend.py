import sys
import os
from pathlib import Path
from typing import Dict, List

from rich import color
from frontend import fake_results_generator, df_prepocessing
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


from textual import on
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textual.widget import Widget
from textual.widgets import Select, Switch
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Button, Digits, Input, Label, Static, RadioSet, RadioButton
from textual.containers import HorizontalGroup, VerticalScroll, Vertical, Horizontal, Container
from textual.reactive import reactive
from textual.screen import ModalScreen
from rich.text import Text
from textual.message import Message

from src.dynamic_approach.optimized_search import the_function
from src.static_approach.SBERT_static_search import sbert_static_search

class SearchTriggered(Message):
    def __init__(self, sender: Widget, query: str, approach: str) -> None:
        super().__init__()
        self.sender = sender
        self.query = query
        self.approach = approach
        
class SettingsScreen(ModalScreen):
    """A modal settings screen"""
    
    def __init__(self, app_settings: dict):
        super().__init__()
        self.app_settings = app_settings
    
    def compose(self) -> ComposeResult:
        yield Container(
            Vertical(
                Label("⚙️  Settings", classes="settings-title"),
                
                # Clustering method selection
                Label("Clustering Method:", classes="setting-label"),
                RadioSet(
                    RadioButton("K-Means", value=self.app_settings.get("clustering_method") == "kmeans", id="kmeans"),
                    RadioButton("HDBSCAN", value=self.app_settings.get("clustering_method") == "hdbscan", id="hdbscan"),
                    id="clustering_method"
                ),
                
                # Number of results - using buttons instead of slider
                Label("Number of Results:", classes="setting-label"),
                Horizontal(
                    Button("5", variant="success" if self.app_settings.get("num_results") == 5 else "default", id="results_5"),
                    Button("10", variant="success" if self.app_settings.get("num_results") == 10 else "default", id="results_10"),
                    Button("25", variant="success" if self.app_settings.get("num_results") == 25 else "default", id="results_25"),
                    Button("50", variant="success" if self.app_settings.get("num_results") == 50 else "default", id="results_50"),
                    classes="results-buttons"
                ),
                
                # Enable debug mode
                Horizontal(
                    Label("Debug Mode:", classes="setting-label"),
                    Switch(value=self.app_settings.get("debug_mode", False), id="debug_switch"),
                ),
                
                # Show similarity scores
                Horizontal(
                    Label("Show Similarity Scores:", classes="setting-label"),
                    Switch(value=self.app_settings.get("show_scores", True), id="scores_switch"),
                ),
                
                # Buttons - Fixed container structure with better spacing
                Horizontal(
                    Button("Cancel", variant="default", id="cancel"),
                    Button("Apply", variant="success", id="apply"),
                    classes="button-group"
                ),
                
                classes="settings-container"
            ),
            id="settings_dialog"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss()
        elif event.button.id == "apply":
            # Collect settings
            clustering_radio = self.query_one("#clustering_method", RadioSet)
            clustering_method = clustering_radio.pressed_button.id if clustering_radio.pressed_button else "hdbscan"
            
            debug_mode = self.query_one("#debug_switch", Switch).value
            show_scores = self.query_one("#scores_switch", Switch).value
            
            # Update settings
            self.app_settings.update({
                "clustering_method": clustering_method,
                "num_results": self.app_settings.get("num_results", 5),  # Keep current value
                "debug_mode": debug_mode,
                "show_scores": show_scores
            })
            
            if debug_mode:
                print(f"Settings applied: {self.app_settings}")
            
            self.dismiss(self.app_settings)
        
        # Handle results count buttons
        elif event.button.id.startswith("results_"):
            num_results = int(event.button.id.split("_")[1])
            self.app_settings["num_results"] = num_results
            self._update_results_buttons(event.button.id)
    
    def _update_results_buttons(self, active_id: str):
        """Update the styling of results count buttons"""
        for button_id in ["results_5", "results_10", "results_25", "results_50"]:
            try:
                button = self.query_one(f"#{button_id}", Button)
                if button_id == active_id:
                    button.variant = "success"
                else:
                    button.variant = "default"
            except:
                pass  # Button might not exist yet

class SearchEngineFrontend(App):
    CSS_PATH = Path(__file__).parent / "style.css"
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("s", "show_settings", "Show settings"),
        ("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        # App settings with defaults
        self.settings = {
            "clustering_method": "hdbscan",
            "num_results": 5,
            "debug_mode": False,
            "show_scores": True
        }
        
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield SearchBar()
        yield VerticalScroll(id="results_container")

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light")

    def action_show_settings(self) -> None:
        """Show the settings modal"""
        def handle_settings_result(settings: dict) -> None:
            if settings:  # Only update if user clicked Apply
                self.settings = settings
                if self.settings.get("debug_mode"):
                    print(f"Settings updated: {self.settings}")
        
        self.push_screen(SettingsScreen(self.settings.copy()), handle_settings_result)

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
                similarity_score=entry.get("similarity_score", 0.0),
                show_scores=self.settings.get("show_scores", True)
            )
            results_container.mount(rf)

    def on_search_triggered(self, message: SearchTriggered) -> None:
        """Handle search trigger and apply reranking using the_function"""
        if not message.query.strip():
            return

        method = self.settings.get("clustering_method", "hdbscan")
        num_results = self.settings.get("num_results", 5)
        debug_mode = self.settings.get("debug_mode", False)
        
        if debug_mode:
            print(f"Searching with query: '{message.query}' using approach: {message.approach}")
            print(f"Method: {method}, Results: {num_results}")
        
        if message.approach == "dynamic":
            try:
                results = the_function(query=message.query, k=num_results, method=method)
            except Exception as e:
                self.clear_results()
                results_container: Widget = self.query_one("#results_container")
                error_field = Static(f"Error: {str(e)}", classes="error-message")
                results_container.mount(error_field)
                return
        
        elif message.approach == "static":
            try:
                results = sbert_static_search(query=message.query, num_docs_to_retrieve=num_results)
            except Exception as e:
                self.clear_results()
                results_container: Widget = self.query_one("#results_container")
                error_field = Static(f"Error: {str(e)}", classes="error-message")
                results_container.mount(error_field)
                return
            
        results_df = results
        self.populate_results_from_dict(results_df)
        
    def clear_results(self) -> None:
        """Clear all results from the UI"""
        results_container: Widget = self.query_one("#results_container")
        for child in results_container.children:
            child.remove()

LINES = """dynamic
static""".splitlines()

class SearchBar(HorizontalGroup):
    def __init__(self):
        super().__init__()
        self.selected_approach = "dynamic"  # Fixed variable name
        
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter search term...", id="search_input", classes="search_input")
        yield Button("Search 🔎", id="start", variant="success")
        yield Select(((line, line) for line in LINES), value="dynamic", id="approach_select")
        yield Button("Set ⚙️", id="settings", variant="primary", classes="settings-button")
        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            query_input = self.query_one("#search_input", Input)
            # Fixed variable name
            self.post_message(SearchTriggered(self, query_input.value, self.selected_approach))
        elif event.button.id == "settings":
            app = self.app
            app.action_show_settings()
            
       
    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        # Fixed variable name
        self.selected_approach = str(event.value)
        app = self.app
        if app.settings.get("debug_mode"):
            print(f"Approach selection changed to: {self.selected_approach}")

class ResultField(Vertical):
    def __init__(self, original_ranking: int, new_ranking: int, cluster: str, doc_id: str, label: str, text: str, similarity_score: float = 0.0, show_scores: bool = True) -> None:
        super().__init__()
        self.original_ranking = original_ranking
        self.new_ranking = new_ranking
        self.cluster = cluster
        self.doc_id = doc_id
        self.label = label
        self.text = text or "No text available"
        self.similarity_score = similarity_score
        self.show_scores = show_scores

    def compose(self):
        # Create header with ranking information
        header_text = f"#{self.new_ranking} {self.label}"
        if self.new_ranking != self.original_ranking:
            header_text += f" (was #{self.original_ranking})"
        header_text += f" | Cluster: {self.cluster} | ID: {self.doc_id}"
        
        # Only show similarity score if setting is enabled
        if self.show_scores and self.similarity_score > 0:
            header_text += f" | Score: {self.similarity_score:.4f}"

        # Truncate very long text for display
        display_text = self.text[:500] + "..." if len(self.text) > 500 else self.text

        yield Container(
            Vertical(
                Label(header_text, classes="result-header"),
                Static(display_text, classes="result-body"),
            ),
            classes="result-frame"
        )