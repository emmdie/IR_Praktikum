from typing import Dict, List
import pandas as pd

from textual.widget import Widget
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Button, Digits, Input, Label, Static
from textual.containers import HorizontalGroup, VerticalScroll, Vertical, Horizontal, Container
from textual.reactive import reactive
from rich.text import Text
from textual.message import Message

class SearchTriggered(Message):
    def __init__(self, sender: Widget, query: str) -> None:
        super().__init__()
        self.sender = sender
        self.query = query

class SearchEngineFrontend(App):
    CSS_PATH = "style.css"
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Footer()
        yield SearchBar()
        yield VerticalScroll(id="results_container")

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light")

    def populate_results_from_dict(self, result_list: List[Dict]) -> None:
        self.clear_results()
        results_container: Widget = self.query_one("#results_container")
        top_ten = result_list[:10]

        for entry in top_ten:
            rf = ResultField(
                ranking=entry["init_ranking"],
                cluster=str(entry["cluster"]),
                doc_id=str(entry["doc_id"]),
                text=entry["text"]
            )
            results_container.mount(rf)
    

    def generate_fake_results(self) -> List[Dict]:
        return [
            {"doc_id": 182348321, "init_ranking": 55, "category": "Ohrenschmalzentferner", "cluster": 6, "text": "Der Ohrenschmalzentferner schläft nie"},
            {"doc_id": 182348322, "init_ranking": 48, "category": "Kaffeemaschine", "cluster": 2, "text": "Die Kaffeemaschine ist kaputt"},
            {"doc_id": 1823245322, "init_ranking": 13, "category": "Jaguar", "cluster": 2, "text": "Der Jaguar hat den Tofu erlegt"},
            {"doc_id": 1568756722, "init_ranking": 38, "category": "Car", "cluster": 2, "text": "Das Auto hat keinen Tofu erlegt"},
            {"doc_id": 1828857682, "init_ranking": 28, "category": "puma", "cluster": 2, "text": "Eine Schuhmarke, hat nichts mit Jaguaren zu tun"},
            {"doc_id": 1823999992, "init_ranking": 48, "category": "Ksafd", "cluster": 2, "text": "Ein zufälliger String, es scheint keine Naheliegende Verbindung zu "},
 
            ]

    def on_search_triggered(self, message: SearchTriggered) -> None:
        fake_results = self.generate_fake_results()
        self.populate_results_from_dict(fake_results)
   
    def clear_results(self) -> None:
        results_container: Widget = self.query_one("#results_container")
        for child in results_container.children:
            child.remove()
        
class SearchBar(HorizontalGroup):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter search term...", id="search_input", classes="search_input")
        yield Button("Search 🔎", id="start", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            query_input = self.query_one("#search_input", Input)
            self.post_message(SearchTriggered(self, query_input.value))

class ResultField(Vertical):
    def __init__(self, ranking: int, cluster: str, doc_id: str, text: str) -> None:
        super().__init__()
        self.ranking = ranking
        self.cluster = cluster
        self.doc_id = doc_id
        self.text = text

    def compose(self):
        yield Container(
            Vertical(
                Label(f"#{self.ranking} | Cluster: {self.cluster} | ID: {self.doc_id}", classes="result-header"),
                Static(Text.from_markup(self.text), classes="result-body"),
            ),
            classes="result-frame"
                        )
