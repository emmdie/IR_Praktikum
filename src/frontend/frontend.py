from typing import Dict, List

from rich import color
from frontend import fake_results_generator, df_prepocessing
import pandas as pd

from textual import on
from textual.widget import Widget
from textual.widgets import Select
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Button, Input, Label, Static
from textual.containers import HorizontalGroup, VerticalScroll, Vertical, Container

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
        yield Header()
        yield Footer()
        yield SearchBar()
        yield VerticalScroll(id="results_container")

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light")

    def populate_results_from_df(self, df: pd.DataFrame) -> None:
        self.clear_results()
        results_container: Widget = self.query_one("#results_container")
        top_ten = df.head(10)

        for _, row in top_ten.iterrows():
            rf = ResultField(
                ranking=row["init_ranking"],
                cluster=str(row["cluster"]),
                doc_id=str(row["doc_id"]),
                text=row["text"],
                color=row["cluster_color"]
            )
            results_container.mount(rf)

    def on_search_triggered(self, message: SearchTriggered) -> None:
        results_df = fake_results_generator.generate_fake_results_df()
        results_df = df_prepocessing.assign_cluster_colors(results_df)
        self.populate_results_from_df(results_df)
   
    def clear_results(self) -> None:
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

class ResultField(Vertical):
    def __init__(self, ranking: int, cluster: str, doc_id: str, text: str, color: str) -> None:
        super().__init__()
        self.ranking = ranking
        self.cluster = cluster
        self.doc_id = doc_id
        self.text = text
        self.color = color

    def compose(self):
        container = Container(
            Vertical(
                Label(f"#{self.ranking} | Cluster: {self.cluster} | ID: {self.doc_id}", classes="result-header"),
                Static(Text.from_markup(self.text), classes="result-body"),
            ),
            classes="result-frame",
            )
        container.styles.border = ("round", self.color)
        #container.styles.padding = 1
        yield container
