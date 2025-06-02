import pandas as pd

from textual.widget import Widget
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Button, Digits, Input, Label, Static
from textual.containers import HorizontalGroup, VerticalScroll, Vertical, Horizontal, Container
from textual.reactive import reactive
from rich.text import Text
from textual.message import Message

class SearchEngineFrontend(App):
    CSS_PATH = "style.css"
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Footer()
        yield SearchBar()
        yield VerticalScroll(
            ResultField(
                ranking=1,
                cluster="A",
                doc_id="doc_123",
                text="This is a long document content. It may span multiple lines.\nSecond line of content.\nThird line.\nFourth line.\nFifth line."
            ),
            ResultField(
                ranking=2,
                cluster="B",
                doc_id="doc_125",
                text="This is another very long document content. It may span multiple lines.\nSecond line of content.\nThird line.\nFourth line.\nFifth line."
            ),
            ResultField(
                ranking=2,
                cluster="B",
                doc_id="doc_125",
                text="This is yet another very long document content. Maybe talking about Jaguars. It may span multiple lines.\nSecond line of content.\nThird line.\nFourth line.\nFifth line."
            ),
            id="results_container"
        )

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light")

    def populate_results_from_df(self, df: pd.DataFrame) -> None:
        results_container: Widget = self.query_one("#results_container")
        results_container.clear()

        top_rows = df.head(10)

        for _, row in top_rows.iterrows():
            rf = ResultField(
                ranking=row["Original ranking"],
                cluster=str(row["Cluster ID"]),
                doc_id=str(row["Document ID"]),
                text=row["Text"]
            )
            results_container.mount(rf)

class SearchBar(HorizontalGroup):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter search term...", id="search_input", classes="search_input")
        yield Button("Search 🔎", id="start", variant="success")

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
