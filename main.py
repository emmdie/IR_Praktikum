from pathlib import Path
import sys
from textual.app import App
import importlib.util

# Some useful paths
script_path = Path(__file__).resolve()
repo_root = script_path.parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))  # Arkane Magie um src/frontend als package einzubinden

from frontend.frontend import SearchEngineFrontend

if __name__ == "__main__":
    app = SearchEngineFrontend()
    app.run()
