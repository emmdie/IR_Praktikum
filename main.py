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

def run_textual_app_from_path(file_path: str, class_name: str = "App"):
    spec = importlib.util.spec_from_file_location("dynamic_textual_app", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_textual_app"] = module
    spec.loader.exec_module(module)
    app_class = getattr(module, class_name)
    if not issubclass(app_class, App):
        raise TypeError(f"{class_name} is not a subclass of textual.app.App")
    app = app_class()
    app.run()

if __name__ == "__main__":
    app = SearchEngineFrontend()
    app.run()
