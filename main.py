from pathlib import Path
import sys
from textual.app import App
import importlib.util

# Some useful paths
script_path = Path(__file__).resolve()
script_dir = script_path.parent
frontend_path = script_dir / "src" / "frontend" / "frontend.py"
    
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

def create_test_results():
    result_1 = {"doc_id": 123414, "init_ranking": 98, "category": "film", "cluster": 9, "text": "CSV files are the Comma Separated Files. It allows users to load tabular data into a DataFrame, which is a powerful structure for data manipulation and analysis. To access data from the CSV file, we require a function read_csv() from Pandas that retrieves data in the form of the data frame. Here’s a quick example to get you started. "}
    return [result_1]

if __name__ == "__main__":
    run_textual_app_from_path(frontend_path, class_name="SearchEngineFrontend")


