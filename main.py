from pathlib import Path
import subprocess
import sys

# Some useful paths
script_path = Path(__file__).resolve()
script_dir = script_path.parent
frontend_path = script_dir / "src" / "frontend" / "frontend.py"

if __name__ == "__main__":
    subprocess.run([sys.executable, str(frontend_path)])
