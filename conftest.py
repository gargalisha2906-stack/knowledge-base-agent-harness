"""Make the kbharness package importable under bare `pytest` from the repo root.

`python -m pytest` puts the cwd on sys.path, but the `pytest` console script
does not - CI (and anyone cloning the repo) runs the console script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
