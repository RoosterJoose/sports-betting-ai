"""Project-root conftest for pytest.

Makes the project root importable as a path so `from src.data.fotmob import …`
works in any test file without each test doing its own `sys.path.insert(0, …)`.

This is the canonical fix for "ModuleNotFoundError: No module named 'src'"
when running `python -m pytest` from the project root. With this conftest
plus the `testpaths = ["tests"]` entry in pyproject.toml, no test file
should ever need to mutate sys.path.
"""
import sys
from pathlib import Path

# Project root = parent of the tests/ directory containing this conftest.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to sys.path so `src.*` imports resolve cleanly
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
