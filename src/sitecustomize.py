"""Project-wide Python runtime configuration.

This module is imported automatically when the Python interpreter starts.
It disables creation of Python bytecode caches and removes any
``__pycache__`` directories that might be created during execution.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
from pathlib import Path

# Prevent the interpreter from writing ``.pyc`` files.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

# Repository root directory (``src`` is the parent of ``sitecustomize.py``).
_ROOT = Path(__file__).resolve().parents[1]


def _cleanup_pycache() -> None:
    """Delete all ``__pycache__`` directories under the project root."""

    for path in _ROOT.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_pycache)
