"""Test configuration for lb2dgeom.

This module adjusts ``sys.path`` so that the local ``src`` layout can be
imported without requiring ``pip install -e`` during testing.  It also forces
Matplotlib to use the non-interactive ``Agg`` backend for headless CI runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Ensure Matplotlib uses a backend that works without a display.
matplotlib.use("Agg", force=True)

# Add the repository's ``src`` directory to ``sys.path`` for local imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.is_dir():
    sys.path.insert(0, str(_SRC_PATH))

