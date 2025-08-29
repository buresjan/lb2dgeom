"""Convenience re-exports for :mod:`lb2dgeom`.

This module exposes the most commonly used classes, functions and
subpackages at the top level for easy access.  Users can simply import
``lb2dgeom`` and access :class:`Grid`, :func:`rasterize`,
:func:`compute_bouzidi`, as well as the :mod:`viz` and :mod:`shapes`
subpackages without needing to traverse the package hierarchy.
"""

from . import shapes, viz
from .bouzidi import compute_bouzidi
from .grids import Grid
from .raster import rasterize, classify_cells

__all__ = [
    "Grid",
    "rasterize",
    "classify_cells",
    "compute_bouzidi",
    "viz",
    "shapes",
]
