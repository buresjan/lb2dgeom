import numpy as np
from typing import Tuple
from .grids import Grid
from .shapes.base import Shape

def rasterize(grid: Grid, shape: Shape, threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize a shape on the given grid.

    Parameters
    ----------
    grid : Grid
        Uniform grid object defining coordinates.
    shape : Shape
        Shape object with .sdf(x,y) method.
    threshold : float, default=0.0
        Boundary threshold; φ <= threshold is considered solid.

    Returns
    -------
    phi : np.ndarray of shape (ny, nx), dtype=float32
        Signed distance (or level set) field sampled at cell centers.
    solid : np.ndarray of shape (ny, nx), dtype=uint8
        1 for solid cells, 0 for fluid cells.
    """
    X, Y = grid.coords()
    phi = shape.sdf(X, Y).astype(np.float32)
    solid = (phi <= threshold).astype(np.uint8)
    return phi, solid
