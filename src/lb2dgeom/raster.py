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


def classify_cells(
    solid: np.ndarray,
    *,
    fluid_code: int = 0,
    near_wall_code: int = 1,
    wall_code: int = 2,
) -> np.ndarray:
    """
    Classify cells as fluid, near-wall fluid, or wall (solid).

    The "near-wall" class includes fluid cells that are 8-connected neighbors
    of any wall (solid) cell, i.e. adjacency includes diagonals.

    Parameters
    ----------
    solid : np.ndarray
        Binary mask where 1 indicates wall/solid cells and 0 indicates fluid.
    fluid_code : int, optional
        Integer code to use for fluid cells. Defaults to ``0``.
    near_wall_code : int, optional
        Integer code to use for near-wall fluid cells. Defaults to ``1``.
    wall_code : int, optional
        Integer code to use for wall/solid cells. Defaults to ``2``.

    Returns
    -------
    cell_types : np.ndarray
        Integer array of the same shape as ``solid`` with values from
        ``{fluid_code, near_wall_code, wall_code}``.
    """
    if solid.ndim != 2:
        raise ValueError("solid mask must be 2D")

    ny, nx = solid.shape
    solid_bool = solid.astype(bool)

    # Build an 8-neighborhood dilation of the solid mask via logical shifts
    nb = np.zeros_like(solid_bool)
    # Cardinal neighbors
    nb[1:, :] |= solid_bool[:-1, :]   # north
    nb[:-1, :] |= solid_bool[1:, :]   # south
    nb[:, 1:] |= solid_bool[:, :-1]   # west
    nb[:, :-1] |= solid_bool[:, 1:]   # east
    # Diagonal neighbors
    nb[1:, 1:] |= solid_bool[:-1, :-1]     # northwest
    nb[1:, :-1] |= solid_bool[:-1, 1:]     # northeast
    nb[:-1, 1:] |= solid_bool[1:, :-1]     # southwest
    nb[:-1, :-1] |= solid_bool[1:, 1:]     # southeast

    fluid = ~solid_bool
    near_wall = fluid & nb

    cell_types = np.full((ny, nx), fluid_code, dtype=np.int8)
    cell_types[near_wall] = near_wall_code
    cell_types[solid_bool] = wall_code
    return cell_types
