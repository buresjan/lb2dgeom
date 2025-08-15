import numpy as np
from typing import Tuple


class Grid:
    """
    Uniform Cartesian grid for 2D domain. Provides cell-center coordinates.

    Attributes
    ----------
    nx : int
        Number of cells in x-direction.
    ny : int
        Number of cells in y-direction.
    dx : float
        Lattice spacing (cell size).
    origin : tuple[float, float]
        Physical coordinates of the cell at index (0,0).

    Methods
    -------
    coords() -> tuple[np.ndarray, np.ndarray]
        Returns X, Y coordinate arrays of shape (ny, nx) for cell centers.
    """

    def __init__(self, nx: int, ny: int, dx: float = 1.0, origin=(0.0, 0.0)):
        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive")
        if dx <= 0:
            raise ValueError("dx must be positive")

        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        ox, oy = origin
        self.origin = (float(ox), float(oy))

    def coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        X, Y : np.ndarray
            Meshgrid arrays of shape (ny, nx) containing the coordinates of cell centers.
        """
        ox, oy = self.origin
        x_coords = ox + np.arange(self.nx) * self.dx
        y_coords = oy + np.arange(self.ny) * self.dx
        return np.meshgrid(x_coords, y_coords, indexing="xy")
