import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]

class Shape:
    """
    Base class for shapes defined via an implicit signed distance or level set function.

    Convention: φ(x,y) < 0 inside solid, φ > 0 in fluid, φ = 0 on boundary.
    """

    def sdf(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """
        Signed distance or level set function.

        Parameters
        ----------
        x, y : float or np.ndarray
            Coordinates to evaluate.

        Returns
        -------
        φ : np.ndarray
            Negative inside solid, positive outside, zero on boundary.
        """
        raise NotImplementedError("sdf() must be implemented by subclasses.")

    def contains(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Boolean mask where True indicates inside or on the solid."""
        return self.sdf(x, y) <= 0

    def rotate(self, theta: float):
        """Return a rotated view of this shape by theta radians (about shape's reference)."""
        from .ops import RotatedShape
        return RotatedShape(self, theta)
