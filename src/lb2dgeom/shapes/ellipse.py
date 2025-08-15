import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]


class Ellipse(Shape):
    """Ellipse shape with true signed distance evaluation.

    Parameters
    ----------
    x0, y0 : float
        Center of the ellipse.
    a, b : float
        Semi-axes in ``x`` and ``y`` directions of the unrotated ellipse.
    theta : float, optional
        Rotation angle in radians.

    Notes
    -----
    The :meth:`sdf` method computes the Euclidean signed distance to the
    ellipse boundary via a Newton iteration. Negative values indicate points
    inside the ellipse, positive values outside.
    """

    def __init__(self, x0: float, y0: float, a: float, b: float, theta: float = 0.0):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.a = float(a)
        self.b = float(b)
        self.theta = float(theta)
        # Precompute rotation to local frame
        self._cos = np.cos(-self.theta)
        self._sin = np.sin(-self.theta)

    def sdf(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Signed distance to the ellipse boundary."""

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        X = x_arr - self.x0
        Y = y_arr - self.y0
        x_local = self._cos * X - self._sin * Y
        y_local = self._sin * X + self._cos * Y

        px = x_local.copy()
        py = y_local.copy()

        for _ in range(25):
            f_val = (px / self.a) ** 2 + (py / self.b) ** 2 - 1.0
            gx = 2.0 * px / (self.a**2)
            gy = 2.0 * py / (self.b**2)
            denom = gx * gx + gy * gy + 1e-12
            px -= f_val * gx / denom
            py -= f_val * gy / denom

        dist = np.sqrt((px - x_local) ** 2 + (py - y_local) ** 2)
        sign = np.sign((x_local / self.a) ** 2 + (y_local / self.b) ** 2 - 1.0)

        if np.isscalar(dist):
            if sign < 0 and dist == 0.0:
                dist = min(self.a, self.b)
            return sign * dist

        mask = (sign < 0) & (dist == 0.0)
        if np.any(mask):
            dist = dist + 0.0  # ensure copy
            dist[mask] = min(self.a, self.b)
        return sign * dist
