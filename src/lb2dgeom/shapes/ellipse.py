import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]

class Ellipse(Shape):
    """
    Ellipse defined by center (x0, y0), semi-axes a (x-dir) and b (y-dir),
    and rotation angle theta (radians).

    Implicit level set function:
        f(x,y) = (x'/a)^2 + (y'/b)^2 - 1
    where (x',y') are coordinates in the ellipse's local (unrotated) frame.

    Negative inside, zero on boundary, positive outside.
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
        # Translate point to ellipse center
        X = x - self.x0
        Y = y - self.y0
        # Rotate by -theta into local frame
        x_local = self._cos * X - self._sin * Y
        y_local = self._sin * X + self._cos * Y
        # Implicit function (not true distance, but sign-correct)
        return (x_local / self.a) ** 2 + (y_local / self.b) ** 2 - 1.0
