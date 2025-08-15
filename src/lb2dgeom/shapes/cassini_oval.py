import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]

class CassiniOval(Shape):
    """
    Cassini oval defined by center (x0, y0), parameter a, focal half-distance c,
    and rotation theta (radians).

    For foci at (±c, 0) in local frame, the implicit equation is:
        f(x,y) = (X^2 + Y^2)^2 - 2*c^2*(X^2 - Y^2) + (c^4 - a^4) = 0
    Negative inside (depending on a, c), positive outside.

    One-loop if a > c, peanut/lemniscate shape if a ≈ c, two loops if c > a.
    """

    def __init__(self, x0: float, y0: float, a: float, c: float, theta: float = 0.0):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.a = float(a)
        self.c = float(c)
        self.theta = float(theta)
        self._cos = np.cos(-self.theta)
        self._sin = np.sin(-self.theta)

    def sdf(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        # Translate and rotate into local frame
        X = x - self.x0
        Y = y - self.y0
        if self.theta != 0.0:
            x_local = self._cos * X - self._sin * Y
            y_local = self._sin * X + self._cos * Y
        else:
            x_local = X
            y_local = Y
        X2 = x_local**2
        Y2 = y_local**2
        f_val = (X2 + Y2)**2 - 2*(self.c**2)*(X2 - Y2) + (self.c**4 - self.a**4)
        return f_val
