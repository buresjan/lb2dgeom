import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]

class Rectangle(Shape):
    """
    Axis-aligned or rotated rectangle defined by center (x0, y0),
    width w, height h, and rotation theta (radians).

    SDF formula (axis-aligned frame):
        Let hw = w/2, hh = h/2.
        dx = |x_local| - hw
        dy = |y_local| - hh
        outside_dist = sqrt(max(dx,0)^2 + max(dy,0)^2)
        inside_dist = min(max(dx, dy), 0)
        φ = outside_dist + inside_dist

    Negative inside, zero on edges, positive outside.
    """

    def __init__(self, x0: float, y0: float, w: float, h: float, theta: float = 0.0):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.w = float(w)
        self.h = float(h)
        self.theta = float(theta)
        self._cos = np.cos(-self.theta)
        self._sin = np.sin(-self.theta)

    def sdf(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        # Translate and rotate to rectangle's local frame
        X = x - self.x0
        Y = y - self.y0
        if self.theta != 0.0:
            x_local = self._cos * X - self._sin * Y
            y_local = self._sin * X + self._cos * Y
        else:
            x_local = X
            y_local = Y
        hw = 0.5 * self.w
        hh = 0.5 * self.h
        dx = np.abs(x_local) - hw
        dy = np.abs(y_local) - hh
        outside_dx = np.maximum(dx, 0.0)
        outside_dy = np.maximum(dy, 0.0)
        outside_dist = np.sqrt(outside_dx**2 + outside_dy**2)
        inside_dist = np.minimum(np.maximum(dx, dy), 0.0)
        return outside_dist + inside_dist
