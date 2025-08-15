import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]

class RoundedRect(Shape):
    """
    Rounded rectangle defined by center (x0, y0), width w, height h,
    corner radius rx (and optional ry), and rotation theta (radians).

    We approximate the SDF by shrinking the rectangle by the radius
    and subtracting the radius from the distance field (circular corners).

    Formula:
        hw0 = max(w/2 - rx, 0)
        hh0 = max(h/2 - ry, 0)
        dx = |x_local| - hw0
        dy = |y_local| - hh0
        outside_dist = sqrt(max(dx,0)^2 + max(dy,0)^2)
        inside_dist = min(max(dx, dy), 0)
        φ = outside_dist + inside_dist - min(rx, ry)
    """

    def __init__(self, x0: float, y0: float, w: float, h: float,
                 rx: float, ry: float = None, theta: float = 0.0):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.w = float(w)
        self.h = float(h)
        self.rx = float(rx)
        self.ry = float(rx if ry is None else ry)
        self.theta = float(theta)
        # Cap radii so they don't exceed half-dimensions
        self.rx = min(self.rx, 0.5 * self.w)
        self.ry = min(self.ry, 0.5 * self.h)
        self._cos = np.cos(-self.theta)
        self._sin = np.sin(-self.theta)

    def sdf(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        # Translate and rotate to local frame
        X = x - self.x0
        Y = y - self.y0
        if self.theta != 0.0:
            x_local = self._cos * X - self._sin * Y
            y_local = self._sin * X + self._cos * Y
        else:
            x_local = X
            y_local = Y
        hw0 = max(0.5 * self.w - self.rx, 0.0)
        hh0 = max(0.5 * self.h - self.ry, 0.0)
        dx0 = np.abs(x_local) - hw0
        dy0 = np.abs(y_local) - hh0
        outside_dist0 = np.sqrt(np.maximum(dx0, 0.0)**2 + np.maximum(dy0, 0.0)**2)
        inside_dist0 = np.minimum(np.maximum(dx0, dy0), 0.0)
        return outside_dist0 + inside_dist0 - min(self.rx, self.ry)
