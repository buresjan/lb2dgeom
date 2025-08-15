import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]

class Circle(Shape):
    """
    Circle defined by center (x0, y0) and radius r.

    Signed distance function:
        φ(x,y) = sqrt((x - x0)^2 + (y - y0)^2) - r
    Negative inside, zero on boundary, positive outside.
    """
    def __init__(self, x0: float, y0: float, r: float):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.r = float(r)

    def sdf(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        dx = x - self.x0
        dy = y - self.y0
        return np.sqrt(dx*dx + dy*dy) - self.r
