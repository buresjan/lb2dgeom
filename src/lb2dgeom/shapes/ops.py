import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]

class RotatedShape(Shape):
    """
    Wrapper that rotates an underlying shape by a given angle around a pivot point.
    """

    def __init__(self, shape: Shape, theta: float, origin=None):
        self.shape = shape
        self.theta = float(theta)
        if origin is None:
            self.cx = getattr(shape, "x0", 0.0)
            self.cy = getattr(shape, "y0", 0.0)
        else:
            self.cx, self.cy = origin
        # Precompute rotation for query transform
        self._cos = np.cos(-self.theta)
        self._sin = np.sin(-self.theta)

    def sdf(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        # Translate to pivot, rotate by -theta, translate back
        X = x - self.cx
        Y = y - self.cy
        x_rot = self._cos * X - self._sin * Y + self.cx
        y_rot = self._sin * X + self._cos * Y + self.cy
        return self.shape.sdf(x_rot, y_rot)

class Union(Shape):
    """Union of two shapes: inside if either shape is inside."""
    def __init__(self, shape1: Shape, shape2: Shape):
        self.s1 = shape1
        self.s2 = shape2

    def sdf(self, x: ArrayLike, y: ArrayLike):
        return np.minimum(self.s1.sdf(x, y), self.s2.sdf(x, y))


class Intersection(Shape):
    """Intersection of two shapes: inside if both shapes are inside."""
    def __init__(self, shape1: Shape, shape2: Shape):
        self.s1 = shape1
        self.s2 = shape2

    def sdf(self, x: ArrayLike, y: ArrayLike):
        return np.maximum(self.s1.sdf(x, y), self.s2.sdf(x, y))


class Difference(Shape):
    """Difference of two shapes: inside if inside shape1 and outside shape2."""
    def __init__(self, shape1: Shape, shape2: Shape):
        self.s1 = shape1
        self.s2 = shape2

    def sdf(self, x: ArrayLike, y: ArrayLike):
        return np.maximum(self.s1.sdf(x, y), -self.s2.sdf(x, y))
