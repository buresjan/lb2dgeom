import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]


class CassiniOval(Shape):
    """Cassini oval with signed distance evaluation.

    Parameters
    ----------
    x0, y0 : float
        Center of the oval.
    a : float
        Constant product of distances to the two foci.
    c : float
        Half-distance between the foci along the ``x`` axis in the local frame.
    theta : float, optional
        Rotation angle in radians.

    Notes
    -----
    The :meth:`sdf` routine returns the Euclidean signed distance to the
    Cassini-oval boundary using a Newton projection. The shape may consist of
    one or two loops depending on the relation between ``a`` and ``c``.
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
        """Signed distance to the Cassini-oval boundary."""

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        X = x_arr - self.x0
        Y = y_arr - self.y0
        if self.theta != 0.0:
            x_local = self._cos * X - self._sin * Y
            y_local = self._sin * X + self._cos * Y
        else:
            x_local = X
            y_local = Y

        px = x_local.copy()
        py = y_local.copy()

        for _ in range(25):
            r2 = px * px + py * py
            f_val = (
                r2**2 - 2 * (self.c**2) * (px * px - py * py) + (self.c**4 - self.a**4)
            )
            gx = 4.0 * px * (r2 - self.c**2)
            gy = 4.0 * py * (r2 + self.c**2)
            denom = gx * gx + gy * gy + 1e-12
            px -= f_val * gx / denom
            py -= f_val * gy / denom

        dist = np.sqrt((px - x_local) ** 2 + (py - y_local) ** 2)
        initial = (
            (x_local**2 + y_local**2) ** 2
            - 2 * (self.c**2) * (x_local**2 - y_local**2)
            + (self.c**4 - self.a**4)
        )
        sign = np.sign(initial)

        if dist.ndim == 0:
            dist = float(dist)
            sign = float(sign)
            if dist == 0.0:
                if sign < 0 and self.a > self.c:
                    dist = np.sqrt(self.a**2 - self.c**2)
                elif sign > 0 and self.c > self.a:
                    dist = np.sqrt(self.c**2 - self.a**2)
            return sign * dist

        mask_inside = (sign < 0) & (dist == 0.0) & (self.a > self.c)
        mask_outside = (sign > 0) & (dist == 0.0) & (self.c > self.a)
        if np.any(mask_inside) or np.any(mask_outside):
            dist = dist + 0.0
            if np.any(mask_inside):
                dist[mask_inside] = np.sqrt(self.a**2 - self.c**2)
            if np.any(mask_outside):
                dist[mask_outside] = np.sqrt(self.c**2 - self.a**2)
        return sign * dist
