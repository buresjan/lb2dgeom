import numpy as np
from typing import Union
from .base import Shape

ArrayLike = Union[float, np.ndarray]


class RoundedRect(Shape):
    r"""Rounded rectangle with potentially different x/y corner radii.

    Parameters
    ----------
    x0, y0 : float
        Center of the rectangle.
    w, h : float
        Width and height of the rectangle.
    rx : float
        Corner radius along the x-direction.
    ry : float, optional
        Corner radius along the y-direction. If not provided, ``ry`` equals
        ``rx``.
    theta : float, optional
        Rotation angle in radians.

    Notes
    -----
    The signed distance function is computed by shrinking the rectangle by
    ``(rx, ry)`` along the respective axes and subtracting these radii from the
    local coordinates. The corners are therefore treated as quarter ellipses
    with radii ``rx`` and ``ry``. For local coordinates ``(x_l, y_l)`` the
    distance is

    .. math::

       hw_0 = \max(w/2 - r_x, 0)\\
       hh_0 = \max(h/2 - r_y, 0)\\
       d_x = |x_l| - hw_0 - r_x\\
       d_y = |y_l| - hh_0 - r_y\\
       \phi = \operatorname{hypot}(\max(d_x,0),\max(d_y,0))
              + \min(\max(d_x,d_y),0)
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        w: float,
        h: float,
        rx: float,
        ry: float = None,
        theta: float = 0.0,
    ):
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
        dx = np.abs(x_local) - hw0 - self.rx
        dy = np.abs(y_local) - hh0 - self.ry
        outside_dist = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
        inside_dist = np.minimum(np.maximum(dx, dy), 0.0)
        return outside_dist + inside_dist
