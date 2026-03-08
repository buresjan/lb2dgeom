from functools import lru_cache
from typing import Tuple, Union

import numpy as np

from .base import Shape

ArrayLike = Union[float, np.ndarray]


@lru_cache(maxsize=None)
def _cassini_area_quadrature(
    num_theta: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cached angular quadrature data for Cassini-area evaluation."""
    if num_theta < 32:
        raise ValueError("num_theta must be at least 32")

    theta = np.linspace(0.0, 2.0 * np.pi, num_theta + 1, dtype=float)
    cos_2theta = np.cos(2.0 * theta)
    sin_2theta = np.sin(2.0 * theta)

    theta.setflags(write=False)
    cos_2theta.setflags(write=False)
    sin_2theta.setflags(write=False)
    return theta, cos_2theta, sin_2theta


def cassini_oval_area(a: float, b: float, *, num_theta: int = 8192) -> float:
    """Return the area of a Cassini oval in standard ``a``/``b`` notation.

    Parameters
    ----------
    a : float
        Half-distance between the two foci in the standard Cassini notation.
    b : float
        Distance-product parameter in the standard Cassini notation.
    num_theta : int, optional
        Number of angular intervals used by the numerical quadrature.
        Defaults to ``8192``.

    Returns
    -------
    float
        Total enclosed area of the Cassini oval.

    Notes
    -----
    The :class:`CassiniOval` class stores the same geometry as ``c=a`` and
    ``a=b``. This helper keeps the conventional notation for callers that
    specify the focal parameter and want to solve for the product constant.
    """
    a = float(a)
    b = float(b)
    if a < 0.0:
        raise ValueError("a must be non-negative")
    if b < 0.0:
        raise ValueError("b must be non-negative")
    if b == 0.0:
        return 0.0
    if a == 0.0:
        return float(np.pi * b * b)

    theta, cos_2theta, sin_2theta = _cassini_area_quadrature(num_theta)
    radicand = np.maximum(b**4 - a**4 * sin_2theta**2, 0.0)

    if b < a:
        return float(np.trapz(np.sqrt(radicand), theta))

    r2_outer = a * a * cos_2theta + np.sqrt(radicand)
    r2_outer = np.maximum(r2_outer, 0.0)
    return float(0.5 * np.trapz(r2_outer, theta))


def cassini_b_from_area(
    a: float,
    area: float,
    *,
    num_theta: int = 8192,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> float:
    """Solve the standard Cassini ``b`` parameter for a target area.

    Parameters
    ----------
    a : float
        Half-distance between the two foci in the standard Cassini notation.
    area : float
        Desired enclosed area of the Cassini oval.
    num_theta : int, optional
        Number of angular intervals used by the area quadrature.
        Defaults to ``8192``.
    tol : float, optional
        Absolute tolerance on the area residual. Defaults to ``1e-12``.
    max_iter : int, optional
        Maximum number of bisection iterations. Defaults to ``80``.

    Returns
    -------
    float
        The standard Cassini ``b`` parameter that matches ``area``.
    """
    a = float(a)
    area = float(area)
    if a < 0.0:
        raise ValueError("a must be non-negative")
    if area <= 0.0:
        raise ValueError("area must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    if a == 0.0:
        return float(np.sqrt(area / np.pi))

    lower = 0.0
    upper = max(a, np.sqrt(area / np.pi), 1e-12)
    for _ in range(max_iter):
        if cassini_oval_area(a, upper, num_theta=num_theta) >= area:
            break
        upper *= 2.0
    else:
        raise RuntimeError("failed to bracket Cassini b for the requested area")

    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        area_mid = cassini_oval_area(a, mid, num_theta=num_theta)
        if abs(area_mid - area) <= tol:
            return mid
        if area_mid < area:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


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
    Standard Cassini notation usually uses ``a`` for the half-focus spacing
    and ``b`` for the distance-product parameter. This class stores the same
    quantities as ``c`` and ``a``, respectively.
    """

    def __init__(self, x0: float, y0: float, a: float, c: float, theta: float = 0.0):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.a = float(a)
        self.c = float(c)
        self.theta = float(theta)
        self._cos = np.cos(-self.theta)
        self._sin = np.sin(-self.theta)

    @classmethod
    def from_standard_parameters(
        cls,
        x0: float,
        y0: float,
        a: float,
        b: float,
        theta: float = 0.0,
    ) -> "CassiniOval":
        """Construct a Cassini oval from the standard ``a``/``b`` notation."""
        return cls(x0=x0, y0=y0, a=b, c=a, theta=theta)

    @classmethod
    def from_standard_area(
        cls,
        x0: float,
        y0: float,
        a: float,
        area: float,
        theta: float = 0.0,
        *,
        num_theta: int = 8192,
        tol: float = 1e-12,
        max_iter: int = 80,
    ) -> "CassiniOval":
        """Construct a Cassini oval from standard ``a`` and a target area."""
        b = cassini_b_from_area(
            a,
            area,
            num_theta=num_theta,
            tol=tol,
            max_iter=max_iter,
        )
        return cls.from_standard_parameters(x0=x0, y0=y0, a=a, b=b, theta=theta)

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
