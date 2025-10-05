from typing import Optional

import numpy as np

from .d2q9 import E, E_LENGTHS
from .grids import Grid


def compute_bouzidi(
    grid: Grid,
    phi: np.ndarray,
    solid: np.ndarray,
    *,
    phi_threshold: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> np.ndarray:
    """
    Compute Bouzidi q_i fractions for D2Q9 model.

    Parameters
    ----------
    grid : Grid
        The grid specification.
    phi : np.ndarray
        Signed distance field (negative in solid).
    solid : np.ndarray
        Solid mask (1=solid, 0=fluid).
    phi_threshold : float, optional
        Level-set value corresponding to the solid/fluid interface. This should
        match the ``threshold`` used in :func:`lb2dgeom.raster.rasterize`. When
        ``None`` (default), the value is inferred from ``phi`` by taking the
        maximum signed-distance value observed inside solid cells. This allows
        Bouzidi link fractions to remain valid even when the solid mask is
        dilated or eroded via non-zero thresholds.
    tol : float
        Relative tolerance for boundary intersection root-finding.
    max_iter : int
        Maximum iterations for bisection.

    Returns
    -------
    bouzidi : np.ndarray of shape (ny, nx, 9), dtype=float32
        q_i fractions. NaN where no boundary link.
    """
    if phi.ndim != 2:
        raise ValueError("phi must be a 2D array")
    if solid.ndim != 2:
        raise ValueError("solid mask must be 2D")
    if phi.shape != solid.shape:
        raise ValueError("phi and solid must share the same shape")
    expected_shape = (grid.ny, grid.nx)
    if phi.shape != expected_shape:
        raise ValueError(
            "phi/solid shape must match the Grid dimensions (ny, nx)"
        )

    ny, nx = solid.shape
    dx = grid.dx
    bouzidi = np.full((ny, nx, 9), np.nan, dtype=np.float32)

    solid_mask = solid.astype(bool)
    if phi_threshold is None:
        inferred_threshold = 0.0
        if np.any(solid_mask):
            solid_values = phi[solid_mask]
            finite_vals = solid_values[np.isfinite(solid_values)]
            if finite_vals.size:
                solid_max = float(np.max(finite_vals))
                inferred_threshold = solid_max
                fluid_mask = ~solid_mask
                fluid_values = phi[fluid_mask]
                fluid_finite = fluid_values[np.isfinite(fluid_values)]
                if (
                    solid_max < 0.0
                    and fluid_finite.size
                    and float(np.min(fluid_finite)) > 0.0
                ):
                    inferred_threshold = 0.0
        threshold = inferred_threshold
    else:
        threshold = float(phi_threshold)

    Xc, Yc = grid.coords()

    for y in range(ny):
        for x in range(nx):
            if solid[y, x]:
                continue  # Only fluid nodes processed
            for i in range(1, 9):  # skip rest velocity
                ex, ey = E[i]
                nxn = x + ex
                nyn = y + ey
                # Skip if neighbor out of bounds
                if nxn < 0 or nxn >= nx or nyn < 0 or nyn >= ny:
                    continue
                # Check neighbor: if fluid-fluid, no boundary link
                if not solid[nyn, nxn]:
                    continue
                # Bracket along ray from current cell center toward neighbor
                L = E_LENGTHS[i] * dx
                xf = Xc[y, x]
                yf = Yc[y, x]
                phi_f = phi[y, x]
                phi_b = phi[nyn, nxn]
                phi_f_adj = phi_f - threshold
                phi_b_adj = phi_b - threshold
                # Ensure bracket: fluid strictly positive, solid strictly negative
                if (
                    phi_f_adj <= 0
                    or phi_b_adj >= 0
                    or np.isclose(phi_f_adj, 0.0, atol=tol)
                    or np.isclose(phi_b_adj, 0.0, atol=tol)
                ):
                    if np.isclose(phi_f_adj, 0.0, atol=tol) and phi_b_adj < 0:
                        bouzidi[y, x, i] = 0.0
                    elif np.isclose(phi_b_adj, 0.0, atol=tol) and phi_f_adj > 0:
                        bouzidi[y, x, i] = 1.0
                    continue
                s0 = 0.0
                s1 = L
                encountered_nan = False
                inv_len = 1.0 / E_LENGTHS[i]
                for _ in range(max_iter):
                    sm = 0.5 * (s0 + s1)
                    xm = xf + ex * sm * inv_len
                    ym = yf + ey * sm * inv_len
                    phi_m = float(interp_phi(xm, ym, grid, phi))
                    if np.isnan(phi_m):
                        encountered_nan = True
                        break
                    phi_m_adj = phi_m - threshold
                    if phi_m_adj > 0:
                        s0 = sm
                    else:
                        s1 = sm
                    if abs(s1 - s0) < tol * dx:
                        break
                if encountered_nan:
                    continue
                d_wall = s1
                q_i = d_wall / L
                bouzidi[y, x, i] = q_i
    return bouzidi


def interp_phi(x: float, y: float, grid: Grid, phi: np.ndarray) -> float:
    """
    Bilinear interpolation of phi at physical coords (x,y).
    """
    nx = grid.nx
    ny = grid.ny
    dx = grid.dx
    ox, oy = grid.origin
    gx = (x - ox) / dx
    gy = (y - oy) / dx
    ix = int(np.floor(gx))
    iy = int(np.floor(gy))
    if ix < 0 or ix >= nx - 1 or iy < 0 or iy >= ny - 1:
        return np.nan
    fx = gx - ix
    fy = gy - iy
    p00 = phi[iy, ix]
    p10 = phi[iy, ix + 1]
    p01 = phi[iy + 1, ix]
    p11 = phi[iy + 1, ix + 1]
    return (
        p00 * (1 - fx) * (1 - fy)
        + p10 * fx * (1 - fy)
        + p01 * (1 - fx) * fy
        + p11 * fx * fy
    )
