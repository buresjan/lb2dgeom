import numpy as np
from typing import Tuple
from .d2q9 import E, E_LENGTHS
from .grids import Grid


def compute_bouzidi(
    grid: Grid,
    phi: np.ndarray,
    solid: np.ndarray,
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
    tol : float
        Relative tolerance for boundary intersection root-finding.
    max_iter : int
        Maximum iterations for bisection.

    Returns
    -------
    bouzidi : np.ndarray of shape (ny, nx, 9), dtype=float32
        q_i fractions. NaN where no boundary link.
    """
    ny, nx = solid.shape
    dx = grid.dx
    bouzidi = np.full((ny, nx, 9), np.nan, dtype=np.float32)

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
                # Ensure bracket: fluid positive, solid negative
                if phi_f < 0 or phi_b > 0:
                    continue
                s0 = 0.0
                s1 = L
                encountered_nan = False
                for _ in range(max_iter):
                    sm = 0.5 * (s0 + s1)
                    xm = xf + (ex / E_LENGTHS[i]) * sm
                    ym = yf + (ey / E_LENGTHS[i]) * sm
                    phi_m = float(interp_phi(xm, ym, grid, phi))
                    if np.isnan(phi_m):
                        encountered_nan = True
                        break
                    if phi_m > 0:
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
