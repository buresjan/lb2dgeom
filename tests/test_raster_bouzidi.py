import numpy as np
import pytest
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.grids import Grid
from lb2dgeom.raster import rasterize
from lb2dgeom.shapes.circle import Circle


def test_rasterize_circle_shape():
    g = Grid(11, 11, dx=1.0, origin=(-5, -5))
    shape = Circle(0, 0, 3.0)
    phi, solid = rasterize(g, shape)
    assert phi.shape == (11, 11)
    assert solid.shape == (11, 11)
    # Boundary cells should have phi ~ 0
    boundary_mask = np.abs(phi) < 1.0
    assert np.any(boundary_mask)
    # Inside solid should have phi < 0
    assert np.all(phi[solid == 1] <= 0.0)


def test_bouzidi_qi_range():
    g = Grid(21, 21, dx=1.0, origin=(-10, -10))
    shape = Circle(0, 0, 5.0)
    phi, solid = rasterize(g, shape)
    bouzidi = compute_bouzidi(g, phi, solid)
    assert bouzidi.shape == (21, 21, 9)
    # q_i should be between 0 and 1 for non-NaN entries
    mask = ~np.isnan(bouzidi)
    assert np.all((bouzidi[mask] >= 0) & (bouzidi[mask] <= 1))
    # Rest velocity direction should always be NaN
    assert np.all(np.isnan(bouzidi[:, :, 0]))


def test_bouzidi_handles_boundary_on_neighbor_center():
    g = Grid(21, 21, dx=1.0, origin=(-10, -10))
    shape = Circle(0, 0, 5.0)
    phi, solid = rasterize(g, shape)
    bouzidi = compute_bouzidi(g, phi, solid)
    y = int(0 - g.origin[1])
    x = int(6 - g.origin[0])  # Fluid cell at (6, 0)
    assert np.isclose(bouzidi[y, x, 3], 1.0)
