"""Boolean operations demonstration script."""

import os
import numpy as np

from lb2dgeom import viz
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.grids import Grid
from lb2dgeom.io import save_npz
from lb2dgeom.raster import rasterize
from lb2dgeom.shapes.circle import Circle
from lb2dgeom.shapes.rectangle import Rectangle
from lb2dgeom.shapes.ops import Union, Difference


if __name__ == "__main__":
    # Grid setup
    g = Grid(nx=120, ny=80, dx=1.0, origin=(-60.0, -40.0))

    # --- Union: circle and rectangle ---
    circle_u = Circle(x0=-15.0, y0=0.0, r=20.0)
    rect_u = Rectangle(x0=15.0, y0=0.0, w=30.0, h=20.0)
    shape_union = Union(circle_u, rect_u)

    # Rasterize union
    phi_u, solid_u = rasterize(g, shape_union)

    # Bouzidi coefficients for union
    bouzidi_u = compute_bouzidi(g, phi_u, solid_u)

    # Save arrays for union
    out_dir = os.path.join("examples", "output")
    os.makedirs(out_dir, exist_ok=True)
    save_npz(os.path.join(out_dir, "boolean_union_geom.npz"), solid_u, phi_u, bouzidi_u)

    # Diagnostics for union
    viz.plot_solid(solid_u, "boolean_union_solid.png", show=False)
    viz.plot_phi(phi_u, "boolean_union_phi.png", levels=30, show=False)
    viz.plot_bouzidi_hist(bouzidi_u, "boolean_union_bouzidi_hist.png", show=False)
    viz.plot_bouzidi_dirs(bouzidi_u, "boolean_union_bouzidi_dir", show=False)

    # --- Difference: circle minus rectangle ---
    circle_d = Circle(x0=0.0, y0=0.0, r=25.0)
    rect_d = Rectangle(x0=0.0, y0=0.0, w=20.0, h=20.0)
    shape_diff = Difference(circle_d, rect_d)

    # Rasterize difference
    phi_d, solid_d = rasterize(g, shape_diff)

    # Bouzidi coefficients for difference
    bouzidi_d = compute_bouzidi(g, phi_d, solid_d)

    # Save arrays for difference
    save_npz(os.path.join(out_dir, "boolean_diff_geom.npz"), solid_d, phi_d, bouzidi_d)

    # Diagnostics for difference
    viz.plot_solid(solid_d, "boolean_diff_solid.png", show=False)
    viz.plot_phi(phi_d, "boolean_diff_phi.png", levels=30, show=False)
    viz.plot_bouzidi_hist(bouzidi_d, "boolean_diff_bouzidi_hist.png", show=False)
    viz.plot_bouzidi_dirs(bouzidi_d, "boolean_diff_bouzidi_dir", show=False)

    print("Boolean ops demo complete. Outputs saved in examples/output/")
