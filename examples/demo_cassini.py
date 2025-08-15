"""Cassini oval demonstration script."""

import os
import numpy as np

from lb2dgeom import viz
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.grids import Grid
from lb2dgeom.io import save_npz
from lb2dgeom.raster import rasterize
from lb2dgeom.shapes.cassini_oval import CassiniOval


if __name__ == "__main__":
    # Grid setup
    g = Grid(nx=120, ny=120, dx=1.0, origin=(-60.0, -60.0))

    # Define a rotated Cassini oval
    shape = CassiniOval(x0=0.0, y0=0.0, a=25.0, c=15.0, theta=np.pi / 4)

    # Rasterize
    phi, solid = rasterize(g, shape)

    # Bouzidi coefficients
    bouzidi = compute_bouzidi(g, phi, solid)

    # Save arrays
    out_dir = os.path.join("examples", "output")
    os.makedirs(out_dir, exist_ok=True)
    save_npz(os.path.join(out_dir, "cassini_geom.npz"), solid, phi, bouzidi)

    # Diagnostics
    viz.plot_solid(solid, "cassini_solid.png", out_dir=out_dir, show=False)
    viz.plot_phi(phi, "cassini_phi.png", levels=30, out_dir=out_dir, show=False)
    viz.plot_bouzidi_hist(
        bouzidi, "cassini_bouzidi_hist.png", out_dir=out_dir, show=False
    )
    viz.plot_bouzidi_dirs(bouzidi, "cassini_bouzidi_dir", out_dir=out_dir, show=False)

    print("Demo Cassini oval complete. Outputs saved in examples/output/")
