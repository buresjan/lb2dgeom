import numpy as np
import os
from lbmgeom.grids import Grid
from lbmgeom.shapes.ellipse import Ellipse
from lbmgeom.raster import rasterize
from lbmgeom.bouzidi import compute_bouzidi
from lbmgeom.io import save_npz
from lbmgeom import viz

if __name__ == "__main__":
    # Grid setup
    g = Grid(nx=100, ny=80, dx=1.0, origin=(-50.0, -40.0))

    # Define a rotated ellipse
    shape = Ellipse(x0=0.0, y0=0.0, a=20.0, b=10.0, theta=np.pi/6)

    # Rasterize
    phi, solid = rasterize(g, shape)

    # Bouzidi
    bouzidi = compute_bouzidi(g, phi, solid)

    # Save arrays
    out_dir = os.path.join("examples", "output")
    os.makedirs(out_dir, exist_ok=True)
    save_npz(os.path.join(out_dir, "ellipse_geom.npz"), solid, phi, bouzidi)

    # Diagnostics
    viz.plot_solid(solid, "ellipse_solid.png", show=False)
    viz.plot_phi(phi, "ellipse_phi.png", levels=30, show=False)
    viz.plot_bouzidi_hist(bouzidi, "ellipse_bouzidi_hist.png", show=False)

    print("Demo ellipse complete. Outputs saved in examples/output/")
