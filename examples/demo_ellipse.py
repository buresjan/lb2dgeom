import os
import numpy as np
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.grids import Grid
from lb2dgeom.io import save_npz, save_txt
from lb2dgeom.raster import rasterize, classify_cells
from lb2dgeom.shapes.ellipse import Ellipse
from lb2dgeom import viz

if __name__ == "__main__":
    # Grid setup
    g = Grid(nx=100, ny=80, dx=1.0, origin=(-50.0, -40.0))

    # Define a rotated ellipse
    shape = Ellipse(x0=0.0, y0=0.0, a=20.0, b=10.0, theta=np.pi / 6)

    # Rasterize
    phi, solid = rasterize(g, shape)

    # Bouzidi
    bouzidi = compute_bouzidi(g, phi, solid)

    # Cell-type encoding (0=fluid, 1=near-wall, 2=wall)
    cell_types = classify_cells(solid)

    # Save arrays
    out_dir = os.path.join("examples", "output")
    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        os.path.join(out_dir, "ellipse_geom.npz"), solid, phi, bouzidi, cell_types=cell_types
    )

    # Diagnostics
    viz.plot_solid(solid, "ellipse_solid.png", show=False, out_dir=out_dir)
    viz.plot_phi(phi, "ellipse_phi.png", levels=30, show=False, out_dir=out_dir)
    viz.plot_bouzidi_hist(
        bouzidi, "ellipse_bouzidi_hist.png", show=False, out_dir=out_dir
    )
    viz.plot_cell_types(cell_types, "ellipse_cell_types.png", show=False, out_dir=out_dir)

    print("Demo ellipse complete. Outputs saved in examples/output/")

    # Optional: export as whitespace-delimited text (x y type q1..q8)
    save_txt(
        os.path.join(out_dir, "ellipse_geom.txt"),
        g,
        cell_types,
        bouzidi,
        selection="all",
        include_header=True,
    )
