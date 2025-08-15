# lb2dgeom

Analytic 2D geometry generator for Lattice Boltzmann Method (LBM) simulations.
It provides parametric shapes, boolean operations, signed distance fields, and
Bouzidi link-fraction calculations for the D2Q9 model.

## Features

**lb2dgeom** covers the entire workflow of preparing complex 2‑D boundaries for
LBM solvers:

- **Parametric shapes** – circle, ellipse, rectangle (axis aligned or rotated),
  rounded rectangle and Cassini oval.
- **Boolean operations** – union, intersection and difference are implemented
  using standard signed–distance blending rules.
- **Signed distance fields** – shapes provide analytic `sdf(x, y)` methods and
  are rasterised onto uniform Cartesian grids.
- **Bouzidi boundary fractions** – compute `q_i` link fractions for all
  D2Q9 lattice directions with physical length normalisation.
- **I/O helpers** – save and load geometry data sets in `.npz` format.
- **Visualisation utilities** – plot solid masks, signed distance fields,
  Bouzidi histograms and per‑direction `q_i` fields.

## Installation

### Conda environment

Create and activate the conda environment specified in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate lb2dgeom
```

### Editable install

For a lightweight development install simply run:

```bash
pip install -e .
```

## Quickstart

The snippet below demonstrates the typical work flow: build a grid, define a
shape, rasterise its signed distance field, compute Bouzidi coefficients and
produce diagnostic plots.

```python
import numpy as np
from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.grids import Grid
from lb2dgeom.io import save_npz
from lb2dgeom.raster import rasterize
from lb2dgeom.shapes.circle import Circle
from lb2dgeom import viz

# Grid and shape
g = Grid(nx=100, ny=80, dx=1.0, origin=(-50.0, -40.0))
shape = Circle(x0=0.0, y0=0.0, r=20.0)

# Rasterise and compute Bouzidi q_i
phi, solid = rasterize(g, shape)
bouzidi = compute_bouzidi(g, phi, solid)

# Save and plot
save_npz("circle_geom.npz", solid, phi, bouzidi)
viz.plot_phi(phi, "circle_phi.png")
viz.plot_bouzidi_dirs(bouzidi, "circle_bouzidi_dir")
```

See the `examples/` directory for complete scripts for an ellipse and a
Cassini oval. Running an example (e.g. `python examples/demo_cassini.py`)
produces geometry files and PNG diagnostics in `examples/output/`.

### Feature guide

- **Grid generation** – `Grid(nx, ny, dx, origin)` provides cell centres and
  spacing in physical units.
- **Shapes** – analytic `sdf` implementations live in `lb2dgeom.shapes` and
  share a common `Shape` base class. Shapes may be combined via boolean
  operations from `lb2dgeom.shapes.ops`.
- **Rasterisation** – `rasterize(grid, shape)` samples `phi` and returns the
  solid mask for LBM nodes.
- **Bouzidi coefficients** – `compute_bouzidi` locates boundary intersections
  for all lattice directions and returns an array of `q_i` fractions with
  `NaN` in cells without solid neighbours.
- **I/O** – `lb2dgeom.io.save_npz` and `load_npz` persist geometry arrays to
  disk.
- **Visualisation** – `lb2dgeom.viz` contains
  `plot_solid`, `plot_phi`, `plot_bouzidi_hist` and
  `plot_bouzidi_dirs` for inspecting geometry and boundary data.

### Boolean operations

The `lb2dgeom.shapes.ops` module provides `union`, `difference` and
`intersection` helpers for combining shapes. The snippet below joins a circle
and rectangle and renders the union mask:

```python
from lb2dgeom.grids import Grid
from lb2dgeom.raster import rasterize
from lb2dgeom.shapes.circle import Circle
from lb2dgeom.shapes.ops import difference, intersection, union
from lb2dgeom.shapes.rectangle import Rectangle
from lb2dgeom import viz

circle = Circle(x0=-10.0, y0=0.0, r=15.0)
rect = Rectangle(x0=10.0, y0=0.0, w=20.0, h=30.0)

u = union(circle, rect)
i = intersection(circle, rect)
d = difference(circle, rect)

g = Grid(nx=120, ny=80, dx=1.0, origin=(-60.0, -40.0))
phi_u, solid_u = rasterize(g, u)
viz.plot_solid(solid_u, "union.png")
```

## Running tests

After installing the package, run the test-suite to verify the installation:

```bash
pytest -q
```

## License

This project is released under the MIT License.
