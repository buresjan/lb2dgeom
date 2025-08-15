# lb2dgeom

Analytic 2D geometry generator for Lattice Boltzmann Method (LBM) simulations.
It provides parametric shapes, boolean operations, signed distance fields, and
Bouzidi link-fraction calculations for the D2Q9 model.

## Features

- Parametric shapes: circle, ellipse, rectangle, rounded rectangle, Cassini oval
- Boolean composition: union, intersection, difference
- Accurate signed distance functions for voxelization and boundary location
- Bouzidi interpolated boundary link-fraction calculation for D2Q9
- I/O helpers for saving/loading `.npz` geometry datasets
- Visualization utilities for solid masks, φ field, and Bouzidi diagnostics

## Installation

### Conda environment

Create and activate the conda environment specified in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate lb2dgeom
```

### Editable install

If you prefer a manual install, run:

```bash
pip install -e .
```

## Usage

```python
from lb2dgeom.grids import Grid
from lb2dgeom.shapes.circle import Circle
from lb2dgeom.raster import rasterize

grid = Grid(nx=100, ny=100, dx=1.0, origin=(-50.0, -50.0))
shape = Circle(0.0, 0.0, 20.0)
phi, solid = rasterize(grid, shape)
```

See the `examples/` directory for complete scripts including Bouzidi
coefficients, file I/O, and plotting utilities.

## Running tests

After installing the package, execute:

```bash
pytest -q
```

## License

This project is released under the MIT License.

