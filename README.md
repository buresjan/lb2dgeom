# lbmgeom

**Analytic geometry generator for Lattice Boltzmann Method (LBM) simulations**  
Generates parametric 2D shapes on a uniform Cartesian grid, rasterizes them into a signed distance field (φ),  
and computes Bouzidi interpolated boundary condition parameters for the D2Q9 model.

## Features

- **Analytic shapes**: Circle, Ellipse, Rectangle (axis-aligned / rotated), Rounded Rectangle, Cassini Oval
- **Boolean composition**: Union, Intersection, Difference
- **Accurate signed distance functions** for voxelization and boundary location
- **Uniform grid** with customizable spacing, extent, and origin
- **Bouzidi interpolated boundary** link-fraction calculation for D2Q9
- **I/O helpers** for saving/loading `.npz` geometry datasets
- **Visualization utilities** for solid mask, φ field, and Bouzidi diagnostics
- Designed for integration with external LBM solvers (e.g., OptiLB) via direct array outputs

## Installation (Local)

```bash
git clone <this-repo-url>
cd lbmgeom
pip install -e .
