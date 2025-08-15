# lb2dgeom – Agents & Contributor Guide

**lb2dgeom** is a local-install Python package for generating analytic 2D geometries, rasterizing them to uniform grids for Lattice Boltzmann Method (LBM) simulations, and computing Bouzidi interpolated boundary-condition parameters for the D2Q9 model.
This document describes the project structure, coding conventions, and the specific “agent” role of each module so contributors (human or automated) can maintain a consistent architecture.

---

## Project Overview

**Purpose:**
The library’s goal is to provide high-accuracy signed distance field (SDF) generation and Bouzidi link-fraction computation for a variety of analytic shapes, with outputs directly consumable by existing LBM solvers (e.g., OptiLB). It supports boolean shape composition, robust rasterization, and clear I/O and plotting utilities for validation.

**Key Features:**

* Parametric analytic shapes: Circle, Ellipse, Rectangle (axis-aligned/rotated), Rounded Rectangle, Cassini Oval.
* Boolean operations: Union, Intersection, Difference.
* Signed distance field (SDF) framework.
* Bouzidi qᵢ calculation for D2Q9 with physical-length normalization.
* Local-install setup for development and integration testing.
* Visualization and diagnostic plots (saved as PNGs by default).

---

## Repository Structure

```
.
├── AGENTS.md
├── README.md
├── setup.cfg
├── pyproject.toml
├── examples
│   ├── demo_cassini.py
│   └── demo_ellipse.py
├── src
│   └── lb2dgeom
│       ├── bouzidi.py
│       ├── d2q9.py
│       ├── grids.py
│       ├── __init__.py
│       ├── io.py
│       ├── raster.py
│       ├── shapes
│       │   ├── base.py
│       │   ├── cassini_oval.py
│       │   ├── circle.py
│       │   ├── ellipse.py
│       │   ├── __init__.py
│       │   ├── ops.py
│       │   ├── rectangle.py
│       │   └── rounded_rect.py
│       └── viz.py
└── tests
    ├── test_raster_bouzidi.py
    └── test_shapes.py
```

---

## Module Responsibilities

* **`grids.py`** – Handles creation and management of a uniform Cartesian grid, including cell-center coordinates, grid spacing, and origin.
* **`shapes/base.py`** – Defines the base `Shape` interface, including the `.sdf(x, y)` method for signed distance evaluation and support for rotation.
* **`shapes/<primitive>.py`** – Contains implementations of signed distance functions for specific parametric shapes (circle, ellipse, rectangle, rounded rectangle, Cassini oval).
* **`shapes/ops.py`** – Implements boolean shape composition (union, intersection, difference) using standard SDF min/max rules.
* **`raster.py`** – Rasterizes a shape’s SDF onto the grid, producing the signed distance field (`phi`) and the binary solid mask.
* **`d2q9.py`** – Defines the D2Q9 discrete velocity set and associated constants for LBM simulations.
* **`bouzidi.py`** – Computes Bouzidi link fractions (`q_i`) for fluid cells adjacent to solid boundaries, using the signed distance field.
* **`io.py`** – Provides helper functions to save and load geometry datasets in `.npz` format.
* **`viz.py`** – Provides visualization tools to plot the solid mask, φ contours, and Bouzidi histograms, saving PNGs by default.
* **`examples/`** – Contains demonstration scripts showing end-to-end usage of the package; all examples save outputs for documentation.
* **`tests/`** – Contains unit tests covering shape SDFs, rasterization, and Bouzidi computations.

---

## Code Style and Conventions

* **PEP 8:** Follow Python naming and formatting guidelines.
* **Black:** Use `black` for autoformatting (`pip install black`).
* **Numpydoc:** All public functions/classes must have Numpydoc-style docstrings.
* **Type Hints:** Include type hints for all parameters and returns.
* **Imports:** Absolute imports preferred (`from lb2dgeom...`).
* **Numerics:**

  * Always use double precision in internal calculations unless memory is critical, then cast to float32 for storage.
  * Handle diagonals in D2Q9 using actual Euclidean length.

---

## Pull Request Guidelines

* Keep PRs focused and atomic.
* Include or update tests for any new/changed functionality.
* Ensure all tests pass (`pytest -q`).
* Ensure visual outputs remain correct—run example scripts and confirm saved PNGs look as expected.
* Document new features in README.md and update AGENTS.md if responsibilities change.

---

## Development Environment Setup

1. Python 3.8+ (preferably 3.10+)
2. Clone the repo:

   ```bash
   git clone <repo-url>
   cd lb2dgeom
   ```
3. Install in editable mode:

   ```bash
   pip install -e .
   ```
4. Install dev tools:

   ```bash
   pip install black mypy pytest matplotlib numpy
   ```
5. Format before committing:

   ```bash
   black .
   ```
6. Run tests:

   ```bash
   pytest
   ```

---

## Agent Interaction Notes

* **Grid agent** must provide coordinates in *physical* units for SDF evaluation.
* **Shape agents** must treat negative φ as inside solid.
* **Bouzidi agent** assumes φ is continuous; SDF discontinuities will break root-finding.
* **Viz agent** writes plots to a specified `out_dir` (defaulting to the current
  working directory) and shows interactively only in `__main__`.
* **I/O agent** should always save `solid`, `phi`, `bouzidi` arrays with matching dimensions.

---

Following this **Agents & Contributor Guide** keeps `lb2dgeom` maintainable, reproducible, and integration-ready for larger CFD studies.
