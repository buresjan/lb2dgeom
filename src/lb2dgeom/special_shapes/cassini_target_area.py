"""Special-case Cassini geometry with target-area parameterisation."""

import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from lb2dgeom.bouzidi import compute_bouzidi
from lb2dgeom.grids import Grid
from lb2dgeom.io import save_txt as export_txt
from lb2dgeom.raster import classify_cells, rasterize
from lb2dgeom.shapes.cassini_oval import (
    CassiniOval,
    cassini_b_from_area,
    cassini_oval_area,
)

PathLike = Union[str, os.PathLike]


@dataclass
class TargetAreaCassiniGeometry:
    """Generate the fixed-domain target-area Cassini special case.

    Parameters
    ----------
    a : float
        Standard Cassini ``a`` parameter, i.e. the half-distance between the
        two foci.
    area : float
        Target enclosed area of the Cassini oval.
    theta : float, optional
        Rotation angle in radians. Defaults to ``0.0``.
    nx : int, optional
        Number of grid cells in the ``x`` direction. Defaults to ``2048``.
    ny : int, optional
        Number of grid cells in the ``y`` direction. Defaults to ``256``.
    txt_path : path-like or None, optional
        Path where the TXT geometry export is written during initialisation.
        If ``None``, the automatic TXT export is skipped. Defaults to
        ``"cassini_target_area_geom.txt"``.
    include_header : bool, optional
        Whether to include the TXT header row. Defaults to ``True``.
    selection : {"all", "fluid", "near_wall", "wall"}, optional
        Cell selection passed to :func:`lb2dgeom.io.save_txt`. Defaults to
        ``"all"``.
    domain_x : float, optional
        Physical domain size in ``x``. Defaults to ``4.0``.
    domain_y : float, optional
        Physical domain size in ``y``. Defaults to ``0.5``.
    x0 : float, optional
        Cassini-oval centre ``x`` coordinate. Defaults to ``0.75``.
    y0 : float, optional
        Cassini-oval centre ``y`` coordinate. Defaults to ``0.25``.
    num_theta : int, optional
        Angular quadrature intervals used to solve the standard Cassini ``b``
        parameter from the target area. Defaults to ``8192``.

    Attributes
    ----------
    b : float
        Solved standard Cassini ``b`` parameter.
    grid : Grid
        Uniform Cartesian grid for the special-case domain.
    shape : CassiniOval
        Analytic Cassini-oval shape instance.
    phi : np.ndarray
        Signed distance field on the grid.
    solid : np.ndarray
        Solid-mask array where ``1`` denotes solid cells.
    bouzidi : np.ndarray
        Bouzidi coefficients on the grid.
    cell_types : np.ndarray
        Cell-type classification with the default ``0/1/2`` codes.
    analytic_area : float
        Area computed from the analytic Cassini parameterisation.
    raster_area : float
        Area implied by the rasterised solid mask.

    Notes
    -----
    This special case keeps the physical domain and centre used by the
    corresponding example script. The grid must remain isotropic because
    :class:`lb2dgeom.grids.Grid` stores a single spacing value.
    """

    a: float
    area: float
    theta: float = 0.0
    nx: int = 2048
    ny: int = 256
    txt_path: Optional[PathLike] = "cassini_target_area_geom.txt"
    include_header: bool = True
    selection: str = "all"
    domain_x: float = 4.0
    domain_y: float = 0.5
    x0: float = 0.75
    y0: float = 0.25
    num_theta: int = 8192

    b: float = field(init=False)
    grid: Grid = field(init=False)
    shape: CassiniOval = field(init=False)
    phi: np.ndarray = field(init=False)
    solid: np.ndarray = field(init=False)
    bouzidi: np.ndarray = field(init=False)
    cell_types: np.ndarray = field(init=False)
    analytic_area: float = field(init=False)
    raster_area: float = field(init=False)

    def __post_init__(self) -> None:
        """Build the geometry, export TXT if requested, and print a summary."""
        self._validate_inputs()

        dx = self.domain_x / self.nx
        dy = self.domain_y / self.ny
        if not np.isclose(dx, dy):
            raise ValueError(
                "nx and ny must preserve isotropic spacing for the fixed domain; "
                "choose values such that domain_x / nx == domain_y / ny"
            )

        self.b = cassini_b_from_area(self.a, self.area, num_theta=self.num_theta)
        self.analytic_area = cassini_oval_area(self.a, self.b, num_theta=self.num_theta)
        self.shape = CassiniOval.from_standard_parameters(
            x0=self.x0,
            y0=self.y0,
            a=self.a,
            b=self.b,
            theta=self.theta,
        )
        self.grid = Grid(nx=self.nx, ny=self.ny, dx=dx, origin=(0.0, 0.0))
        self.phi, self.solid = rasterize(self.grid, self.shape)
        self.bouzidi = compute_bouzidi(self.grid, self.phi, self.solid)
        self.cell_types = classify_cells(self.solid)
        self.raster_area = float(np.count_nonzero(self.solid)) * self.grid.dx**2

        if self.txt_path is not None:
            self.txt_path = self.save_txt(
                self.txt_path,
                selection=self.selection,
                include_header=self.include_header,
            )

        print(self.summary())

    @property
    def topology(self) -> str:
        """Return ``single-loop`` or ``two-loop`` for the solved geometry."""
        return "single-loop" if self.b >= self.a else "two-loop"

    def summary(self) -> str:
        """Return the terminal summary used when the object is created."""
        txt_info = (
            f"TXT geometry: {self.txt_path}"
            if self.txt_path is not None
            else "TXT geometry: not written"
        )
        return (
            "Cassini target-area geometry created.\n"
            f"Domain: {self.domain_x} x {self.domain_y}\n"
            f"Grid: {self.nx} x {self.ny} (dx = {self.grid.dx:.8f})\n"
            f"Center: ({self.x0:.8f}, {self.y0:.8f})\n"
            f"Standard parameters: a = {self.a:.8f}, b = {self.b:.8f}\n"
            f"Rotation theta: {self.theta:.8f} rad\n"
            f"Topology: {self.topology}\n"
            f"Target area: {self.area:.8f}\n"
            f"Analytic area: {self.analytic_area:.8f}\n"
            f"Rasterized area: {self.raster_area:.8f}\n"
            f"{txt_info}"
        )

    def save_txt(
        self,
        path: PathLike,
        *,
        selection: str = "all",
        include_header: bool = True,
    ) -> str:
        """Write the TXT geometry export and return the resolved path.

        Parameters
        ----------
        path : path-like
            Output TXT path.
        selection : {"all", "fluid", "near_wall", "wall"}, optional
            Cell selection written to the file. Defaults to ``"all"``.
        include_header : bool, optional
            Whether to include the header row. Defaults to ``True``.

        Returns
        -------
        str
            The filesystem path used for the TXT export.
        """
        resolved = os.fspath(path)
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        export_txt(
            resolved,
            self.grid,
            self.cell_types,
            self.bouzidi,
            selection=selection,
            include_header=include_header,
        )
        return resolved

    def _validate_inputs(self) -> None:
        """Validate user-facing constructor inputs."""
        if self.a < 0.0:
            raise ValueError("a must be non-negative")
        if self.area <= 0.0:
            raise ValueError("area must be positive")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.domain_x <= 0.0 or self.domain_y <= 0.0:
            raise ValueError("domain_x and domain_y must be positive")
        if self.num_theta < 32:
            raise ValueError("num_theta must be at least 32")
