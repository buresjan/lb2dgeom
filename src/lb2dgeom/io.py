import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

from .grids import Grid


def save_npz(
    path: str,
    solid: np.ndarray,
    phi: np.ndarray,
    bouzidi: np.ndarray,
    **extras: Any,
) -> None:
    """
    Save geometry arrays to a compressed .npz file.

    Parameters
    ----------
    path : str
        Output file path.
    solid : ndarray
        Solid mask array (uint8 or bool).
    phi : ndarray
        Signed distance field (float32).
    bouzidi : ndarray
        Bouzidi q_i array (float32).
    extras : dict
        Additional arrays or metadata to save.
    """
    np.savez_compressed(path, solid=solid, phi=phi, bouzidi=bouzidi, **extras)


def load_npz(path: str, allow_pickle: bool = False) -> Dict[str, Any]:
    """Load geometry arrays from a compressed ``.npz`` file.

    Parameters
    ----------
    path : str
        File path.
    allow_pickle : bool, optional
        Whether to allow loading pickled object arrays. Defaults to ``False``.

    Returns
    -------
    data : dict
        Dictionary with ``'solid'``, ``'phi'``, ``'bouzidi'``, and any extra keys.
    """
    with np.load(path, allow_pickle=allow_pickle) as data:
        return {key: data[key] for key in data.files}


def save_txt(
    path: Union[str, os.PathLike],
    grid: Grid,
    cell_types: np.ndarray,
    bouzidi: np.ndarray,
    *,
    selection: Optional[str] = "all",
    codes: Tuple[int, int, int] = (0, 1, 2),
    include_header: bool = False,
    float_fmt: str = "%.8g",
) -> None:
    """Export per-cell data to a whitespace-delimited text file.

    Each row contains 11 numbers: ``x y type q1 q2 q3 q4 q5 q6 q7 q8``.
    Coordinates are physical cell-center coordinates from the provided ``grid``.
    Bouzidi coefficients are taken from directions 1..8 (moving directions),
    in the standard D2Q9 ordering. Non-existent coefficients (NaN) are written
    as ``-1`` for easier downstream parsing.

    Parameters
    ----------
    path : str or os.PathLike
        Output file path.
    grid : Grid
        Grid providing cell-center coordinates (physical units).
    cell_types : np.ndarray
        Integer array of shape ``(ny, nx)`` encoding cell categories,
        typically produced by ``lb2dgeom.raster.classify_cells``.
    bouzidi : np.ndarray
        Array of shape ``(ny, nx, 9)`` with Bouzidi coefficients; only
        directions 1..8 are written. Values may be ``NaN`` when no boundary
        link exists.
    selection : {"all", "fluid", "near_wall", "wall"}, optional
        Subset of cells to export. Defaults to ``"all"``.
    codes : tuple of int, optional
        Mapping of codes in ``cell_types`` as ``(fluid, near_wall, wall)``.
        Defaults to ``(0, 1, 2)``.
    include_header : bool, optional
        If ``True``, write a single header line with column names.
        Defaults to ``False`` (numeric-only file).
    float_fmt : str, optional
        Format string for floating-point values. Defaults to ``"%.8g"``.

    Returns
    -------
    None
    """
    if cell_types.ndim != 2:
        raise ValueError("cell_types must be 2D (ny, nx)")
    if bouzidi.ndim != 3 or bouzidi.shape[:2] != cell_types.shape or bouzidi.shape[2] != 9:
        raise ValueError("bouzidi must have shape (ny, nx, 9) matching cell_types")

    ny, nx = cell_types.shape
    X, Y = grid.coords()

    # Flatten columns in row-major order
    x_col = X.ravel()
    y_col = Y.ravel()
    t_col = cell_types.astype(np.int64).ravel()
    q_cols = [bouzidi[:, :, i].ravel() for i in range(1, 9)]
    # Replace NaNs with -1 to indicate non-existent boundary links
    q_cols = [np.where(np.isnan(q), -1.0, q) for q in q_cols]

    # Apply selection mask
    sel = (selection or "all").lower()
    fluid_code, near_code, wall_code = codes
    if sel == "all":
        mask = np.ones_like(t_col, dtype=bool)
    elif sel == "fluid":
        mask = t_col == fluid_code
    elif sel == "near_wall":
        mask = t_col == near_code
    elif sel == "wall":
        mask = t_col == wall_code
    else:
        raise ValueError(
            "selection must be one of {'all','fluid','near_wall','wall'} or None"
        )

    # Stack the selected rows: shape (N, 11)
    cols: Iterable[np.ndarray] = [x_col[mask], y_col[mask], t_col[mask]] + [q[mask] for q in q_cols]
    data = np.column_stack(tuple(cols))

    # Mixed formatting: x, y (float), type (int), q1..q8 (float)
    fmts = [float_fmt, float_fmt, "%d"] + [float_fmt] * 8
    header = "x y type q1 q2 q3 q4 q5 q6 q7 q8" if include_header else ""

    # Use numpy.savetxt for efficiency and consistency
    np.savetxt(path, data, fmt=fmts, header=header, comments="" if include_header else "")
