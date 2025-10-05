"""Visualization utilities for geometry and Bouzidi diagnostics."""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def _ensure_output_dir(out_dir: Optional[str] = None) -> str:
    """Return a directory path for plot outputs, creating it if needed.

    Parameters
    ----------
    out_dir : str, optional
        Target directory for output images. If ``None``, the current working
        directory is used.

    Returns
    -------
    str
        Absolute path to the output directory.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_solid(
    solid: np.ndarray,
    fname: str,
    show: bool = False,
    out_dir: Optional[str] = None,
) -> None:
    """Plot a binary solid mask.

    Parameters
    ----------
    solid : np.ndarray
        Binary array marking solid cells.
    fname : str
        Output PNG filename.
    show : bool, optional
        If ``True``, display the figure interactively.
    out_dir : str, optional
        Directory to save the output image. Defaults to the current working
        directory.

    Returns
    -------
    None
    """
    out_dir = _ensure_output_dir(out_dir)
    plt.figure()
    plt.imshow(solid, origin="lower", cmap="gray_r")
    plt.title("Solid mask")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_cell_types(
    cell_types: np.ndarray,
    fname: str,
    codes: Tuple[int, int, int] = (0, 1, 2),
    show: bool = False,
    out_dir: Optional[str] = None,
) -> None:
    """Plot categorical cell types: fluid, near-wall, and wall.

    Parameters
    ----------
    cell_types : np.ndarray
        Integer array encoding categories. By default, ``0`` = fluid,
        ``1`` = near-wall (8-neighbour adjacency), ``2`` = wall.
    fname : str
        Output PNG filename.
    codes : tuple of int, optional
        A triplet ``(fluid_code, near_wall_code, wall_code)`` to match the
        values in ``cell_types``. Defaults to ``(0, 1, 2)``.
    show : bool, optional
        If ``True``, display the figure interactively.
    out_dir : str, optional
        Directory to save the output image. Defaults to the current working
        directory.

    Notes
    -----
    Colors are chosen for clarity on a light background: fluid (light gray),
    near-wall (orange), wall (black). A colorbar with labels is included.
    """
    out_dir = _ensure_output_dir(out_dir)

    fluid_code, near_code, wall_code = codes
    # Map arbitrary codes to 0,1,2 for stable coloring
    mapped = np.full_like(cell_types, fill_value=-1, dtype=np.int8)
    mapped[cell_types == fluid_code] = 0
    mapped[cell_types == near_code] = 1
    mapped[cell_types == wall_code] = 2

    cmap = colors.ListedColormap(["#d9d9d9", "#ff7f0e", "#000000"])  # fluid, near, wall

    plt.figure()
    im = plt.imshow(mapped, origin="lower", cmap=cmap, interpolation="nearest")
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["fluid", "near-wall", "wall"])  # type: ignore[attr-defined]
    plt.title("Cell types: fluid / near-wall / wall")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_phi(
    phi: np.ndarray,
    fname: str,
    levels: Optional[int] = 20,
    show: bool = False,
    out_dir: Optional[str] = None,
) -> None:
    """
    Plot signed distance field with a diverging colormap centered at zero.

    Parameters
    ----------
    phi : np.ndarray
        Signed distance values at cell centres.
    fname : str
        Output PNG filename.
    levels : int, optional
        Number of contour levels to overlay. Set to ``None`` to skip contours.
    show : bool, optional
        If ``True``, display the figure interactively.
    out_dir : str, optional
        Directory to save the output image. Defaults to the current working
        directory.
    """
    out_dir = _ensure_output_dir(out_dir)
    # Use explicit fig/ax and bind colorbar to the imshow mappable.
    # Otherwise, adding a monochrome contour afterwards makes plt.colorbar()
    # attach to the contour set, yielding a blank colorbar.
    fig, ax = plt.subplots()
    abs_phi = np.abs(phi)
    if np.isnan(abs_phi).all():
        max_abs = 0.0
    else:
        with np.errstate(all="ignore"):
            try:
                max_abs = float(np.nanmax(abs_phi))
            except ValueError:
                max_abs = 0.0
    if not np.isfinite(max_abs):
        max_abs = 0.0

    norm = None
    if max_abs > 0.0 and np.isfinite(max_abs):
        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    im = ax.imshow(phi, origin="lower", cmap="coolwarm", norm=norm)
    if levels and max_abs > 0.0:
        ax.contour(phi, levels=levels, colors="k", linewidths=0.5)
    ax.set_title("Signed distance field φ")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_bouzidi_hist(
    bouzidi: np.ndarray,
    fname: str,
    show: bool = False,
    out_dir: Optional[str] = None,
) -> None:
    """Plot a histogram of Bouzidi ``q_i`` values.

    Parameters
    ----------
    bouzidi : np.ndarray
        Bouzidi coefficients; NaNs are ignored.
    fname : str
        Output PNG filename.
    show : bool, optional
        If ``True``, display the figure interactively.
    out_dir : str, optional
        Directory to save the output image. Defaults to the current working
        directory.

    Returns
    -------
    None
    """
    out_dir = _ensure_output_dir(out_dir)
    plt.figure()
    vals = bouzidi[~np.isnan(bouzidi)]
    plt.hist(vals, bins=50, range=(0, 1))
    plt.title("Bouzidi q_i histogram")
    plt.xlabel("q_i")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_bouzidi_dirs(
    bouzidi: np.ndarray,
    fname_prefix: str,
    show: bool = False,
    out_dir: Optional[str] = None,
) -> None:
    """Plot Bouzidi ``q_i`` fields for each direction separately.

    Parameters
    ----------
    bouzidi : np.ndarray
        Array of shape ``(ny, nx, 9)`` containing Bouzidi coefficients.
    fname_prefix : str
        Prefix for output PNG files. Images are saved as ``<prefix>_i.png``.
    show : bool, optional
        If ``True``, display each figure interactively.
    out_dir : str, optional
        Directory to save the output images. Defaults to the current working
        directory.
    """
    out_dir = _ensure_output_dir(out_dir)
    for i in range(9):
        plt.figure()
        plt.imshow(bouzidi[:, :, i], origin="lower", cmap="viridis")
        plt.title(f"Bouzidi q_{i}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname_prefix}_{i}.png"), dpi=150)
        if show:
            plt.show()
        plt.close()
