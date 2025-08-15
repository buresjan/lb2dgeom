"""Visualization utilities for geometry and Bouzidi diagnostics."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def _ensure_output_dir():
    out_dir = os.path.join("examples", "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_solid(solid: np.ndarray, fname: str, show: bool = False) -> None:
    """Plot solid mask."""
    out_dir = _ensure_output_dir()
    plt.figure()
    plt.imshow(solid, origin="lower", cmap="gray_r")
    plt.title("Solid mask")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_phi(
    phi: np.ndarray, fname: str, levels: Optional[int] = 20, show: bool = False
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
    """
    out_dir = _ensure_output_dir()
    plt.figure()
    max_abs = float(np.nanmax(np.abs(phi)))
    norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    plt.imshow(phi, origin="lower", cmap="coolwarm", norm=norm)
    if levels:
        plt.contour(phi, levels=levels, colors="k", linewidths=0.5)
    plt.title("Signed distance field φ")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_bouzidi_hist(bouzidi: np.ndarray, fname: str, show: bool = False) -> None:
    """Histogram of Bouzidi q_i values (ignoring NaNs)."""
    out_dir = _ensure_output_dir()
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
    bouzidi: np.ndarray, fname_prefix: str, show: bool = False
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
    """
    out_dir = _ensure_output_dir()
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
