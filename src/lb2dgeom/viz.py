import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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

def plot_phi(phi: np.ndarray, fname: str, levels: Optional[int] = 20, show: bool = False) -> None:
    """Plot signed distance field with contours."""
    out_dir = _ensure_output_dir()
    plt.figure()
    plt.imshow(phi, origin="lower", cmap="coolwarm")
    if levels:
        plt.contour(phi, levels=levels, colors="k", linewidths=0.5, origin="lower")
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
    plt.hist(vals, bins=50, range=(0,1))
    plt.title("Bouzidi q_i histogram")
    plt.xlabel("q_i")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    if show:
        plt.show()
    plt.close()
