import numpy as np
from typing import Any, Dict

def save_npz(path: str, solid: np.ndarray, phi: np.ndarray, bouzidi: np.ndarray, **extras: Any) -> None:
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


def load_npz(path: str) -> Dict[str, Any]:
    """
    Load geometry arrays from a compressed .npz file.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    data : dict
        Dictionary with 'solid', 'phi', 'bouzidi', and any extra keys.
    """
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}
