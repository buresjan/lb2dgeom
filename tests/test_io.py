import numpy as np

from lb2dgeom.io import load_npz, save_npz


def test_save_load_npz(tmp_path):
    solid = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    phi = np.array([[1.0, -1.0], [-0.5, 0.5]], dtype=np.float32)
    bouzidi = np.zeros((2, 2, 9), dtype=np.float32)
    extra = np.array([1, 2, 3], dtype=np.float32)

    path = tmp_path / "geom.npz"
    save_npz(path, solid, phi, bouzidi, extra=extra)

    data = load_npz(path)
    assert np.array_equal(data["solid"], solid)
    assert np.array_equal(data["phi"], phi)
    assert np.array_equal(data["bouzidi"], bouzidi)
    assert np.array_equal(data["extra"], extra)
