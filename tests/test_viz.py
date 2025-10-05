import numpy as np

from lb2dgeom import viz


def test_viz_outputs(tmp_path):
    phi = np.array([[1.0, -1.0], [-0.5, 0.5]], dtype=np.float32)
    bouzidi = np.zeros((2, 2, 9), dtype=np.float32)
    cell_types = np.array([[0, 1], [2, 0]], dtype=np.int8)

    viz.plot_phi(phi, "phi_test.png", levels=5, show=False, out_dir=tmp_path)
    viz.plot_bouzidi_hist(bouzidi, "hist_test.png", show=False, out_dir=tmp_path)
    viz.plot_bouzidi_dirs(bouzidi, "dir_test", show=False, out_dir=tmp_path)
    viz.plot_cell_types(cell_types, "cell_types_test.png", show=False, out_dir=tmp_path)

    assert (tmp_path / "phi_test.png").exists()
    assert (tmp_path / "hist_test.png").exists()
    assert (tmp_path / "dir_test_0.png").exists()
    assert (tmp_path / "cell_types_test.png").exists()


def test_plot_phi_constant_field(tmp_path):
    phi = np.zeros((4, 4), dtype=np.float32)
    viz.plot_phi(phi, "phi_constant.png", show=False, out_dir=tmp_path)
    assert (tmp_path / "phi_constant.png").exists()


def test_plot_phi_all_nan(tmp_path):
    phi = np.full((3, 3), np.nan, dtype=np.float32)
    viz.plot_phi(phi, "phi_all_nan.png", show=False, out_dir=tmp_path)
    assert (tmp_path / "phi_all_nan.png").exists()
