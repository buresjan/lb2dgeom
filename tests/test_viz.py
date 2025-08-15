import numpy as np

from lb2dgeom import viz


def test_viz_outputs(tmp_path):
    phi = np.array([[1.0, -1.0], [-0.5, 0.5]], dtype=np.float32)
    bouzidi = np.zeros((2, 2, 9), dtype=np.float32)

    viz.plot_phi(phi, "phi_test.png", levels=5, out_dir=tmp_path, show=False)
    viz.plot_bouzidi_hist(bouzidi, "hist_test.png", out_dir=tmp_path, show=False)
    viz.plot_bouzidi_dirs(bouzidi, "dir_test", out_dir=tmp_path, show=False)

    assert (tmp_path / "phi_test.png").exists()
    assert (tmp_path / "hist_test.png").exists()
    assert (tmp_path / "dir_test_0.png").exists()
