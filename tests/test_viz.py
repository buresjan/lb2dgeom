import os
import numpy as np

from lb2dgeom import viz


def test_viz_outputs():
    phi = np.array([[1.0, -1.0], [-0.5, 0.5]], dtype=np.float32)
    bouzidi = np.zeros((2, 2, 9), dtype=np.float32)

    viz.plot_phi(phi, "phi_test.png", levels=5, show=False)
    viz.plot_bouzidi_hist(bouzidi, "hist_test.png", show=False)
    viz.plot_bouzidi_dirs(bouzidi, "dir_test", show=False)

    out_dir = os.path.join("examples", "output")
    assert os.path.exists(os.path.join(out_dir, "phi_test.png"))
    assert os.path.exists(os.path.join(out_dir, "hist_test.png"))
    assert os.path.exists(os.path.join(out_dir, "dir_test_0.png"))
