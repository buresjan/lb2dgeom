import numpy as np

from lb2dgeom.grids import Grid
from lb2dgeom.io import save_txt


def test_save_txt_all_and_filtered(tmp_path):
    g = Grid(nx=2, ny=2, dx=1.0, origin=(0.0, 0.0))
    # cell_types: 0=fluid,1=near-wall,2=wall
    cell_types = np.array([[0, 1], [2, 0]], dtype=np.int8)
    bouzidi = np.zeros((2, 2, 9), dtype=np.float32)
    # Fill q1..q8 for determinism
    for i in range(1, 9):
        bouzidi[:, :, i] = i
    # Introduce a NaN for (y=0,x=0) at q1, which should be written as -1
    bouzidi[0, 0, 1] = np.nan

    # All cells
    f_all = tmp_path / "all.txt"
    save_txt(f_all, g, cell_types, bouzidi, selection="all", include_header=True)
    lines = f_all.read_text().strip().splitlines()
    # 1 header + 4 data lines
    assert len(lines) == 5
    # Data lines should have 11 whitespace-separated fields
    for ln in lines[1:]:
        parts = ln.split()
        assert len(parts) == 11
    # The first data line corresponds to x=0,y=0 and should have q1 == -1
    first_vals = lines[1].split()
    assert float(first_vals[3]) == -1.0

    # Filtered near-wall only
    f_near = tmp_path / "near.txt"
    save_txt(f_near, g, cell_types, bouzidi, selection="near_wall", include_header=False)
    lines = f_near.read_text().strip().splitlines()
    # Only one near-wall cell in our map
    assert len(lines) == 1
    assert len(lines[0].split()) == 11
