import numpy as np

from lb2dgeom.raster import classify_cells


def test_classify_cells_8_connected_ring():
    # 3x3 grid with center solid cell
    solid = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    ct = classify_cells(solid, fluid_code=0, near_wall_code=1, wall_code=2)

    # Expected: center=2 (wall), all others=1 (near-wall), none=0 (fluid)
    expected = np.array(
        [
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1],
        ],
        dtype=np.result_type(0, 1, 2),
    )
    assert ct.dtype == np.result_type(0, 1, 2)
    assert np.array_equal(ct, expected)


def test_classify_cells_edges_diagonals_included():
    # 4x4 grid with top-left corner solid block (2x2)
    solid = np.zeros((4, 4), dtype=np.uint8)
    solid[:2, :2] = 1
    ct = classify_cells(solid)

    # Near-wall should include diagonals like (2,2)
    assert ct[2, 2] == 1  # diagonal neighbor to the block
    # Far fluid should remain 0
    assert ct[3, 3] == 0
    # Walls coded as 2
    assert np.all(ct[:2, :2] == 2)


def test_classify_cells_preserves_custom_codes():
    solid = np.zeros((3, 3), dtype=np.uint8)
    solid[0, 0] = 1
    ct = classify_cells(
        solid,
        fluid_code=10,
        near_wall_code=20,
        wall_code=255,
    )
    assert ct.dtype == np.result_type(10, 20, 255)
    assert ct[0, 0] == 255
    assert ct[0, 1] == 20
    assert ct[1, 1] == 20
    assert ct[2, 2] == 10
