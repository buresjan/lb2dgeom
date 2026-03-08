import numpy as np
import pytest

from lb2dgeom.shapes.cassini_oval import cassini_oval_area
from lb2dgeom.special_shapes import TargetAreaCassiniGeometry


def test_target_area_cassini_geometry_creates_txt_and_prints_summary(tmp_path, capsys):
    txt_path = tmp_path / "cassini_target_area_geom.txt"

    geometry = TargetAreaCassiniGeometry(
        a=0.1,
        area=0.04,
        theta=np.pi / 6,
        nx=80,
        ny=10,
        txt_path=txt_path,
        num_theta=4096,
    )

    captured = capsys.readouterr().out

    assert "Cassini target-area geometry created." in captured
    assert f"TXT geometry: {txt_path}" in captured
    assert geometry.cell_types.shape == (10, 80)
    assert np.isclose(geometry.grid.dx, 0.05)
    assert np.isclose(
        cassini_oval_area(geometry.a, geometry.b, num_theta=4096),
        0.04,
        atol=1e-8,
    )
    assert txt_path.exists()

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("x y type q_east")
    assert len(lines) > 1


def test_target_area_cassini_geometry_requires_isotropic_grid():
    with pytest.raises(ValueError, match="isotropic spacing"):
        TargetAreaCassiniGeometry(a=0.1, area=0.04, nx=81, ny=10, txt_path=None)
