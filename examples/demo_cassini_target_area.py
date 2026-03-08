"""Cassini-oval example with fixed domain, resolution, and target area."""

import argparse
import os

import numpy as np

from lb2dgeom import viz
from lb2dgeom.io import save_npz
from lb2dgeom.special_shapes import TargetAreaCassiniGeometry

DOMAIN_X = 4.0
DOMAIN_Y = 0.5
NX = 2048
NY = 256
CENTER_X = 0.75
CENTER_Y = 0.25
TARGET_AREA = 0.04
DEFAULT_STANDARD_A = 0.1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Cassini oval on a 4.0 x 0.5 domain with a 2048 x 256 grid. "
            "The standard Cassini parameter a is user-specified and b is solved "
            "to match the fixed target area of 0.04."
        )
    )
    parser.add_argument(
        "--a",
        type=float,
        default=DEFAULT_STANDARD_A,
        help="Standard Cassini a parameter: half-distance between the two foci.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.0,
        help="Rotation angle in radians. Defaults to 0.0.",
    )
    parser.add_argument(
        "--num-theta",
        type=int,
        default=8192,
        help="Angular quadrature intervals used when solving for b.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("examples", "output"),
        help="Directory for exported arrays, text files, and PNG diagnostics.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    dx = DOMAIN_X / NX
    dy = DOMAIN_Y / NY
    if not np.isclose(dx, dy):
        raise ValueError("Requested resolution must produce square cells")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    geometry = TargetAreaCassiniGeometry(
        a=args.a,
        area=TARGET_AREA,
        theta=args.theta,
        nx=NX,
        ny=NY,
        x0=CENTER_X,
        y0=CENTER_Y,
        domain_x=DOMAIN_X,
        domain_y=DOMAIN_Y,
        num_theta=args.num_theta,
        txt_path=os.path.join(out_dir, "cassini_target_area_geom.txt"),
    )

    save_npz(
        os.path.join(out_dir, "cassini_target_area_geom.npz"),
        geometry.solid,
        geometry.phi,
        geometry.bouzidi,
        cell_types=geometry.cell_types,
        standard_a=np.array(geometry.a),
        standard_b=np.array(geometry.b),
        target_area=np.array(TARGET_AREA),
        analytic_area=np.array(geometry.analytic_area),
        raster_area=np.array(geometry.raster_area),
        theta=np.array(geometry.theta),
    )

    viz.plot_solid(
        geometry.solid,
        "cassini_target_area_solid.png",
        show=False,
        out_dir=out_dir,
    )
    viz.plot_phi(
        geometry.phi,
        "cassini_target_area_phi.png",
        levels=30,
        show=False,
        out_dir=out_dir,
    )
    viz.plot_bouzidi_hist(
        geometry.bouzidi,
        "cassini_target_area_bouzidi_hist.png",
        show=False,
        out_dir=out_dir,
    )
    viz.plot_bouzidi_dirs(
        geometry.bouzidi,
        "cassini_target_area_bouzidi_dir",
        show=False,
        out_dir=out_dir,
    )
    viz.plot_cell_types(
        geometry.cell_types,
        "cassini_target_area_cell_types.png",
        show=False,
        out_dir=out_dir,
    )

    print(f"PNG diagnostics and NPZ saved in {out_dir}/")


if __name__ == "__main__":
    main()
