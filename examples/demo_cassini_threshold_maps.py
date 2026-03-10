"""Generate Cassini voxel-change threshold maps in relative percent."""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from lb2dgeom.shapes.cassini_oval import cassini_b_from_area

DOMAIN_X = 4.0
DOMAIN_Y = 0.5
CENTER_X = 0.75
CENTER_Y = 0.25
TARGET_AREA = 0.035
A_MIN = 0.08
A_MAX = 0.13
THETA_MIN = np.pi / 4.0
THETA_MAX = np.pi / 2.0
RESOLUTIONS: Tuple[Tuple[int, int], ...] = ((4096, 512), (2048, 256))


@dataclass(frozen=True)
class ParameterTable:
    """Interpolants for the target-area Cassini parameterisation."""

    a_values: np.ndarray
    b_values: np.ndarray
    b_derivative: np.ndarray

    def b(self, a: float) -> float:
        """Return the target-area Cassini ``b(a)`` parameter."""
        return float(np.interp(a, self.a_values, self.b_values))

    def b_prime(self, a: float) -> float:
        """Return the derivative of the interpolated ``b(a)`` curve."""
        return float(np.interp(a, self.a_values, self.b_derivative))

    def constant_term(self, a: float) -> float:
        """Return the implicit-field constant ``a^4 - b(a)^4``."""
        b = self.b(a)
        return a**4 - b**4

    def radial_extent(self) -> float:
        """Return a safe radius containing every Cassini oval in the interval."""
        return float(np.max(np.sqrt(self.a_values**2 + self.b_values**2)))


@dataclass(frozen=True)
class GridCache:
    """Precomputed geometry terms on a cropped Cartesian grid."""

    nx: int
    ny: int
    dx: float
    x_bounds: Tuple[float, float]
    y_bounds: Tuple[float, float]
    r2: np.ndarray
    r4: np.ndarray
    u2_minus_v2: np.ndarray
    two_uv: np.ndarray
    alpha: np.ndarray
    valid_r: np.ndarray


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the threshold-map example."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute local parameter thresholds that change the Cassini solid mask "
            "for fixed target area and save percent-threshold maps for 4096x512 "
            "and 2048x256."
        )
    )
    parser.add_argument(
        "--num-a",
        type=int,
        default=41,
        help="Number of sampled baseline values in the a direction.",
    )
    parser.add_argument(
        "--num-theta",
        type=int,
        default=41,
        help="Number of sampled baseline values in the rotation direction.",
    )
    parser.add_argument(
        "--b-table-size",
        type=int,
        default=4001,
        help="Interpolation table size used for the target-area b(a) curve.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("examples", "output"),
        help="Directory for saved NPZ data and PNG threshold maps.",
    )
    return parser


def _make_parameter_table(table_size: int) -> ParameterTable:
    """Return an interpolated Cassini ``b(a)`` table over the requested interval."""
    if table_size < 2:
        raise ValueError("b-table-size must be at least 2")

    a_values = np.linspace(A_MIN, A_MAX, table_size, dtype=float)
    b_values = np.array(
        [cassini_b_from_area(a, TARGET_AREA) for a in a_values],
        dtype=float,
    )
    b_derivative = np.gradient(b_values, a_values)
    return ParameterTable(
        a_values=a_values, b_values=b_values, b_derivative=b_derivative
    )


def _build_grid_cache(nx: int, ny: int, table: ParameterTable) -> GridCache:
    """Return cropped grid terms used in the threshold calculations."""
    dx = DOMAIN_X / nx
    dy = DOMAIN_Y / ny
    if not np.isclose(dx, dy):
        raise ValueError("The requested resolution must produce square cells.")

    radius = table.radial_extent() + 2.0 * dx
    x_min = max(0.0, CENTER_X - radius)
    x_max = min(DOMAIN_X, CENTER_X + radius)
    y_min = max(0.0, CENTER_Y - radius)
    y_max = min(DOMAIN_Y, CENTER_Y + radius)

    x = np.arange(nx, dtype=float) * dx
    y = np.arange(ny, dtype=float) * dy
    x_mask = (x >= x_min) & (x <= x_max)
    y_mask = (y >= y_min) & (y <= y_max)
    xb = x[x_mask]
    yb = y[y_mask]
    X, Y = np.meshgrid(xb, yb, indexing="xy")

    u = X - CENTER_X
    v = Y - CENTER_Y
    r2 = u * u + v * v
    return GridCache(
        nx=nx,
        ny=ny,
        dx=dx,
        x_bounds=(float(xb[0]), float(xb[-1])),
        y_bounds=(float(yb[0]), float(yb[-1])),
        r2=r2,
        r4=r2 * r2,
        u2_minus_v2=u * u - v * v,
        two_uv=2.0 * u * v,
        alpha=np.mod(np.arctan2(v, u), np.pi),
        valid_r=r2 > 0.0,
    )


def _direction_field(cache: GridCache, theta: float) -> np.ndarray:
    """Return the rotated quadratic form entering the Cassini field."""
    return np.cos(2.0 * theta) * cache.u2_minus_v2 + np.sin(2.0 * theta) * cache.two_uv


def _field_from_direction(
    cache: GridCache,
    table: ParameterTable,
    a: float,
    direction_field: np.ndarray,
) -> np.ndarray:
    """Return the implicit Cassini field sampled on the cached grid."""
    return cache.r4 - 2.0 * a * a * direction_field + table.constant_term(a)


def _estimate_a_threshold(
    table: ParameterTable,
    a: float,
    direction_field: np.ndarray,
    field0: np.ndarray,
) -> float:
    """Return a first-order estimate of the local ``a`` threshold."""
    b = table.b(a)
    b_prime = table.b_prime(a)
    derivative = -4.0 * a * direction_field + 4.0 * a**3 - 4.0 * b**3 * b_prime
    valid = np.abs(derivative) > 0.0
    if not np.any(valid):
        return np.inf

    estimates = np.abs(field0[valid]) / np.abs(derivative[valid])
    estimates = estimates[estimates > 0.0]
    if estimates.size == 0:
        return 0.0
    return float(np.min(estimates))


def _exact_a_threshold(
    cache: GridCache,
    table: ParameterTable,
    a0: float,
    direction_field: np.ndarray,
    field0: np.ndarray,
) -> float:
    """Return the smallest signed-magnitude change in ``a`` that flips a voxel."""
    base_mask = field0 <= 0.0
    estimate = _estimate_a_threshold(table, a0, direction_field, field0)
    best = np.inf

    for direction in (-1.0, 1.0):
        limit = (a0 - A_MIN) if direction < 0.0 else (A_MAX - a0)
        if limit <= 0.0:
            continue

        step = estimate if np.isfinite(estimate) and estimate > 0.0 else 1.0e-8
        step = min(limit, max(step, 1.0e-10))
        prev = 0.0

        while True:
            trial_a = a0 + direction * step
            trial_mask = (
                _field_from_direction(cache, table, trial_a, direction_field) <= 0.0
            )
            changed = np.any(trial_mask != base_mask)
            if changed:
                lo = prev
                hi = step
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    trial_a = a0 + direction * mid
                    trial_mask = (
                        _field_from_direction(cache, table, trial_a, direction_field)
                        <= 0.0
                    )
                    if np.any(trial_mask != base_mask):
                        hi = mid
                    else:
                        lo = mid
                best = min(best, hi)
                break

            prev = step
            if step >= limit:
                break
            step = min(limit, 2.0 * step)

    return float(best)


def _theta_transition_roots(
    cache: GridCache,
    table: ParameterTable,
    a: float,
) -> np.ndarray:
    """Return all rotation angles where some sampled cell lies on the boundary."""
    b = table.b(a)
    target = np.full_like(cache.r2, np.nan, dtype=float)
    target[cache.valid_r] = (cache.r4[cache.valid_r] + a**4 - b**4) / (
        2.0 * a * a * cache.r2[cache.valid_r]
    )
    valid = cache.valid_r & (np.abs(target) <= 1.0)
    if not np.any(valid):
        return np.empty(0, dtype=float)

    delta = 0.5 * np.arccos(np.clip(target[valid], -1.0, 1.0))
    roots = np.concatenate(
        (
            np.mod(cache.alpha[valid] + delta, np.pi).ravel(),
            np.mod(cache.alpha[valid] - delta, np.pi).ravel(),
        )
    )
    roots = roots[(roots >= THETA_MIN) & (roots <= THETA_MAX)]
    if roots.size == 0:
        return roots
    return np.sort(roots)


def _nearest_root_distances(roots: np.ndarray, theta_values: np.ndarray) -> np.ndarray:
    """Return the nearest-root distance for each sampled baseline rotation."""
    distances = np.full(theta_values.shape, np.inf, dtype=float)
    if roots.size == 0:
        return distances

    indices = np.searchsorted(roots, theta_values)
    right = indices < roots.size
    if np.any(right):
        distances[right] = np.minimum(
            distances[right],
            np.abs(roots[indices[right]] - theta_values[right]),
        )
    left = indices > 0
    if np.any(left):
        distances[left] = np.minimum(
            distances[left],
            np.abs(theta_values[left] - roots[indices[left] - 1]),
        )
    return distances


def _compute_threshold_maps(
    cache: GridCache,
    table: ParameterTable,
    a_values: np.ndarray,
    theta_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return local ``Δa`` and ``Δtheta`` threshold maps."""
    delta_a = np.empty((a_values.size, theta_values.size), dtype=float)
    delta_theta = np.empty_like(delta_a)
    direction_fields = [_direction_field(cache, theta) for theta in theta_values]

    for i, a in enumerate(a_values):
        print(f"  a-slice {i + 1}/{a_values.size}: a={a:.8f}")
        roots = _theta_transition_roots(cache, table, float(a))
        delta_theta[i, :] = _nearest_root_distances(roots, theta_values)

        for j, theta in enumerate(theta_values):
            direction_field = direction_fields[j]
            field0 = _field_from_direction(cache, table, float(a), direction_field)
            delta_a[i, j] = _exact_a_threshold(
                cache=cache,
                table=table,
                a0=float(a),
                direction_field=direction_field,
                field0=field0,
            )

    return delta_a, delta_theta


def _format_stats(values: np.ndarray) -> str:
    """Return a short percentile summary string for terminal output."""
    flat = values[np.isfinite(values)]
    if flat.size == 0:
        return "no finite values"
    p05, p50, p95 = np.percentile(flat, [5.0, 50.0, 95.0])
    return (
        f"min={flat.min():.3e}, p05={p05:.3e}, median={p50:.3e}, "
        f"p95={p95:.3e}, max={flat.max():.3e}"
    )


def _relative_percent_maps(
    a_values: np.ndarray,
    theta_values: np.ndarray,
    delta_a: np.ndarray,
    delta_theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return local threshold maps normalized by the baseline parameters."""
    delta_a_percent = 100.0 * delta_a / a_values[:, None]
    delta_theta_percent = 100.0 * delta_theta / theta_values[None, :]
    return delta_a_percent, delta_theta_percent


def _plot_threshold_maps(
    resolution: Tuple[int, int],
    a_values: np.ndarray,
    theta_values: np.ndarray,
    delta_a_percent: np.ndarray,
    delta_theta_percent: np.ndarray,
    out_dir: str,
) -> str:
    """Render a two-panel relative-threshold map for one resolution."""
    theta_degrees = np.rad2deg(theta_values)
    extent = (
        float(theta_degrees[0]),
        float(theta_degrees[-1]),
        float(a_values[0]),
        float(a_values[-1]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    panels = (
        (delta_a_percent, "Minimal relative Δa [%]", "Δa / a [%]"),
        (
            delta_theta_percent,
            "Minimal relative Δθ [%]",
            "Δθ / θ [%]",
        ),
    )

    for ax, (data, title, colorbar_label) in zip(axes, panels):
        masked = np.ma.masked_less_equal(data, 0.0)
        positive = data[data > 0.0]
        if positive.size == 0:
            raise ValueError("Threshold map does not contain positive values.")

        image = ax.imshow(
            masked,
            origin="lower",
            extent=extent,
            aspect="auto",
            norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("Rotation θ [deg]")
        ax.set_ylabel("Standard Cassini a")
        fig.colorbar(image, ax=ax, label=colorbar_label)

    nx, ny = resolution
    fig.suptitle(
        f"Cassini relative solid-mask threshold maps, area={TARGET_AREA}, "
        f"grid={nx}x{ny}"
    )
    filename = f"cassini_threshold_percent_maps_{nx}x{ny}.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_threshold_data(
    resolution: Tuple[int, int],
    cache: GridCache,
    a_values: np.ndarray,
    theta_values: np.ndarray,
    delta_a: np.ndarray,
    delta_theta: np.ndarray,
    delta_a_percent: np.ndarray,
    delta_theta_percent: np.ndarray,
    out_dir: str,
) -> str:
    """Write the threshold arrays and metadata to an ``.npz`` archive."""
    nx, ny = resolution
    filename = f"cassini_threshold_percent_data_{nx}x{ny}.npz"
    path = os.path.join(out_dir, filename)
    np.savez(
        path,
        a_values=a_values,
        theta_values=theta_values,
        delta_a=delta_a,
        delta_theta=delta_theta,
        delta_a_percent=delta_a_percent,
        delta_theta_percent=delta_theta_percent,
        target_area=np.array(TARGET_AREA),
        domain_x=np.array(DOMAIN_X),
        domain_y=np.array(DOMAIN_Y),
        center_x=np.array(CENTER_X),
        center_y=np.array(CENTER_Y),
        dx=np.array(cache.dx),
        nx=np.array(cache.nx),
        ny=np.array(cache.ny),
        theta_min=np.array(THETA_MIN),
        theta_max=np.array(THETA_MAX),
        a_min=np.array(A_MIN),
        a_max=np.array(A_MAX),
    )
    return path


def _run_resolution(
    resolution: Tuple[int, int],
    table: ParameterTable,
    a_values: np.ndarray,
    theta_values: np.ndarray,
    out_dir: str,
) -> None:
    """Compute, save, and summarize threshold maps for one resolution."""
    cache = _build_grid_cache(*resolution, table=table)
    delta_a, delta_theta = _compute_threshold_maps(cache, table, a_values, theta_values)
    delta_a_percent, delta_theta_percent = _relative_percent_maps(
        a_values=a_values,
        theta_values=theta_values,
        delta_a=delta_a,
        delta_theta=delta_theta,
    )
    data_path = _save_threshold_data(
        resolution=resolution,
        cache=cache,
        a_values=a_values,
        theta_values=theta_values,
        delta_a=delta_a,
        delta_theta=delta_theta,
        delta_a_percent=delta_a_percent,
        delta_theta_percent=delta_theta_percent,
        out_dir=out_dir,
    )
    plot_path = _plot_threshold_maps(
        resolution=resolution,
        a_values=a_values,
        theta_values=theta_values,
        delta_a_percent=delta_a_percent,
        delta_theta_percent=delta_theta_percent,
        out_dir=out_dir,
    )

    nx, ny = resolution
    print(f"Computed threshold maps for grid {nx}x{ny} (dx={cache.dx:.8f}).")
    print(f"  Δa / a [%] stats: {_format_stats(delta_a_percent)}")
    print(f"  Δθ / θ [%] stats: {_format_stats(delta_theta_percent)}")
    print(f"  Saved data: {data_path}")
    print(f"  Saved plot: {plot_path}")


def main() -> None:
    """Run the Cassini threshold-map analysis for both requested resolutions."""
    args = _build_parser().parse_args()
    if args.num_a < 2:
        raise ValueError("--num-a must be at least 2")
    if args.num_theta < 2:
        raise ValueError("--num-theta must be at least 2")

    os.makedirs(args.out_dir, exist_ok=True)
    table = _make_parameter_table(args.b_table_size)
    a_values = np.linspace(A_MIN, A_MAX, args.num_a, dtype=float)
    theta_values = np.linspace(THETA_MIN, THETA_MAX, args.num_theta, dtype=float)

    print(
        "Cassini voxel-change threshold analysis\n"
        f"  target area: {TARGET_AREA}\n"
        f"  a range: [{A_MIN}, {A_MAX}]\n"
        f"  theta range: [{THETA_MIN:.8f}, {THETA_MAX:.8f}] rad\n"
        "  reported thresholds: 100 * Δa / a and 100 * Δθ / θ\n"
        f"  sampled grid in parameter space: {args.num_a} x {args.num_theta}\n"
        f"  output directory: {args.out_dir}"
    )

    for resolution in RESOLUTIONS:
        _run_resolution(
            resolution=resolution,
            table=table,
            a_values=a_values,
            theta_values=theta_values,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
