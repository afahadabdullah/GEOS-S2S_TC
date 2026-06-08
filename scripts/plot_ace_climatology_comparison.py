#!/usr/bin/env python3
"""Make climatological ACE diagnostics from cached GEOS ACE and IBTrACS."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "geos_s2s_tc_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from ocean_mask_utils import add_ocean_only_args
from calculate_tc_conditioned_ace import read_cache
from plot_ace_yearly_timeseries import (
    BASIN_ORDER,
    BASINS,
    DEFAULT_IBTRACS,
    add_ibtracs_to_rows,
    discover_cache_files,
    parse_months,
    parse_years,
    read_ibtracs_observed_ace,
    setup_style,
    summarize_caches,
)


DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_PLOT_DIR = "plots/ace_climatology_comparison"


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    return float(np.nanmean(array)) if array.size else float("nan")


def finite_std(values: list[float]) -> float:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    return float(np.nanstd(array)) if array.size > 1 else 0.0


def climatology_rows(yearly_rows: list[dict[str, object]], used_years: list[int]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in yearly_rows:
        grouped[str(row["basin_name"])].append(row)

    output: list[dict[str, object]] = []
    for basin_name in BASIN_ORDER + ["All Basins"]:
        rows = grouped.get(basin_name, [])
        geos_values = [float(row.get("mean_ace", np.nan)) for row in rows]
        obs_values = [float(row.get("ibtracs_ace", np.nan)) for row in rows]
        thresholds = [float(row.get("threshold_kt", np.nan)) for row in rows]
        geos_mean = finite_mean(geos_values)
        obs_mean = finite_mean(obs_values)
        output.append(
            {
                "basin_name": basin_name,
                "n_years": len(rows),
                "years": ",".join(str(year) for year in used_years),
                "geos_mean_ace": geos_mean,
                "geos_std_ace": finite_std(geos_values),
                "ibtracs_mean_ace": obs_mean,
                "ibtracs_std_ace": finite_std(obs_values),
                "geos_to_ibtracs_ratio": geos_mean / obs_mean if obs_mean > 0.0 and np.isfinite(geos_mean) else float("nan"),
                "threshold_kt": finite_mean(thresholds),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "basin_name",
        "n_years",
        "years",
        "geos_mean_ace",
        "geos_std_ace",
        "ibtracs_mean_ace",
        "ibtracs_std_ace",
        "geos_to_ibtracs_ratio",
        "threshold_kt",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Wrote {path}")


def row_for_basin(rows: list[dict[str, object]], basin_name: str) -> dict[str, object]:
    for row in rows:
        if row["basin_name"] == basin_name:
            return row
    raise KeyError(basin_name)


def plot_climatology_bars(rows: list[dict[str, object]], path: Path, dpi: int) -> None:
    basins = BASIN_ORDER
    x = np.arange(len(basins))
    width = 0.36
    geos = np.asarray([float(row_for_basin(rows, basin)["geos_mean_ace"]) for basin in basins], dtype="float64")
    obs = np.asarray([float(row_for_basin(rows, basin)["ibtracs_mean_ace"]) for basin in basins], dtype="float64")
    geos_std = np.asarray([float(row_for_basin(rows, basin)["geos_std_ace"]) for basin in basins], dtype="float64")
    obs_std = np.asarray([float(row_for_basin(rows, basin)["ibtracs_std_ace"]) for basin in basins], dtype="float64")

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=dpi)
    ax.bar(x - width / 2, obs, width, color="#344e86", label="IBTrACS", yerr=obs_std, capsize=3)
    ax.bar(x + width / 2, geos, width, color="#e55934", label="GEOS", yerr=geos_std, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(basins, rotation=28, ha="right")
    ax.set_ylabel("Mean Sep+Oct ACE (10$^4$ kt$^2$)")
    ax.set_title("Climatological ACE by Basin", fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_ratio_bars(rows: list[dict[str, object]], path: Path, dpi: int) -> None:
    basins = BASIN_ORDER
    ratios = np.asarray([float(row_for_basin(rows, basin)["geos_to_ibtracs_ratio"]) for basin in basins], dtype="float64")
    colors = [BASINS[basin]["color"] for basin in basins]

    fig, ax = plt.subplots(figsize=(12, 5.3), dpi=dpi)
    ax.bar(np.arange(len(basins)), ratios, color=colors)
    ax.axhline(1.0, color="#2a2d34", linewidth=1.3, linestyle="--")
    ax.set_xticks(np.arange(len(basins)))
    ax.set_xticklabels(basins, rotation=28, ha="right")
    ax.set_ylabel("GEOS / IBTrACS mean ACE")
    ax.set_title("GEOS-to-IBTrACS Climatological ACE Ratio", fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def spatial_climatology(used_files: dict[int, list[Path]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yearly_grids: list[np.ndarray] = []
    ref_lats = None
    ref_lons = None

    for year, paths in sorted(used_files.items()):
        grids = []
        for path in paths:
            _, _, lats, lons, _, local_ace, _, _, _ = read_cache(path)
            if ref_lats is None:
                ref_lats = lats
                ref_lons = lons
            elif not np.allclose(ref_lats, lats) or not np.allclose(ref_lons, lons):
                print(f"WARNING: skipping spatial cache with mismatched grid: {path.name}")
                continue
            grids.append(np.asarray(local_ace, dtype="float64"))
        if grids:
            yearly_grids.append(np.nanmean(np.stack(grids, axis=0), axis=0))

    if ref_lats is None or ref_lons is None or not yearly_grids:
        raise RuntimeError("No spatial ACE grids could be read from selected caches")
    return ref_lats, ref_lons, np.nanmean(np.stack(yearly_grids, axis=0), axis=0)


def plot_spatial_map(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    ace_mean: np.ndarray,
    path: Path,
    dpi: int,
    min_lat: float,
    max_lat: float,
) -> None:
    lons_shifted = (longitudes + 180.0) % 360.0 - 180.0
    sort_idx = np.argsort(lons_shifted)
    lons_plot = lons_shifted[sort_idx]
    ace_plot = np.asarray(ace_mean[:, sort_idx], dtype="float64")
    lat_mask = (latitudes >= min_lat) & (latitudes <= max_lat)
    masked = np.ma.masked_where((ace_plot <= 0.0) | ~np.isfinite(ace_plot), ace_plot)
    finite = np.asarray(masked[lat_mask, :].compressed(), dtype="float64")
    vmax = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.0
    vmax = max(vmax, 0.1)

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(13.5, 4.8), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180.0))
        ax.set_extent([0, 360, min_lat, max_lat], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#dff0f8", zorder=0)
        mesh = ax.pcolormesh(
            lons_plot,
            latitudes,
            masked,
            cmap="magma_r",
            norm=LogNorm(vmin=max(np.nanmin(finite), 1.0e-3) if finite.size else 1.0e-3, vmax=vmax),
            transform=ccrs.PlateCarree(),
            shading="auto",
            zorder=1,
        )
        ax.add_feature(cfeature.LAND, facecolor="#eae6df", edgecolor="#777777", linewidth=0.35, zorder=2)
        ax.coastlines(linewidth=0.45, color="#555555", zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#9aa0a6", alpha=0.45, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        fig, ax = plt.subplots(figsize=(13.5, 4.8), dpi=dpi)
        mesh = ax.pcolormesh(
            lons_plot,
            latitudes,
            masked,
            cmap="magma_r",
            norm=LogNorm(vmin=max(np.nanmin(finite), 1.0e-3) if finite.size else 1.0e-3, vmax=vmax),
            shading="auto",
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.25)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08, shrink=0.72)
    cbar.set_label("Mean GEOS local ACE (10$^4$ kt$^2$)")
    ax.set_title("GEOS TC-Conditioned Local ACE Climatology", fontsize=14, fontweight="bold", pad=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--years", default="1991:2024")
    parser.add_argument("--cache-kind", choices=("auto", "lagged", "init", "all"), default="auto")
    parser.add_argument("--prefix", default="geos_ibtracs_ace_climatology")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS)
    parser.add_argument("--months", default="09:10")
    parser.add_argument("--wind-var", default="usa_wind")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument("--nature-filter", default="TS")
    parser.add_argument("--all-natures", dest="nature_filter", action="store_const", const="")
    parser.add_argument("--basin-method", choices=("boxes", "ibtracs_code"), default="boxes")
    parser.add_argument("--all-hours", dest="synoptic_only", action="store_false")
    parser.add_argument("--min-lat", type=float, default=-25.0)
    parser.add_argument("--max-lat", type=float, default=50.0)
    parser.set_defaults(synoptic_only=True)
    add_ocean_only_args(parser)
    args = parser.parse_args(argv)

    setup_style()
    years = parse_years(args.years) if args.years else set()
    cache_dir = Path(args.cache_dir)
    plot_dir = Path(args.plot_dir)
    files_by_year = discover_cache_files(cache_dir, years, args.cache_kind)
    if not files_by_year:
        print(f"ERROR: no ACE ensmean caches found in {cache_dir}", file=sys.stderr)
        return 1

    yearly_rows, used_files = summarize_caches(files_by_year)
    used_years = sorted(used_files)
    if not used_years:
        print(f"ERROR: no usable ACE caches found in {cache_dir}", file=sys.stderr)
        return 1
    if years:
        missing = sorted(years - set(used_years))
        if missing:
            print(f"Missing requested years with usable caches: {', '.join(str(year) for year in missing)}")
    print(f"Using GEOS ACE years: {', '.join(str(year) for year in used_years)}")

    ibtracs_path = Path(args.ibtracs) if args.ibtracs else None
    if ibtracs_path is not None:
        months = parse_months(args.months)
        print(
            "Reading IBTrACS observed ACE: "
            f"years={used_years[0]}:{used_years[-1]}, months={','.join(sorted(months))}, "
            f"wind={args.wind_var}, threshold={args.threshold_kt:g} kt, nature={args.nature_filter or 'ALL'}"
        )
        ibtracs_ace, ibtracs_fix_counts = read_ibtracs_observed_ace(
            ibtracs_path,
            years=set(used_years),
            months=months,
            wind_var=args.wind_var,
            threshold_kt=args.threshold_kt,
            nature_filter=args.nature_filter,
            basin_method=args.basin_method,
            synoptic_only=args.synoptic_only,
            ocean_only=args.ocean_only,
            ocean_mask_source=args.ocean_mask_source,
            ocean_mask_file=args.ocean_mask_file,
            ocean_threshold=args.ocean_threshold,
        )
        add_ibtracs_to_rows(yearly_rows, ibtracs_ace, ibtracs_fix_counts)

    clim_rows = climatology_rows(yearly_rows, used_years)
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_csv(plot_dir / f"{args.prefix}.csv", clim_rows)
    plot_climatology_bars(clim_rows, plot_dir / f"{args.prefix}_by_basin.png", args.dpi)
    plot_ratio_bars(clim_rows, plot_dir / f"{args.prefix}_geos_to_ibtracs_ratio.png", args.dpi)

    lats, lons, ace_mean = spatial_climatology(used_files)
    plot_spatial_map(
        latitudes=lats,
        longitudes=lons,
        ace_mean=ace_mean,
        path=plot_dir / f"{args.prefix}_geos_spatial_mean_map.png",
        dpi=args.dpi,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
