#!/usr/bin/env python3
"""Plot diagnostics from cached GEOS TC-candidate CSV files.

This reads the candidate inventory written by ``calculate_geos_candidate_thresholds.py``
and makes quick-look plots without reopening GEOS NetCDF files. It is safe to run
on a partially written candidate CSV while the long calibration job is still
running: it uses whatever complete rows are available at read time.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
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
from ocean_mask_utils import add_ocean_only_args, build_ocean_checker, row_over_ocean_value

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


BASINS = {
    "North Atlantic": {
        "lat_range": (0.0, 45.0),
        "lon_range": (-100.0, -10.0),
        "color": "#e55934",
        "label_xy": (-55.0, 38.0),
    },
    "Northeast Pacific": {
        "lat_range": (0.0, 40.0),
        "lon_range": (-180.0, -100.0),
        "color": "#f3a712",
        "label_xy": (-135.0, 32.0),
    },
    "Northwest Pacific": {
        "lat_range": (0.0, 45.0),
        "lon_range": (100.0, 180.0),
        "color": "#2ec4b6",
        "label_xy": (150.0, 30.0),
    },
    "North Indian": {
        "lat_range": (0.0, 40.0),
        "lon_range": (40.0, 100.0),
        "color": "#9b5de5",
        "label_xy": (65.0, 17.0),
    },
    "South Indian": {
        "lat_range": (-25.0, 0.0),
        "lon_range": (20.0, 135.0),
        "color": "#00bbf9",
        "label_xy": (80.0, -15.0),
    },
    "South Pacific": {
        "lat_range": (-25.0, 0.0),
        "lon_ranges": [(135.0, 180.0), (-180.0, -120.0)],
        "color": "#ff007f",
        "label_xy": (-155.0, -15.0),
    },
}

BASIN_ORDER = list(BASINS)
DEFAULT_CANDIDATES = "data/calibration/*_candidates.csv"
DEFAULT_OBS_PERCENTILES = "data/obs/ibtracs/ibtracs_observed_percentiles.csv"
DEFAULT_PLOT_DIR = "plots/geos_candidate_diagnostics"
MONTH_LABELS = {
    "01": "Jan",
    "02": "Feb",
    "03": "Mar",
    "04": "Apr",
    "05": "May",
    "06": "Jun",
    "07": "Jul",
    "08": "Aug",
    "09": "Sep",
    "10": "Oct",
    "11": "Nov",
    "12": "Dec",
}


@dataclass
class Candidate:
    source_file: str
    init_date: str
    ens: str
    valid_time: datetime | None
    forecast_month: str
    basin_name: str
    center_lat: float
    center_lon: float
    vmax_kt: float
    slp_hpa: float
    slp_anom_hpa: float
    warm_core_anom_k: float
    qv850_anom_gpkg: float
    vort850_s1: float
    used_vorticity: int
    over_ocean: bool | None


def configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except (AttributeError, TypeError):
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=[DEFAULT_CANDIDATES],
        help="Candidate CSV path(s) or glob(s). Default: data/calibration/*_candidates.csv",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        default=None,
        help="Optional threshold summary CSV path(s) or glob(s). Default: infer from candidate filenames.",
    )
    parser.add_argument("--observed-percentiles", default=DEFAULT_OBS_PERCENTILES)
    parser.add_argument("--observed-wind-var", default="usa_wind")
    parser.add_argument("--observed-basin-method", default="boxes")
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Default derives from candidate file names.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min-lat", type=float, default=-25.0, help="Minimum candidate latitude to plot. Default trims south of 25S.")
    parser.add_argument("--max-lat", type=float, default=50.0, help="Maximum candidate latitude to plot.")
    parser.add_argument("--density-bin-deg", type=float, default=2.0, help="Lat/lon bin size for the global density map.")
    add_ocean_only_args(parser)
    parser.add_argument(
        "--months",
        default="",
        help="Optional month filter such as 09:10. Default uses all candidate rows in the CSV.",
    )
    return parser.parse_args(argv)


def parse_months(value: str) -> set[str]:
    months: set[str] = set()
    for item in value.replace(",", " ").replace(":", " ").split():
        try:
            month = int(item)
        except ValueError:
            continue
        if 1 <= month <= 12:
            months.add(f"{month:02d}")
    return months


def expand_paths(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [str(candidate)]
        for match in sorted(matches):
            path = Path(match)
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def infer_threshold_paths(candidate_paths: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate_path in candidate_paths:
        name = candidate_path.name
        if not name.endswith("_candidates.csv"):
            continue
        threshold_path = candidate_path.with_name(name.replace("_candidates.csv", ".csv"))
        if threshold_path.exists() and threshold_path not in seen:
            paths.append(threshold_path)
            seen.add(threshold_path)
    return paths


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def lon_to_180(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def read_candidates(paths: list[Path], months: set[str]) -> list[Candidate]:
    candidates: list[Candidate] = []
    skipped = 0
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                basin_name = row.get("basin_name", "")
                if basin_name not in BASINS:
                    skipped += 1
                    continue

                valid_time = parse_datetime(row.get("valid_time"))
                forecast_month = row.get("forecast_month") or (valid_time.strftime("%m") if valid_time else "")
                if months and forecast_month not in months:
                    continue

                center_lat = parse_float(row.get("center_lat"))
                center_lon = lon_to_180(parse_float(row.get("center_lon")))
                vmax_kt = parse_float(row.get("vmax_kt"))
                if not (np.isfinite(center_lat) and np.isfinite(center_lon) and np.isfinite(vmax_kt)):
                    skipped += 1
                    continue
                if center_lat < -25.0:
                    continue

                candidates.append(
                    Candidate(
                        source_file=str(path),
                        init_date=row.get("init_date", ""),
                        ens=row.get("ens", ""),
                        valid_time=valid_time,
                        forecast_month=forecast_month,
                        basin_name=basin_name,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        vmax_kt=vmax_kt,
                        slp_hpa=parse_float(row.get("slp_hpa")),
                        slp_anom_hpa=parse_float(row.get("slp_anom_hpa")),
                        warm_core_anom_k=parse_float(row.get("warm_core_anom_k")),
                        qv850_anom_gpkg=parse_float(row.get("qv850_anom_gpkg")),
                        vort850_s1=parse_float(row.get("vort850_s1")),
                        used_vorticity=parse_int(row.get("used_vorticity")),
                        over_ocean=row_over_ocean_value(row),
                    )
                )
    if skipped:
        print(f"Skipped {skipped} incomplete or unrecognized candidate rows.")
    return candidates


def filter_candidates_by_ocean(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    if not args.ocean_only:
        return candidates

    checker, warning = build_ocean_checker(
        args.ocean_mask_source,
        mask_file=args.ocean_mask_file,
        threshold=args.ocean_threshold,
        require_mask=True,
    )
    print(f"Ocean-only candidate filter enabled: source={checker.source}")
    if warning:
        print(f"WARNING: ocean mask fallback: {warning}")

    filtered: list[Candidate] = []
    skipped_land = 0
    for candidate in candidates:
        is_ocean = False if candidate.over_ocean is False else checker.is_ocean(candidate.center_lat, candidate.center_lon)
        if is_ocean:
            filtered.append(candidate)
        else:
            skipped_land += 1

    print(f"Skipped {skipped_land:,} GEOS candidate rows over land.")
    return filtered


def build_plot_ocean_checker(args: argparse.Namespace):
    if not args.ocean_only:
        return None
    checker, warning = build_ocean_checker(
        args.ocean_mask_source,
        mask_file=args.ocean_mask_file,
        threshold=args.ocean_threshold,
        require_mask=True,
    )
    if warning:
        print(f"WARNING: ocean mask fallback: {warning}")
    return checker


def read_thresholds(paths: list[Path], ocean_only: bool) -> dict[str, dict[str, float | str]]:
    thresholds: dict[str, dict[str, float | str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                basin_name = row.get("basin_name", "")
                if basin_name not in BASINS:
                    continue
                threshold = parse_float(row.get("geos_threshold_kt"))
                if not np.isfinite(threshold):
                    continue
                if ocean_only and row.get("ocean_only") != "1":
                    continue
                thresholds[basin_name] = {
                    "threshold_kt": threshold,
                    "observed_percentile": parse_float(row.get("observed_percentile")),
                    "source": str(path),
                    "source_kind": "threshold_csv",
                }
    return thresholds


def read_observed_percentiles(path: Path, wind_var: str, basin_method: str) -> dict[str, float]:
    percentiles: dict[str, float] = {}
    if not path.exists():
        return percentiles
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basin_name = row.get("basin_name", "")
            if basin_name not in BASINS:
                continue
            if row.get("wind_var") != wind_var:
                continue
            if basin_method and row.get("basin_method") != basin_method:
                continue
            percentile = parse_float(row.get("percentile_obs_threshold"))
            if np.isfinite(percentile):
                percentiles[basin_name] = percentile
    return percentiles


def fill_provisional_thresholds(
    thresholds: dict[str, dict[str, float | str]],
    candidates: list[Candidate],
    observed_percentiles: dict[str, float],
    observed_path: Path,
) -> dict[str, dict[str, float | str]]:
    values_by_basin: dict[str, list[float]] = defaultdict(list)
    for candidate in candidates:
        values_by_basin[candidate.basin_name].append(candidate.vmax_kt)

    filled = dict(thresholds)
    for basin_name in BASIN_ORDER:
        if basin_name in filled:
            continue
        percentile = observed_percentiles.get(basin_name)
        values = np.asarray(values_by_basin.get(basin_name, []), dtype="float64")
        values = values[np.isfinite(values)]
        if percentile is None or values.size == 0:
            continue
        filled[basin_name] = {
            "threshold_kt": float(np.nanpercentile(values, min(100.0, max(0.0, percentile)))),
            "observed_percentile": float(percentile),
            "source": str(observed_path),
            "source_kind": "provisional_from_candidates",
        }
    return filled


def filter_candidates_by_latitude(candidates: list[Candidate], min_lat: float, max_lat: float) -> list[Candidate]:
    return [
        candidate
        for candidate in candidates
        if np.isfinite(candidate.center_lat) and min_lat <= candidate.center_lat <= max_lat
    ]


def derive_prefix(candidate_paths: list[Path]) -> str:
    if len(candidate_paths) == 1:
        name = candidate_paths[0].stem
        return name[:-11] if name.endswith("_candidates") else name
    if not candidate_paths:
        return "geos_candidate_diagnostics"
    stems = [path.stem.replace("_candidates", "") for path in candidate_paths]
    common = os.path.commonprefix(stems).strip("_-.")
    return common or "geos_candidate_diagnostics"


def setup_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "#ffffff"


def is_cartopy_axis(ax) -> bool:
    return HAS_CARTOPY and hasattr(ax, "projection")


def map_kwargs(ax) -> dict:
    return {"transform": ccrs.PlateCarree()} if is_cartopy_axis(ax) else {}


def draw_land_overlay(ax, zorder: float = 4.0) -> None:
    if not HAS_CARTOPY:
        return
    ax.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=zorder)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.5, zorder=zorder + 0.1)
    ax.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.3, linestyle=":", zorder=zorder + 0.1)


def create_global_map_figure(dpi: int, title: str, min_lat: float, max_lat: float):
    fig = plt.figure(figsize=(14, 7), dpi=dpi)
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180.0))
        ax.set_extent([-180.0, 180.0, min_lat, max_lat], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#daeefb", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.3, linestyle=":", zorder=2)
        gridliner = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="#9aa7b1",
            alpha=0.45,
            linestyle="--",
            zorder=3,
        )
        gridliner.top_labels = False
        gridliner.right_labels = False
        gridliner.xlabel_style = {"size": 8, "color": "#555555"}
        gridliner.ylabel_style = {"size": 8, "color": "#555555"}
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#daeefb")
        ax.set_xlim(-180.0, 180.0)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.35, color="#9aa7b1", alpha=0.45, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#1e222a")
    return fig, ax


def draw_basin_boxes(
    ax,
    label_text_by_basin: dict[str, str] | None = None,
    min_lat: float | None = None,
    max_lat: float | None = None,
    line_zorder: float = 4.0,
    text_zorder: float = 6.0,
) -> None:
    for basin_name, basin_def in BASINS.items():
        color = basin_def["color"]
        lat_min, lat_max = basin_def["lat_range"]
        lon_ranges = [basin_def["lon_range"]] if "lon_range" in basin_def else basin_def["lon_ranges"]
        for lon_min, lon_max in lon_ranges:
            lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
            lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
            ax.plot(lons, lats, color=color, linewidth=1.4, alpha=0.85, zorder=line_zorder, **map_kwargs(ax))
        label_lon, label_lat = basin_def["label_xy"]
        if min_lat is not None:
            label_lat = max(label_lat, min_lat + 4.0)
        if max_lat is not None:
            label_lat = min(label_lat, max_lat - 4.0)
        label_text = label_text_by_basin.get(basin_name, basin_name.replace(" ", "\n")) if label_text_by_basin else basin_name.replace(" ", "\n")
        ax.text(
            label_lon,
            label_lat,
            label_text,
            ha="center",
            va="center",
            color="#1e222a",
            fontsize=7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec=color, lw=0.9, alpha=0.82),
            zorder=text_zorder,
            **map_kwargs(ax),
        )


def no_data_text(ax, message: str = "No candidate rows available yet") -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#666666",
    )


def save_figure(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def print_year_coverage(candidates: list[Candidate], label: str = "GEOS candidate years") -> None:
    years = sorted({candidate.init_date[:4] for candidate in candidates if len(candidate.init_date) >= 4})
    if years:
        print(f"{label}: {years[0]}-{years[-1]} ({len(years)} years): {', '.join(years)}")
    else:
        print(f"{label}: none")


def active_member_keys(candidates: list[Candidate]) -> list[tuple[str, str]]:
    keys = {
        (candidate.init_date, candidate.ens)
        for candidate in candidates
        if candidate.init_date and candidate.ens
    }
    return sorted(keys)


def count_for_member(
    candidates: list[Candidate],
    member_key: tuple[str, str],
    basin_name: str,
    month: str | None = None,
) -> int:
    init_date, ens = member_key
    return sum(
        1
        for candidate in candidates
        if candidate.init_date == init_date
        and candidate.ens == ens
        and candidate.basin_name == basin_name
        and (month is None or candidate.forecast_month == month)
    )


def mean_count_per_active_member(candidates: list[Candidate], basin_name: str, month: str | None = None) -> float:
    members = active_member_keys(candidates)
    if not members:
        return float("nan")
    counts = [count_for_member(candidates, member, basin_name, month) for member in members]
    return float(np.mean(counts)) if counts else float("nan")


def plot_global_candidate_map(
    candidates: list[Candidate],
    thresholds: dict[str, dict[str, float | str]],
    path: Path,
    dpi: int,
    min_lat: float,
    max_lat: float,
) -> None:
    fig, ax = create_global_map_figure(dpi, "GEOS TC Candidate Locations From Cached Inventory", min_lat, max_lat)

    if not candidates:
        draw_basin_boxes(ax, min_lat=min_lat, max_lat=max_lat)
        no_data_text(ax)
        save_figure(fig, path, dpi)
        return

    for basin_name in BASIN_ORDER:
        basin_candidates = [candidate for candidate in candidates if candidate.basin_name == basin_name]
        if not basin_candidates:
            continue
        lons = np.asarray([candidate.center_lon for candidate in basin_candidates])
        lats = np.asarray([candidate.center_lat for candidate in basin_candidates])
        vmax = np.asarray([candidate.vmax_kt for candidate in basin_candidates])
        sizes = np.clip(4.0 + vmax * 0.12, 7.0, 18.0)
        threshold = thresholds.get(basin_name, {}).get("threshold_kt", float("nan"))
        if isinstance(threshold, str):
            threshold = float("nan")
        above = np.isfinite(threshold) & (vmax >= float(threshold))
        ax.scatter(
            lons[~above],
            lats[~above],
            s=sizes[~above],
            c=BASINS[basin_name]["color"],
            alpha=0.34,
            edgecolors="none",
            label=f"{basin_name} ({len(basin_candidates)})",
            zorder=5,
            **map_kwargs(ax),
        )
        if np.any(above):
            ax.scatter(
                lons[above],
                lats[above],
                s=sizes[above],
                c=BASINS[basin_name]["color"],
                alpha=0.78,
                edgecolors="#1e222a",
                linewidths=0.25,
                zorder=7,
                **map_kwargs(ax),
            )

    draw_land_overlay(ax, zorder=8.0)
    draw_basin_boxes(ax, min_lat=min_lat, max_lat=max_lat, line_zorder=9.0, text_zorder=10.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    save_figure(fig, path, dpi)


def plot_global_candidate_density_map(
    candidates: list[Candidate],
    path: Path,
    dpi: int,
    min_lat: float,
    max_lat: float,
    bin_deg: float,
    ocean_checker=None,
) -> None:
    fig, ax = create_global_map_figure(dpi, "GEOS TC Candidate Density From Cached Inventory", min_lat, max_lat)

    if not candidates:
        draw_basin_boxes(ax, min_lat=min_lat, max_lat=max_lat)
        no_data_text(ax)
        save_figure(fig, path, dpi)
        return

    bin_deg = max(0.5, float(bin_deg))
    lon_edges = np.arange(-180.0, 180.0 + bin_deg, bin_deg)
    lat_edges = np.arange(min_lat, max_lat + bin_deg, bin_deg)
    lons = np.asarray([candidate.center_lon for candidate in candidates], dtype="float64")
    lats = np.asarray([candidate.center_lat for candidate in candidates], dtype="float64")
    counts, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
    masked_out = counts <= 0.0
    if ocean_checker is not None:
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        ocean_bins = np.zeros_like(counts, dtype=bool)
        for lat_index, lat_value in enumerate(lat_centers):
            for lon_index, lon_value in enumerate(lon_centers):
                ocean_bins[lat_index, lon_index] = ocean_checker.is_ocean(float(lat_value), float(lon_value))
        masked_out = masked_out | ~ocean_bins
    masked_counts = np.ma.masked_where(masked_out, counts)

    visible_counts = np.ma.filled(masked_counts, 0.0)
    max_count = float(np.nanmax(visible_counts)) if np.nanmax(visible_counts) > 0 else 1.0
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        masked_counts,
        cmap="YlOrRd",
        norm=LogNorm(vmin=1.0, vmax=max_count),
        alpha=0.82,
        zorder=3,
        **map_kwargs(ax),
    )
    draw_land_overlay(ax, zorder=6.0)
    draw_basin_boxes(ax, min_lat=min_lat, max_lat=max_lat, line_zorder=7.0, text_zorder=8.0)
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, aspect=45, shrink=0.72)
    cbar.set_label(f"Candidate count per {bin_deg:g}deg x {bin_deg:g}deg bin", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    save_figure(fig, path, dpi)


def plot_global_basin_count_map(candidates: list[Candidate], path: Path, dpi: int, min_lat: float, max_lat: float) -> None:
    members = active_member_keys(candidates)
    label_text_by_basin: dict[str, str] = {}
    for basin_name in BASIN_ORDER:
        total_count = sum(1 for candidate in candidates if candidate.basin_name == basin_name)
        mean_count = mean_count_per_active_member(candidates, basin_name)
        if np.isfinite(mean_count):
            label_text_by_basin[basin_name] = f"{basin_name}\nN={total_count:,}\nmean/member={mean_count:.2f}"
        else:
            label_text_by_basin[basin_name] = f"{basin_name}\nN={total_count:,}"

    fig, ax = create_global_map_figure(dpi, "GEOS TC Candidate Counts by Basin", min_lat, max_lat)
    if not candidates:
        draw_basin_boxes(ax, label_text_by_basin, min_lat=min_lat, max_lat=max_lat)
        no_data_text(ax)
    else:
        counts = np.asarray([sum(1 for candidate in candidates if candidate.basin_name == basin_name) for basin_name in BASIN_ORDER])
        max_count = float(np.nanmax(counts)) if counts.size and np.nanmax(counts) > 0 else 1.0
        for basin_name, count in zip(BASIN_ORDER, counts):
            label_lon, label_lat = BASINS[basin_name]["label_xy"]
            size = 120.0 + 700.0 * (float(count) / max_count)
            ax.scatter(
                [label_lon],
                [label_lat],
                s=size,
                color=BASINS[basin_name]["color"],
                alpha=0.22,
                edgecolors=BASINS[basin_name]["color"],
                linewidths=1.0,
                zorder=5,
                **map_kwargs(ax),
            )
        draw_land_overlay(ax, zorder=6.0)
        draw_basin_boxes(ax, label_text_by_basin, min_lat=min_lat, max_lat=max_lat, line_zorder=7.0, text_zorder=8.0)
    save_figure(fig, path, dpi)


def plot_counts_by_basin_month(candidates: list[Candidate], path: Path, dpi: int) -> None:
    months = sorted({candidate.forecast_month for candidate in candidates if candidate.forecast_month})
    if not months:
        months = ["09", "10"]
    month_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2"]

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=dpi)
    x = np.arange(len(BASIN_ORDER))
    bottom = np.zeros(len(BASIN_ORDER), dtype="float64")
    for idx, month in enumerate(months):
        counts = np.asarray(
            [
                sum(1 for candidate in candidates if candidate.basin_name == basin_name and candidate.forecast_month == month)
                for basin_name in BASIN_ORDER
            ],
            dtype="float64",
        )
        ax.bar(
            x,
            counts,
            bottom=bottom,
            color=month_colors[idx % len(month_colors)],
            label=MONTH_LABELS.get(month, month),
            width=0.68,
            edgecolor="#ffffff",
            linewidth=0.7,
        )
        bottom += counts

    if not candidates:
        no_data_text(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(BASIN_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Accepted Candidate Count")
    ax.set_title("GEOS TC Candidate Counts by Basin and Forecast Month", fontsize=12, fontweight="bold", pad=10, color="#1e222a")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=max(1, min(len(months), 4)))
    save_figure(fig, path, dpi)


def plot_ensemble_mean_counts_by_basin_month(candidates: list[Candidate], path: Path, dpi: int) -> None:
    months = sorted({candidate.forecast_month for candidate in candidates if candidate.forecast_month})
    if not months:
        months = ["09", "10"]
    month_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2"]
    members = active_member_keys(candidates)

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=dpi)
    x = np.arange(len(BASIN_ORDER))
    bottom = np.zeros(len(BASIN_ORDER), dtype="float64")
    for idx, month in enumerate(months):
        mean_counts = np.asarray(
            [mean_count_per_active_member(candidates, basin_name, month) for basin_name in BASIN_ORDER],
            dtype="float64",
        )
        mean_counts = np.nan_to_num(mean_counts, nan=0.0)
        ax.bar(
            x,
            mean_counts,
            bottom=bottom,
            color=month_colors[idx % len(month_colors)],
            label=MONTH_LABELS.get(month, month),
            width=0.68,
            edgecolor="#ffffff",
            linewidth=0.7,
        )
        bottom += mean_counts

    if not candidates:
        no_data_text(ax)
    else:
        for x_value, total in zip(x, bottom):
            ax.text(x_value, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8, color="#1e222a")
    ax.set_xticks(x)
    ax.set_xticklabels(BASIN_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Mean Accepted Candidates per Active Init/Member")
    ax.set_title(
        "GEOS TC Candidate Counts by Basin and Month, Ensemble-Normalized",
        fontsize=12,
        fontweight="bold",
        pad=10,
        color="#1e222a",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=max(1, min(len(months), 4)))
    ax.text(
        0.01,
        0.98,
        f"Denominator: {len(members):,} active init/member pairs represented in candidate rows.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cccccc", alpha=0.85),
    )
    save_figure(fig, path, dpi)


def plot_vmax_histograms(candidates: list[Candidate], thresholds: dict[str, dict[str, float | str]], path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=dpi, sharex=False, sharey=False)
    axes_flat = axes.ravel()
    for ax, basin_name in zip(axes_flat, BASIN_ORDER):
        values = np.asarray([candidate.vmax_kt for candidate in candidates if candidate.basin_name == basin_name], dtype="float64")
        values = values[np.isfinite(values)]
        color = BASINS[basin_name]["color"]
        if values.size:
            bins = min(24, max(8, int(math.sqrt(values.size)) + 4))
            ax.hist(values, bins=bins, color=color, alpha=0.72, edgecolor="#ffffff", linewidth=0.7)
            ax.axvline(float(np.nanmedian(values)), color="#1e222a", linewidth=1.2, linestyle="--", label="median")
        else:
            no_data_text(ax, "No candidates")

        threshold_info = thresholds.get(basin_name)
        if threshold_info:
            threshold = threshold_info.get("threshold_kt", float("nan"))
            if not isinstance(threshold, str) and np.isfinite(threshold):
                source_kind = threshold_info.get("source_kind", "")
                label = "threshold" if source_kind == "threshold_csv" else "provisional threshold"
                ax.axvline(float(threshold), color="#c1121f", linewidth=1.5, label=label)

        ax.set_title(f"{basin_name}\nn={values.size:,}", fontsize=10, fontweight="bold", color="#1e222a")
        ax.set_xlabel("Candidate Vmax (kt)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=7)
    fig.suptitle("GEOS Candidate Vmax Distributions by Basin", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_year_basin_heatmap(candidates: list[Candidate], path: Path, dpi: int) -> None:
    years = sorted({candidate.init_date[:4] for candidate in candidates if len(candidate.init_date) >= 4})
    fig, ax = plt.subplots(figsize=(max(9.0, 0.28 * max(len(years), 1)), 5.5), dpi=dpi)
    if not candidates or not years:
        no_data_text(ax)
        ax.set_axis_off()
        save_figure(fig, path, dpi)
        return

    matrix = np.zeros((len(BASIN_ORDER), len(years)), dtype="float64")
    year_index = {year: idx for idx, year in enumerate(years)}
    basin_index = {basin_name: idx for idx, basin_name in enumerate(BASIN_ORDER)}
    for candidate in candidates:
        year = candidate.init_date[:4]
        if year in year_index and candidate.basin_name in basin_index:
            matrix[basin_index[candidate.basin_name], year_index[year]] += 1.0

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(np.arange(len(BASIN_ORDER)))
    ax.set_yticklabels(BASIN_ORDER)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years, rotation=90, fontsize=7)
    ax.set_title("GEOS TC Candidate Coverage by Init Year and Basin", fontsize=12, fontweight="bold", pad=10, color="#1e222a")
    ax.set_xlabel("Initialization Year")
    ax.set_ylabel("Basin")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Candidate Count")
    save_figure(fig, path, dpi)


def plot_threshold_bars(
    candidates: list[Candidate],
    thresholds: dict[str, dict[str, float | str]],
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=dpi)
    x = np.arange(len(BASIN_ORDER))
    values = []
    colors = []
    labels = []
    for basin_name in BASIN_ORDER:
        info = thresholds.get(basin_name, {})
        threshold = info.get("threshold_kt", float("nan"))
        if isinstance(threshold, str):
            threshold = float("nan")
        values.append(float(threshold) if np.isfinite(threshold) else np.nan)
        source_kind = info.get("source_kind", "")
        colors.append(BASINS[basin_name]["color"] if source_kind == "threshold_csv" else "#9aa7b1")
        labels.append(source_kind)

    finite_values = np.asarray(values, dtype="float64")
    has_values = np.any(np.isfinite(finite_values))
    bar_values = np.nan_to_num(finite_values, nan=0.0)
    bars = ax.bar(x, bar_values, color=colors, edgecolor="#ffffff", linewidth=0.8, width=0.68)

    if not has_values:
        no_data_text(ax, "No thresholds available yet")
    else:
        for bar, value, basin_name in zip(bars, finite_values, BASIN_ORDER):
            if not np.isfinite(value):
                continue
            n_candidates = sum(1 for candidate in candidates if candidate.basin_name == basin_name)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f} kt\nn={n_candidates:,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1e222a",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(BASIN_ORDER, rotation=20, ha="right")
    ax.set_ylabel("GEOS Candidate Wind Threshold (kt)")
    ax.set_title("Basin-Dependent GEOS Candidate Thresholds", fontsize=12, fontweight="bold", pad=10, color="#1e222a")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        0.98,
        "Colored bars: threshold CSV. Gray bars: provisional from current candidates + observed percentile.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cccccc", alpha=0.85),
    )
    save_figure(fig, path, dpi)


def finite_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "min": float(np.nanmin(array)),
        "median": float(np.nanmedian(array)),
        "p90": float(np.nanpercentile(array, 90)),
        "max": float(np.nanmax(array)),
    }


def write_summary_csv(
    candidates: list[Candidate],
    thresholds: dict[str, dict[str, float | str]],
    path: Path,
) -> None:
    months = sorted({candidate.forecast_month for candidate in candidates if candidate.forecast_month})
    month_fields = [f"month_{month}_count" for month in months]
    month_mean_fields = [f"month_{month}_member_mean_count" for month in months]
    active_members = active_member_keys(candidates)
    fieldnames = [
        "basin_name",
        "candidate_count",
        *month_fields,
        "active_init_member_pairs",
        "member_mean_count",
        *month_mean_fields,
        "n_init_dates",
        "n_ensembles",
        "min_vmax_kt",
        "median_vmax_kt",
        "p90_vmax_kt",
        "max_vmax_kt",
        "threshold_kt",
        "threshold_source_kind",
        "observed_percentile",
        "above_threshold_count",
        "above_threshold_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for basin_name in BASIN_ORDER:
            basin_candidates = [candidate for candidate in candidates if candidate.basin_name == basin_name]
            values = [candidate.vmax_kt for candidate in basin_candidates]
            stats = finite_stats(values)
            threshold_info = thresholds.get(basin_name, {})
            threshold = threshold_info.get("threshold_kt", float("nan"))
            if isinstance(threshold, str):
                threshold = float("nan")
            if np.isfinite(threshold):
                above_count = sum(1 for candidate in basin_candidates if candidate.vmax_kt >= float(threshold))
                above_fraction = above_count / len(basin_candidates) if basin_candidates else float("nan")
            else:
                above_count = 0
                above_fraction = float("nan")

            row = {
                "basin_name": basin_name,
                "candidate_count": len(basin_candidates),
                "active_init_member_pairs": len(active_members),
                "member_mean_count": mean_count_per_active_member(candidates, basin_name),
                "n_init_dates": len({candidate.init_date for candidate in basin_candidates if candidate.init_date}),
                "n_ensembles": len({candidate.ens for candidate in basin_candidates if candidate.ens}),
                "min_vmax_kt": stats["min"],
                "median_vmax_kt": stats["median"],
                "p90_vmax_kt": stats["p90"],
                "max_vmax_kt": stats["max"],
                "threshold_kt": threshold,
                "threshold_source_kind": threshold_info.get("source_kind", ""),
                "observed_percentile": threshold_info.get("observed_percentile", float("nan")),
                "above_threshold_count": above_count,
                "above_threshold_fraction": above_fraction,
            }
            for month in months:
                row[f"month_{month}_count"] = sum(1 for candidate in basin_candidates if candidate.forecast_month == month)
                row[f"month_{month}_member_mean_count"] = mean_count_per_active_member(candidates, basin_name, month)
            writer.writerow(row)
    print(f"  -> Wrote {path}")


def print_console_summary(candidates: list[Candidate], thresholds: dict[str, dict[str, float | str]]) -> None:
    print("")
    print("GEOS candidate diagnostic summary")
    print(f"{'basin':20s} {'n':>8s} {'mean/member':>12s} {'median':>8s} {'p90':>8s} {'threshold':>10s} {'source':>14s}")
    for basin_name in BASIN_ORDER:
        values = [candidate.vmax_kt for candidate in candidates if candidate.basin_name == basin_name]
        stats = finite_stats(values)
        mean_count = mean_count_per_active_member(candidates, basin_name)
        threshold_info = thresholds.get(basin_name, {})
        threshold = threshold_info.get("threshold_kt", float("nan"))
        if isinstance(threshold, str):
            threshold = float("nan")
        print(
            f"{basin_name:20s} {len(values):8d} "
            f"{mean_count:12.2f} "
            f"{stats['median']:8.2f} {stats['p90']:8.2f} "
            f"{float(threshold):10.2f} {str(threshold_info.get('source_kind', '')):>14s}"
        )


def main(argv: list[str] | None = None) -> int:
    configure_streams()
    args = parse_args(argv)
    setup_style()

    candidate_paths = expand_paths(args.candidates)
    if not candidate_paths:
        print(f"ERROR: no candidate CSV files matched: {', '.join(args.candidates)}", file=sys.stderr)
        return 1

    months = parse_months(args.months)
    print("Reading candidate CSV files:")
    for path in candidate_paths:
        print(f"  - {path}")
    all_candidates = read_candidates(candidate_paths, months)
    all_candidates = filter_candidates_by_ocean(all_candidates, args)
    print_year_coverage(all_candidates, "GEOS candidate init years after month/ocean filters")
    candidates = filter_candidates_by_latitude(all_candidates, args.min_lat, args.max_lat)
    print_year_coverage(candidates, "GEOS candidate init years after latitude filter")
    print(f"Loaded {len(all_candidates):,} complete candidate rows.")
    print(f"Using {len(candidates):,} candidate rows after latitude filter {args.min_lat:.1f} to {args.max_lat:.1f}.")

    if args.thresholds is None:
        threshold_paths = infer_threshold_paths(candidate_paths)
    else:
        threshold_paths = expand_paths(args.thresholds)
    if threshold_paths:
        print("Reading threshold CSV files:")
        for path in threshold_paths:
            print(f"  - {path}")
    thresholds = read_thresholds(threshold_paths, args.ocean_only)

    observed_path = Path(args.observed_percentiles)
    observed_percentiles = read_observed_percentiles(
        observed_path,
        wind_var=args.observed_wind_var,
        basin_method=args.observed_basin_method,
    )
    thresholds = fill_provisional_thresholds(thresholds, all_candidates, observed_percentiles, observed_path)

    prefix = args.prefix or derive_prefix(candidate_paths)
    if months:
        prefix = f"{prefix}_{'-'.join(sorted(months))}"
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    density_ocean_checker = build_plot_ocean_checker(args)
    if density_ocean_checker is not None:
        print(f"Ocean-only density-bin mask enabled: source={density_ocean_checker.source}")

    print(f"Writing diagnostics to: {plot_dir}")
    plot_global_candidate_density_map(
        candidates,
        plot_dir / f"{prefix}_global_candidate_density_map.png",
        args.dpi,
        args.min_lat,
        args.max_lat,
        args.density_bin_deg,
        density_ocean_checker,
    )
    plot_global_candidate_map(candidates, thresholds, plot_dir / f"{prefix}_global_candidate_map.png", args.dpi, args.min_lat, args.max_lat)
    plot_global_basin_count_map(candidates, plot_dir / f"{prefix}_global_basin_count_map.png", args.dpi, args.min_lat, args.max_lat)
    plot_counts_by_basin_month(candidates, plot_dir / f"{prefix}_candidate_counts_by_basin_month.png", args.dpi)
    plot_ensemble_mean_counts_by_basin_month(candidates, plot_dir / f"{prefix}_candidate_member_mean_counts_by_basin_month.png", args.dpi)
    plot_vmax_histograms(candidates, thresholds, plot_dir / f"{prefix}_vmax_histograms_by_basin.png", args.dpi)
    plot_year_basin_heatmap(candidates, plot_dir / f"{prefix}_candidate_counts_by_init_year_basin.png", args.dpi)
    plot_threshold_bars(candidates, thresholds, plot_dir / f"{prefix}_thresholds_by_basin.png", args.dpi)
    write_summary_csv(candidates, thresholds, plot_dir / f"{prefix}_summary.csv")
    print_console_summary(candidates, thresholds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
