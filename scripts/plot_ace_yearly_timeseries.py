#!/usr/bin/env python3
"""Plot yearly ACE time series from cached TC-conditioned ACE NetCDF files."""

from __future__ import annotations

import argparse
import csv
import os
import re
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

try:
    import netCDF4
except ImportError:
    netCDF4 = None

from calculate_tc_conditioned_ace import BASINS, read_cache, thresholds_from_diagnostics
from ocean_mask_utils import add_ocean_only_args, build_ocean_checker


BASIN_ORDER = list(BASINS)
DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_PLOT_DIR = "plots/ace_yearly_timeseries"
DEFAULT_IBTRACS = "data/obs/ibtracs/IBTrACS.since1980.v04r01.nc"


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def parse_months(value: str) -> set[str]:
    months: set[str] = set()
    for item in parse_list(value):
        month = int(item)
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {item}")
        months.add(f"{month:02d}")
    return months


def parse_years(value: str) -> set[int]:
    years: set[int] = set()
    for item in re.split(r"[\s,]+", value.strip()):
        if not item:
            continue
        if ":" in item:
            start_text, end_text = item.split(":", 1)
            start_year = int(start_text)
            end_year = int(end_text)
            years.update(range(start_year, end_year + 1))
        else:
            years.add(int(item))
    return years


def cache_year_and_kind(path: Path) -> tuple[int | None, str]:
    lagged = re.match(r"tc_conditioned_ace_(\d{4})_lagged_ensmean\.nc4$", path.name)
    if lagged:
        return int(lagged.group(1)), "lagged"
    init = re.match(r"tc_conditioned_ace_(\d{8})_ensmean\.nc4$", path.name)
    if init:
        return int(init.group(1)[:4]), "init"
    return None, "other"


def decode_chars(values) -> str:
    array = np.ma.filled(np.asarray(values), b" ")
    chars: list[str] = []
    for item in array.reshape(-1):
        if isinstance(item, bytes):
            chars.append(item.decode("utf-8", errors="ignore"))
        elif isinstance(item, str):
            chars.append(item)
        else:
            value = int(item)
            if value > 0:
                chars.append(chr(value))
    return "".join(chars).strip()


def read_storm_time_text(variable, storm_index: int, time_index: int) -> str:
    if getattr(variable, "ndim", 0) <= 2:
        return decode_chars(variable[storm_index, time_index])
    return decode_chars(variable[storm_index, time_index, :])


def normalize_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def basin_from_boxes(lat: float, lon: float) -> str | None:
    lon = normalize_lon(lon)
    for basin_name, basin_def in BASINS.items():
        lat_min, lat_max = basin_def["lat_range"]
        if not (lat_min <= lat <= lat_max):
            continue
        if "lon_range" in basin_def:
            lon_min, lon_max = basin_def["lon_range"]
            if lon_min <= lon <= lon_max:
                return basin_name
        else:
            for lon_min, lon_max in basin_def["lon_ranges"]:
                if lon_min <= lon <= lon_max:
                    return basin_name
    return None


def basin_from_code(code: str) -> str | None:
    code = code.strip().upper()
    for basin_name, basin_def in BASINS.items():
        if code in basin_def.get("codes", ()):
            return basin_name
    code_to_name = {
        "NA": "North Atlantic",
        "EP": "Northeast Pacific",
        "CP": "Northeast Pacific",
        "WP": "Northwest Pacific",
        "NI": "North Indian",
        "SI": "South Indian",
        "SP": "South Pacific",
    }
    return code_to_name.get(code)


def finite_value(value) -> float | None:
    if np.ma.is_masked(value):
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def nature_is_selected(nature_value: str, allowed_natures: set[str]) -> bool:
    return not allowed_natures or nature_value.strip().upper() in allowed_natures


def discover_cache_files(cache_dir: Path, years: set[int], cache_kind: str) -> dict[int, list[Path]]:
    files_by_year: dict[int, dict[str, list[Path]]] = defaultdict(lambda: {"lagged": [], "init": [], "other": []})
    for path in sorted(cache_dir.glob("tc_conditioned_ace_*_ensmean.nc4")):
        year, kind = cache_year_and_kind(path)
        if year is None:
            continue
        if years and year not in years:
            continue
        files_by_year[year][kind].append(path)

    selected: dict[int, list[Path]] = {}
    for year, groups in sorted(files_by_year.items()):
        if cache_kind == "auto":
            selected[year] = groups["lagged"] if groups["lagged"] else groups["init"]
        elif cache_kind == "lagged":
            selected[year] = groups["lagged"]
        elif cache_kind == "init":
            selected[year] = groups["init"]
        else:
            selected[year] = groups["lagged"] + groups["init"] + groups["other"]
        selected[year] = sorted(selected[year])
    return {year: paths for year, paths in selected.items() if paths}


def discover_lagged_member_files(cache_dir: Path, year: int, init_dates: list[str]) -> list[Path]:
    paths: list[Path] = []
    for init_date_md in init_dates:
        for path in sorted(cache_dir.glob(f"tc_conditioned_ace_{year}{init_date_md}_ens*.nc4")):
            if path.name.endswith("_ensmean.nc4") or "lagged" in path.name:
                continue
            paths.append(path)
    return sorted(paths)


def read_cache_ace_values(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    _, _, _, _, _, _, diagnostics, _, _ = read_cache(path)
    thresholds = thresholds_from_diagnostics(diagnostics)
    values: dict[str, float] = {}
    total = 0.0
    for basin_name in BASIN_ORDER:
        curve = np.asarray(diagnostics[basin_name]["cumulative_ace"], dtype="float64")
        value = float(curve[-1]) if curve.size else float("nan")
        values[basin_name] = value
        if np.isfinite(value):
            total += value
    values["All Basins"] = total
    return values, thresholds


def setup_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "#ffffff"


def summarize_caches(
    files_by_year: dict[int, list[Path]],
    lagged_member_spread: bool = False,
    lagged_init_dates: list[str] | None = None,
) -> tuple[list[dict[str, object]], dict[int, list[Path]]]:
    rows: list[dict[str, object]] = []
    used_files: dict[int, list[Path]] = {}
    lagged_init_dates = lagged_init_dates or ["0824", "0829"]
    for year, paths in sorted(files_by_year.items()):
        basin_values: dict[str, list[float]] = defaultdict(list)
        basin_thresholds: dict[str, list[float]] = defaultdict(list)
        good_paths: list[Path] = []
        for path in paths:
            try:
                values, thresholds = read_cache_ace_values(path)
            except Exception as exc:
                print(f"WARNING: skipping unreadable cache {path}: {exc}")
                continue
            good_paths.append(path)
            for basin_name in BASIN_ORDER:
                basin_values[basin_name].append(values.get(basin_name, float("nan")))
                if basin_name in thresholds:
                    basin_thresholds[basin_name].append(float(thresholds[basin_name]))
            basin_values["All Basins"].append(values.get("All Basins", float("nan")))

        if not good_paths:
            continue
        used_files[year] = good_paths

        spread_values: dict[str, list[float]] = defaultdict(list)
        spread_paths: list[Path] = []
        spread_source = "selected_caches"
        selected_has_lagged = any(cache_year_and_kind(path)[1] == "lagged" for path in good_paths)
        if lagged_member_spread and selected_has_lagged:
            spread_paths = discover_lagged_member_files(good_paths[0].parent, year, lagged_init_dates)
            if spread_paths:
                spread_source = "lagged_members:" + ",".join(lagged_init_dates)
                for spread_path in spread_paths:
                    try:
                        spread_cache_values, _ = read_cache_ace_values(spread_path)
                    except Exception as exc:
                        print(f"WARNING: skipping unreadable member-spread cache {spread_path}: {exc}")
                        continue
                    for basin_name in BASIN_ORDER + ["All Basins"]:
                        spread_values[basin_name].append(spread_cache_values.get(basin_name, float("nan")))
                print(f"Using {len(spread_paths)} lagged member cache(s) for {year} GEOS spread.")
            else:
                print(
                    f"WARNING: no lagged member caches found for {year} init dates "
                    f"{','.join(lagged_init_dates)}; using selected cache spread."
                )

        if not spread_values:
            spread_values = basin_values

        for basin_name in BASIN_ORDER:
            values = np.asarray(basin_values[basin_name], dtype="float64")
            values = values[np.isfinite(values)]
            spread = np.asarray(spread_values[basin_name], dtype="float64")
            spread = spread[np.isfinite(spread)]
            thresholds = np.asarray(basin_thresholds[basin_name], dtype="float64")
            thresholds = thresholds[np.isfinite(thresholds)]
            rows.append(
                {
                    "year": year,
                    "basin_name": basin_name,
                    "mean_ace": float(np.nanmean(values)) if values.size else float("nan"),
                    "std_ace": float(np.nanstd(spread)) if spread.size > 1 else 0.0,
                    "n_caches": int(values.size),
                    "n_spread_members": int(spread.size),
                    "spread_source": spread_source,
                    "threshold_kt": float(np.nanmean(thresholds)) if thresholds.size else float("nan"),
                    "cache_files": ",".join(path.name for path in good_paths),
                    "spread_cache_files": ",".join(path.name for path in spread_paths),
                }
            )
        total_array = np.asarray(basin_values["All Basins"], dtype="float64")
        total_array = total_array[np.isfinite(total_array)]
        total_spread_array = np.asarray(spread_values["All Basins"], dtype="float64")
        total_spread_array = total_spread_array[np.isfinite(total_spread_array)]
        rows.append(
            {
                "year": year,
                "basin_name": "All Basins",
                "mean_ace": float(np.nanmean(total_array)) if total_array.size else float("nan"),
                "std_ace": float(np.nanstd(total_spread_array)) if total_spread_array.size > 1 else 0.0,
                "n_caches": int(total_array.size),
                "n_spread_members": int(total_spread_array.size),
                "spread_source": spread_source,
                "threshold_kt": float("nan"),
                "cache_files": ",".join(path.name for path in good_paths),
                "spread_cache_files": ",".join(path.name for path in spread_paths),
            }
        )
    return rows, used_files


def read_ibtracs_observed_ace(
    path: Path,
    years: set[int],
    months: set[str],
    wind_var: str,
    threshold_kt: float,
    nature_filter: str,
    basin_method: str,
    synoptic_only: bool,
    ocean_only: bool,
    ocean_mask_source: str,
    ocean_mask_file: str,
    ocean_threshold: float,
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], int]]:
    if netCDF4 is None:
        raise RuntimeError("netCDF4 is required to read IBTrACS")
    if not path.exists():
        raise FileNotFoundError(f"IBTrACS file does not exist: {path}")

    allowed_natures = {item.upper() for item in parse_list(nature_filter)}
    ace_by_year_basin: dict[tuple[int, str], float] = defaultdict(float)
    fix_count_by_year_basin: dict[tuple[int, str], int] = defaultdict(int)
    skipped_nature = 0
    skipped_land = 0
    skipped_bad = 0

    with netCDF4.Dataset(path, "r") as ds:
        required = ["time", "lat", "lon", wind_var]
        if allowed_natures:
            required.append("nature")
        if basin_method == "ibtracs_code":
            required.append("basin")
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"Missing IBTrACS variable(s): {', '.join(missing)}")

        time_var = ds.variables["time"]
        time_values = time_var[:]
        time_mask = np.ma.getmaskarray(time_values)
        dates = netCDF4.num2date(
            time_values,
            time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
        )
        lat_values = ds.variables["lat"][:]
        lon_values = ds.variables["lon"][:]
        wind_values = ds.variables[wind_var][:]
        basin_values = ds.variables["basin"] if "basin" in ds.variables else None
        nature_values = ds.variables["nature"] if allowed_natures else None
        dist2land_values = ds.variables["dist2land"][:] if ocean_only and "dist2land" in ds.variables else None
        ocean_checker = None
        if ocean_only and dist2land_values is None:
            ocean_checker, warning = build_ocean_checker(
                ocean_mask_source,
                mask_file=ocean_mask_file,
                threshold=ocean_threshold,
                require_mask=True,
            )
            print(f"Ocean-only IBTrACS ACE filter enabled: source={ocean_checker.source}")
            if warning:
                print(f"WARNING: IBTrACS ocean mask fallback: {warning}")
        elif ocean_only:
            print("Ocean-only IBTrACS ACE filter enabled: source=dist2land")

        nstorm, ntime = time_values.shape
        for storm_index in range(nstorm):
            for time_index in range(ntime):
                if time_mask[storm_index, time_index]:
                    continue
                date_value = dates[storm_index, time_index]
                year = int(getattr(date_value, "year", 0))
                month = f"{int(getattr(date_value, 'month', 0)):02d}"
                hour = int(getattr(date_value, "hour", 0))
                if years and year not in years:
                    continue
                if months and month not in months:
                    continue
                if synoptic_only and hour not in (0, 6, 12, 18):
                    continue
                if nature_values is not None:
                    nature = read_storm_time_text(nature_values, storm_index, time_index)
                    if not nature_is_selected(nature, allowed_natures):
                        skipped_nature += 1
                        continue

                lat = finite_value(lat_values[storm_index, time_index])
                lon = finite_value(lon_values[storm_index, time_index])
                wind = finite_value(wind_values[storm_index, time_index])
                if lat is None or lon is None or wind is None or wind < 0.0:
                    skipped_bad += 1
                    continue
                if ocean_only:
                    if dist2land_values is not None:
                        dist2land = finite_value(dist2land_values[storm_index, time_index])
                        is_ocean = dist2land is not None and dist2land > 0.0
                    else:
                        is_ocean = ocean_checker.is_ocean(lat, lon)
                    if not is_ocean:
                        skipped_land += 1
                        continue

                if basin_method == "boxes":
                    basin_name = basin_from_boxes(lat, lon)
                else:
                    if basin_values is None:
                        skipped_bad += 1
                        continue
                    basin_name = basin_from_code(decode_chars(basin_values[storm_index, time_index, :]))
                if basin_name is None:
                    skipped_bad += 1
                    continue

                if wind >= threshold_kt:
                    key = (year, basin_name)
                    ace_by_year_basin[key] += float(wind**2) * 1.0e-4
                    fix_count_by_year_basin[key] += 1

    if skipped_nature:
        print(f"Skipped {skipped_nature:,} IBTrACS fixes outside nature filter: {nature_filter or 'ALL'}")
    if skipped_land:
        print(f"Skipped {skipped_land:,} IBTrACS fixes over land.")
    if skipped_bad:
        print(f"Skipped {skipped_bad:,} incomplete/unassigned IBTrACS fixes.")
    return dict(ace_by_year_basin), dict(fix_count_by_year_basin)


def add_ibtracs_to_rows(
    rows: list[dict[str, object]],
    ibtracs_ace: dict[tuple[int, str], float],
    ibtracs_fix_counts: dict[tuple[int, str], int],
) -> None:
    years = sorted({int(row["year"]) for row in rows})
    for row in rows:
        year = int(row["year"])
        basin_name = str(row["basin_name"])
        if basin_name == "All Basins":
            obs = sum(float(ibtracs_ace.get((year, basin), 0.0)) for basin in BASIN_ORDER)
            fixes = sum(int(ibtracs_fix_counts.get((year, basin), 0)) for basin in BASIN_ORDER)
        else:
            obs = float(ibtracs_ace.get((year, basin_name), 0.0))
            fixes = int(ibtracs_fix_counts.get((year, basin_name), 0))
        row["ibtracs_ace"] = obs
        row["ibtracs_fix_count"] = fixes
        geos = float(row["mean_ace"])
        row["geos_to_ibtracs_ratio"] = geos / obs if obs > 0.0 and np.isfinite(geos) else float("nan")

    missing_obs_years = [
        year
        for year in years
        if not any(ibtracs_ace.get((year, basin), 0.0) > 0.0 for basin in BASIN_ORDER)
    ]
    if missing_obs_years:
        print(f"IBTrACS has zero observed ACE in selected months for years: {', '.join(str(year) for year in missing_obs_years)}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "year",
        "basin_name",
        "mean_ace",
        "std_ace",
        "n_caches",
        "n_spread_members",
        "spread_source",
        "threshold_kt",
        "ibtracs_ace",
        "ibtracs_fix_count",
        "geos_to_ibtracs_ratio",
        "cache_files",
        "spread_cache_files",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for fieldname in fieldnames:
                row.setdefault(fieldname, float("nan") if fieldname not in {"cache_files", "spread_cache_files", "spread_source", "basin_name"} else "")
            writer.writerow(row)
    print(f"  -> Wrote {path}")


def rows_for_basin(rows: list[dict[str, object]], basin_name: str) -> list[dict[str, object]]:
    return sorted([row for row in rows if row["basin_name"] == basin_name], key=lambda row: int(row["year"]))


def plot_basin_panels(rows: list[dict[str, object]], path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=dpi, sharex=True)
    for ax, basin_name in zip(axes.ravel(), BASIN_ORDER):
        basin_rows = rows_for_basin(rows, basin_name)
        years = np.asarray([int(row["year"]) for row in basin_rows], dtype="int32")
        mean_values = np.asarray([float(row["mean_ace"]) for row in basin_rows], dtype="float64")
        std_values = np.asarray([float(row["std_ace"]) for row in basin_rows], dtype="float64")
        obs_values = np.asarray([float(row.get("ibtracs_ace", np.nan)) for row in basin_rows], dtype="float64")
        color = BASINS[basin_name]["color"]
        ax.plot(years, mean_values, marker="o", markersize=3.5, linewidth=1.7, color=color, label="GEOS")
        if np.any(std_values > 0):
            lower = np.maximum(mean_values - std_values, 0.0)
            upper = mean_values + std_values
            ax.fill_between(
                years,
                lower,
                upper,
                color=color,
                alpha=0.16,
                linewidth=0,
                label="GEOS member spread" if basin_name == BASIN_ORDER[0] else None,
            )
        if np.any(np.isfinite(obs_values)):
            ax.plot(years, obs_values, marker="s", markersize=3.0, linewidth=1.5, color="#1e222a", linestyle="--", label="IBTrACS")
        ax.set_title(basin_name, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Yearly TC-Conditioned ACE by Basin", fontsize=13, fontweight="bold", y=0.99)
    fig.supxlabel("Initialization Year")
    fig.supylabel("ACE (10^4 kt^2)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_total(rows: list[dict[str, object]], path: Path, dpi: int) -> None:
    total_rows = rows_for_basin(rows, "All Basins")
    years = np.asarray([int(row["year"]) for row in total_rows], dtype="int32")
    mean_values = np.asarray([float(row["mean_ace"]) for row in total_rows], dtype="float64")
    std_values = np.asarray([float(row["std_ace"]) for row in total_rows], dtype="float64")
    obs_values = np.asarray([float(row.get("ibtracs_ace", np.nan)) for row in total_rows], dtype="float64")
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=dpi)
    ax.plot(years, mean_values, marker="o", markersize=4, linewidth=2.0, color="#344e86", label="GEOS")
    if np.any(std_values > 0):
        ax.fill_between(
            years,
            np.maximum(mean_values - std_values, 0.0),
            mean_values + std_values,
            color="#344e86",
            alpha=0.16,
            linewidth=0,
            label="GEOS member spread",
        )
    if np.any(np.isfinite(obs_values)):
        ax.plot(years, obs_values, marker="s", markersize=3.5, linewidth=1.7, color="#1e222a", linestyle="--", label="IBTrACS")
    ax.set_title("Yearly All-Basin TC-Conditioned ACE", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Initialization Year")
    ax.set_ylabel("ACE (10^4 kt^2)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--years", default="", help="Optional years, e.g. 1991:2024 or 1991,1992.")
    parser.add_argument("--cache-kind", choices=("auto", "lagged", "init", "all"), default="auto")
    parser.add_argument(
        "--lagged-init-dates",
        default="0824,0829",
        help="Month/day init dates to pool for lagged member spread, e.g. 0824,0829.",
    )
    parser.add_argument(
        "--lagged-member-spread",
        dest="lagged_member_spread",
        action="store_true",
        default=True,
        help="Use individual member caches from --lagged-init-dates for GEOS shading when lagged caches are selected.",
    )
    parser.add_argument(
        "--no-lagged-member-spread",
        dest="lagged_member_spread",
        action="store_false",
        help="Disable lagged member-cache spread shading.",
    )
    parser.add_argument("--prefix", default="ace_yearly_timeseries")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--ibtracs",
        default=DEFAULT_IBTRACS,
        help="Optional IBTrACS NetCDF for observed ACE overlay. Use an empty value to disable.",
    )
    parser.add_argument("--months", default="09:10", help="Observed IBTrACS months to compare, e.g. 09:10.")
    parser.add_argument("--wind-var", default="usa_wind", help="IBTrACS wind variable for observed ACE.")
    parser.add_argument("--threshold-kt", type=float, default=34.0, help="Observed ACE wind threshold in kt.")
    parser.add_argument(
        "--nature-filter",
        default="TS",
        help="Comma/colon/space separated IBTrACS nature values to keep. Default TS.",
    )
    parser.add_argument(
        "--all-natures",
        dest="nature_filter",
        action="store_const",
        const="",
        help="Disable IBTrACS nature filtering.",
    )
    parser.add_argument(
        "--basin-method",
        choices=("boxes", "ibtracs_code"),
        default="boxes",
        help="How to assign IBTrACS fixes to basins.",
    )
    parser.add_argument(
        "--all-hours",
        dest="synoptic_only",
        action="store_false",
        help="Use all IBTrACS fix hours instead of 00/06/12/18 UTC only.",
    )
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

    print("Using ACE cache files by year:")
    for year, paths in files_by_year.items():
        print(f"  {year}: {', '.join(path.name for path in paths)}")

    rows, used_files = summarize_caches(
        files_by_year,
        lagged_member_spread=args.lagged_member_spread,
        lagged_init_dates=parse_list(args.lagged_init_dates),
    )
    used_years = sorted(used_files)
    if not rows or not used_years:
        print(f"ERROR: no usable ACE caches found in {cache_dir}", file=sys.stderr)
        return 1
    if years:
        missing = sorted(years - set(used_years))
        if missing:
            print(f"Missing requested years with usable caches: {', '.join(str(year) for year in missing)}")
    print(f"Used years: {', '.join(str(year) for year in used_years)}")

    ibtracs_path = Path(args.ibtracs) if args.ibtracs else None
    if ibtracs_path is not None:
        if ibtracs_path.exists():
            months = parse_months(args.months)
            print(
                "Reading IBTrACS observed ACE: "
                f"path={ibtracs_path}, years={used_years[0]}:{used_years[-1]}, "
                f"months={','.join(sorted(months)) or 'ALL'}, wind={args.wind_var}, "
                f"threshold={args.threshold_kt:g} kt, nature={args.nature_filter or 'ALL'}"
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
            add_ibtracs_to_rows(rows, ibtracs_ace, ibtracs_fix_counts)
        else:
            print(f"WARNING: IBTrACS file not found, plotting GEOS only: {ibtracs_path}")

    write_csv(plot_dir / f"{args.prefix}.csv", rows)
    plot_basin_panels(rows, plot_dir / f"{args.prefix}_by_basin.png", args.dpi)
    plot_total(rows, plot_dir / f"{args.prefix}_all_basins.png", args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
