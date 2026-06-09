#!/usr/bin/env python3
"""Diagnose North Indian Ocean IBTrACS behavior without using GEOS data.

The purpose of this script is to make the observed-data side transparent before
calibrating GEOS. It keeps all selected IBTrACS NATURE codes by default, then
reports the tropical-cyclone subset separately. It also computes all-fix and
ocean-only summaries in the same run so coastal/landfall sensitivity is visible.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ocean_mask_utils import build_ocean_checker

try:
    import netCDF4
except ImportError:
    print("ERROR: netCDF4 package is required. Load the earth environment or install netCDF4.", file=sys.stderr)
    sys.exit(2)

from calculate_ibtracs_observed_percentiles import (  # noqa: E402
    BASINS,
    basin_from_boxes,
    basin_from_code,
    decode_chars,
    normalize_lon,
    read_storm_time_text,
)


DEFAULT_IBTRACS = "data/obs/ibtracs/IBTrACS.since1980.v04r01.nc"
DEFAULT_TABLE_DIR = "data/analysis/ibtracs_north_indian_diagnostics"
DEFAULT_PLOT_DIR = "plots/ibtracs_north_indian_diagnostics"
DEFAULT_PREFIX = "ibtracs_north_indian"
PRIMARY_COLOR = "#334f8d"
OCEAN_COLOR = "#1aa6b7"
LAND_COLOR = "#d45a3a"
GRID_COLOR = "#d1d5db"


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def parse_years(value: str) -> set[int]:
    years: set[int] = set()
    for item in parse_list(value):
        if "-" in item:
            left, right = item.split("-", 1)
            years.update(range(int(left), int(right) + 1))
        elif ":" in item:
            left, right = item.split(":", 1)
            years.update(range(int(left), int(right) + 1))
        else:
            years.add(int(item))
    return years


def parse_months(value: str) -> set[str]:
    months: set[str] = set()
    for item in parse_list(value):
        if "-" in item:
            left, right = item.split("-", 1)
            months.update(f"{month:02d}" for month in range(int(left), int(right) + 1))
        elif ":" in item:
            left, right = item.split(":", 1)
            months.update(f"{month:02d}" for month in range(int(left), int(right) + 1))
        else:
            months.add(f"{int(item):02d}")
    return months


def finite_value(value) -> float | None:
    if np.ma.is_masked(value):
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def valid_wind(value) -> float | None:
    out = finite_value(value)
    if out is None or out < 0.0:
        return None
    return out


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def choose_pressure_var(ds, preferred: str) -> str | None:
    if preferred and preferred in ds.variables:
        return preferred
    for name in ("wmo_pres", "usa_pres", "tokyo_pres", "cma_pres", "hko_pres"):
        if name in ds.variables:
            return name
    return None


def read_name(name_var, storm_index: int) -> str:
    if name_var is None:
        return ""
    return decode_chars(name_var[storm_index, :])


def is_selected_basin(lat: float, lon: float, code: str, basin_name: str, method: str) -> bool:
    if method == "boxes":
        return basin_from_boxes(lat, lon) == basin_name
    return basin_from_code(code) == basin_name


def read_fixes(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, int]]:
    ibtracs = Path(args.ibtracs)
    if not ibtracs.exists():
        raise FileNotFoundError(f"IBTrACS file does not exist: {ibtracs}")

    years = parse_years(args.years)
    months = parse_months(args.months)
    keep_natures = {item.upper() for item in parse_list(args.nature_filter)}
    tc_natures = {item.upper() for item in parse_list(args.tc_natures)}
    wind_vars = parse_list(args.wind_vars)
    if args.wind_var not in wind_vars:
        wind_vars.insert(0, args.wind_var)

    rows: list[dict[str, object]] = []
    stats: dict[str, int] = defaultdict(int)

    with netCDF4.Dataset(ibtracs, "r") as ds:
        required = ["time", "lat", "lon", "sid", "season", "basin", "nature", args.wind_var]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"Missing IBTrACS variable(s): {', '.join(missing)}")

        unavailable_winds = [name for name in wind_vars if name not in ds.variables]
        if unavailable_winds:
            raise ValueError(f"Missing requested wind variable(s): {', '.join(unavailable_winds)}")

        time_var = ds.variables["time"]
        time_values = time_var[:]
        time_mask = np.ma.getmaskarray(time_values)
        dates = netCDF4.num2date(time_values, time_var.units, calendar=getattr(time_var, "calendar", "standard"))
        lat_values = ds.variables["lat"][:]
        lon_values = ds.variables["lon"][:]
        basin_values = ds.variables["basin"]
        nature_values = ds.variables["nature"]
        sid_values = ds.variables["sid"]
        season_values = ds.variables["season"][:]
        name_values = ds.variables["name"] if "name" in ds.variables else None
        wind_values = {name: ds.variables[name][:] for name in wind_vars}
        pressure_var_name = choose_pressure_var(ds, args.pressure_var)
        pressure_values = ds.variables[pressure_var_name][:] if pressure_var_name else None
        dist2land_values = ds.variables["dist2land"][:] if "dist2land" in ds.variables else None

        ocean_checker = None
        if dist2land_values is None and args.ocean_mask_source != "none":
            ocean_checker, warning = build_ocean_checker(
                args.ocean_mask_source,
                mask_file=args.ocean_mask_file,
                threshold=args.ocean_threshold,
                require_mask=True,
            )
            print(f"IBTrACS ocean flag source: {ocean_checker.source}")
            if warning:
                print(f"WARNING: IBTrACS ocean mask fallback: {warning}")
        elif dist2land_values is not None:
            print("IBTrACS ocean flag source: dist2land")
        else:
            print("IBTrACS ocean flag source: none")

        nstorm, ntime = time_values.shape
        for storm_index in range(nstorm):
            sid = decode_chars(sid_values[storm_index, :]) or f"storm_{storm_index}"
            name = read_name(name_values, storm_index)
            season = int(safe_float(season_values[storm_index], 0.0))
            for time_index in range(ntime):
                if time_mask[storm_index, time_index]:
                    continue
                date_value = dates[storm_index, time_index]
                year = int(getattr(date_value, "year", 0))
                month = f"{int(getattr(date_value, 'month', 0)):02d}"
                day = int(getattr(date_value, "day", 0))
                hour = int(getattr(date_value, "hour", 0))
                if year not in years or month not in months:
                    continue
                if args.synoptic_only and hour not in (0, 6, 12, 18):
                    stats["skipped_non_synoptic"] += 1
                    continue

                nature = read_storm_time_text(nature_values, storm_index, time_index).upper()
                if keep_natures and nature not in keep_natures:
                    stats["skipped_nature"] += 1
                    continue

                lat = finite_value(lat_values[storm_index, time_index])
                lon = finite_value(lon_values[storm_index, time_index])
                primary_wind = valid_wind(wind_values[args.wind_var][storm_index, time_index])
                if lat is None or lon is None or primary_wind is None:
                    stats["skipped_bad_position_or_wind"] += 1
                    continue

                basin_code = decode_chars(basin_values[storm_index, time_index, :]).upper()
                if not is_selected_basin(lat, lon, basin_code, args.basin, args.basin_method):
                    continue

                if dist2land_values is not None:
                    dist2land = finite_value(dist2land_values[storm_index, time_index])
                    is_ocean = bool(dist2land is not None and dist2land > 0.0)
                    ocean_source = "dist2land"
                elif ocean_checker is not None:
                    dist2land = float("nan")
                    is_ocean = bool(ocean_checker.is_ocean(lat, lon))
                    ocean_source = ocean_checker.source
                else:
                    dist2land = float("nan")
                    is_ocean = True
                    ocean_source = "none"

                pressure = finite_value(pressure_values[storm_index, time_index]) if pressure_values is not None else None
                wind_row = {
                    f"{name}_kt": valid_wind(values[storm_index, time_index])
                    for name, values in wind_values.items()
                }
                is_tc_nature = nature in tc_natures if tc_natures else True
                is_ts = bool(is_tc_nature and primary_wind >= args.threshold_kt)

                row: dict[str, object] = {
                    "sid": sid,
                    "name": name,
                    "season": season,
                    "time": f"{year:04d}-{month}-{day:02d} {hour:02d}:00",
                    "year": year,
                    "month": month,
                    "hour": hour,
                    "nature": nature or "MISSING",
                    "basin_code": basin_code or "MISSING",
                    "lat": float(lat),
                    "lon": normalize_lon(float(lon)),
                    "wind_kt": float(primary_wind),
                    "pressure_hpa": float(pressure) if pressure is not None and pressure >= 0.0 else float("nan"),
                    "dist2land": float(dist2land) if dist2land is not None else float("nan"),
                    "is_ocean": int(is_ocean),
                    "ocean_source": ocean_source,
                    "is_tc_nature": int(is_tc_nature),
                    "is_ts": int(is_ts),
                    "ace": float(primary_wind**2 * 1.0e-4) if is_ts else 0.0,
                }
                row.update(wind_row)
                rows.append(row)

    stats["selected_fixes"] = len(rows)
    return rows, dict(stats)


def summarize_values(values: list[float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype="float64")
    if finite.size == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
    return {
        "mean": float(np.nanmean(finite)),
        "median": float(np.nanmedian(finite)),
        "p90": float(np.nanpercentile(finite, 90)),
        "max": float(np.nanmax(finite)),
    }


def subset_for_scope(rows: list[dict[str, object]], scope: str) -> list[dict[str, object]]:
    if scope == "all":
        return rows
    if scope == "ocean":
        return [row for row in rows if int(row["is_ocean"]) == 1]
    if scope == "land_or_coast":
        return [row for row in rows if int(row["is_ocean"]) == 0]
    raise ValueError(scope)


def build_yearly_rows(rows: list[dict[str, object]], years: set[int], months: set[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    month_labels = ["all"] + sorted(months)
    for scope in ("all", "ocean", "land_or_coast"):
        scope_rows = subset_for_scope(rows, scope)
        for year in sorted(years):
            for month in month_labels:
                group = [
                    row for row in scope_rows
                    if int(row["year"]) == year and (month == "all" or str(row["month"]) == month)
                ]
                tc_group = [row for row in group if int(row["is_tc_nature"]) == 1]
                ts_group = [row for row in group if int(row["is_ts"]) == 1]
                winds = [float(row["wind_kt"]) for row in group]
                wind_stats = summarize_values(winds)
                output.append(
                    {
                        "year": year,
                        "month": month,
                        "scope": scope,
                        "n_fixes": len(group),
                        "n_tc_nature_fixes": len(tc_group),
                        "n_ts_fixes": len(ts_group),
                        "n_storms": len({str(row["sid"]) for row in group}),
                        "n_tc_nature_storms": len({str(row["sid"]) for row in tc_group}),
                        "n_ts_storms": len({str(row["sid"]) for row in ts_group}),
                        "n_active_times": len({str(row["time"]) for row in ts_group}),
                        "ace": float(sum(float(row["ace"]) for row in ts_group)),
                        "mean_wind_kt": wind_stats["mean"],
                        "median_wind_kt": wind_stats["median"],
                        "p90_wind_kt": wind_stats["p90"],
                        "max_wind_kt": wind_stats["max"],
                        "nature_counts": ";".join(
                            f"{key}:{value}" for key, value in sorted(Counter(str(row["nature"]) for row in group).items())
                        ),
                    }
                )
    return output


def build_nature_rows(rows: list[dict[str, object]], years: set[int], months: set[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    month_labels = ["all"] + sorted(months)
    for scope in ("all", "ocean", "land_or_coast"):
        scope_rows = subset_for_scope(rows, scope)
        natures = sorted({str(row["nature"]) for row in scope_rows})
        for year in sorted(years):
            for month in month_labels:
                for nature in natures:
                    group = [
                        row for row in scope_rows
                        if int(row["year"]) == year
                        and (month == "all" or str(row["month"]) == month)
                        and str(row["nature"]) == nature
                    ]
                    if not group:
                        continue
                    ts_group = [row for row in group if int(row["is_ts"]) == 1]
                    output.append(
                        {
                            "year": year,
                            "month": month,
                            "scope": scope,
                            "nature": nature,
                            "n_fixes": len(group),
                            "n_storms": len({str(row["sid"]) for row in group}),
                            "n_ge_threshold": len(ts_group),
                            "ace": float(sum(float(row["ace"]) for row in ts_group)),
                            "max_wind_kt": max(float(row["wind_kt"]) for row in group),
                        }
                    )
    return output


def build_storm_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sid"])].append(row)

    output: list[dict[str, object]] = []
    for sid, group in sorted(grouped.items()):
        ocean_group = [row for row in group if int(row["is_ocean"]) == 1]
        land_group = [row for row in group if int(row["is_ocean"]) == 0]
        ts_all = [row for row in group if int(row["is_ts"]) == 1]
        ts_ocean = [row for row in ocean_group if int(row["is_ts"]) == 1]
        pressures = [float(row["pressure_hpa"]) for row in group if np.isfinite(float(row["pressure_hpa"]))]
        output.append(
            {
                "sid": sid,
                "name": next((str(row["name"]) for row in group if str(row["name"])), ""),
                "season": int(group[0]["season"]),
                "first_time": min(str(row["time"]) for row in group),
                "last_time": max(str(row["time"]) for row in group),
                "months": ",".join(sorted({str(row["month"]) for row in group})),
                "nature_values": ";".join(f"{key}:{value}" for key, value in sorted(Counter(str(row["nature"]) for row in group).items())),
                "basin_codes": ";".join(f"{key}:{value}" for key, value in sorted(Counter(str(row["basin_code"]) for row in group).items())),
                "n_fixes": len(group),
                "n_ocean_fixes": len(ocean_group),
                "n_land_or_coast_fixes": len(land_group),
                "n_ts_fixes_all": len(ts_all),
                "n_ts_fixes_ocean": len(ts_ocean),
                "ace_all": float(sum(float(row["ace"]) for row in ts_all)),
                "ace_ocean": float(sum(float(row["ace"]) for row in ts_ocean)),
                "max_wind_kt": max(float(row["wind_kt"]) for row in group),
                "min_pressure_hpa": min(pressures) if pressures else float("nan"),
                "lat_min": min(float(row["lat"]) for row in group),
                "lat_max": max(float(row["lat"]) for row in group),
                "lon_min": min(float(row["lon"]) for row in group),
                "lon_max": max(float(row["lon"]) for row in group),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  -> Wrote {path}")


def rows_by_key(rows: list[dict[str, object]], key_field: str, value: object) -> list[dict[str, object]]:
    return [row for row in rows if row.get(key_field) == value]


def numeric_series(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.asarray([safe_float(row.get(field)) for row in rows], dtype="float64")


def setup_axis(ax, ylabel: str) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel)


def plot_ocean_sensitivity(yearly_rows: list[dict[str, object]], path: Path, basin: str, dpi: int) -> None:
    rows_all = [row for row in yearly_rows if row["month"] == "all" and row["scope"] == "all"]
    rows_ocean = [row for row in yearly_rows if row["month"] == "all" and row["scope"] == "ocean"]
    if not rows_all:
        return
    years = np.asarray([int(row["year"]) for row in rows_all], dtype="int32")
    fields = [
        ("ace", "ACE (10^4 kt^2)", "TS ACE"),
        ("n_ts_fixes", "Fix count", "TS fixes >= threshold"),
        ("n_ts_storms", "Storm count", "TS storms"),
        ("n_tc_nature_fixes", "Fix count", "TC-nature fixes"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2), constrained_layout=True)
    for ax, (field, ylabel, title) in zip(axes.ravel(), fields):
        ax.plot(years, numeric_series(rows_all, field), color=PRIMARY_COLOR, linewidth=1.8, marker="o", markersize=3.0, label="all fixes")
        ax.plot(years, numeric_series(rows_ocean, field), color=OCEAN_COLOR, linewidth=1.8, marker="s", markersize=3.0, label="ocean only")
        setup_axis(ax, ylabel)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"IBTrACS {basin} Sep+Oct Ocean Sensitivity", fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_monthly_ace(yearly_rows: list[dict[str, object]], path: Path, basin: str, months: set[str], dpi: int) -> None:
    fig, axes = plt.subplots(len(months), 1, figsize=(12.5, max(3.1 * len(months), 4.0)), sharex=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for ax, month in zip(axes_array, sorted(months)):
        rows_all = [row for row in yearly_rows if row["month"] == month and row["scope"] == "all"]
        rows_ocean = [row for row in yearly_rows if row["month"] == month and row["scope"] == "ocean"]
        years = np.asarray([int(row["year"]) for row in rows_all], dtype="int32")
        ax.plot(years, numeric_series(rows_all, "ace"), color=PRIMARY_COLOR, linewidth=1.8, marker="o", markersize=3.0, label="all fixes")
        ax.plot(years, numeric_series(rows_ocean, "ace"), color=OCEAN_COLOR, linewidth=1.8, marker="s", markersize=3.0, label="ocean only")
        setup_axis(ax, "ACE")
        ax.set_title(f"Month {month}", fontsize=10.5, fontweight="bold")
    axes_array[0].legend(frameon=False, fontsize=9)
    axes_array[-1].set_xlabel("Year")
    fig.suptitle(f"IBTrACS {basin} Monthly TS ACE", fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_nature_counts(nature_rows: list[dict[str, object]], path: Path, basin: str, dpi: int) -> None:
    rows = [row for row in nature_rows if row["month"] == "all" and row["scope"] == "all"]
    if not rows:
        return
    years = sorted({int(row["year"]) for row in rows})
    nature_totals = Counter()
    for row in rows:
        nature_totals[str(row["nature"])] += int(row["n_fixes"])
    top_natures = [nature for nature, _ in nature_totals.most_common(7)]
    colors = ["#334f8d", "#1aa6b7", "#e8a11c", "#9b5de5", "#ef476f", "#64748b", "#57cc99"]
    bottoms = np.zeros(len(years), dtype="float64")
    fig, ax = plt.subplots(figsize=(13.0, 4.8), constrained_layout=True)
    for color, nature in zip(colors, top_natures):
        values = []
        for year in years:
            value = sum(int(row["n_fixes"]) for row in rows if int(row["year"]) == year and str(row["nature"]) == nature)
            values.append(value)
        ax.bar(years, values, bottom=bottoms, width=0.78, color=color, label=nature)
        bottoms += np.asarray(values, dtype="float64")
    setup_axis(ax, "Fix count")
    ax.set_xlabel("Year")
    ax.set_title(f"IBTrACS {basin} Sep+Oct Fix Counts by NATURE", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, ncol=min(len(top_natures), 5), fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_wind_distribution(rows: list[dict[str, object]], path: Path, basin: str, threshold_kt: float, dpi: int) -> None:
    ocean = np.asarray([float(row["wind_kt"]) for row in rows if int(row["is_ocean"]) == 1], dtype="float64")
    land = np.asarray([float(row["wind_kt"]) for row in rows if int(row["is_ocean"]) == 0], dtype="float64")
    tc = np.asarray([float(row["wind_kt"]) for row in rows if int(row["is_tc_nature"]) == 1], dtype="float64")
    other = np.asarray([float(row["wind_kt"]) for row in rows if int(row["is_tc_nature"]) == 0], dtype="float64")

    arrays = [array for array in (ocean, land, tc, other) if array.size]
    if not arrays:
        return
    max_wind = max(float(np.nanmax(array)) for array in arrays)
    bins = np.arange(0.0, max(60.0, np.ceil(max_wind / 5.0) * 5.0) + 5.0, 5.0)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), constrained_layout=True)
    if ocean.size:
        axes[0].hist(ocean, bins=bins, color=OCEAN_COLOR, alpha=0.75, label=f"ocean n={ocean.size:,}")
    if land.size:
        axes[0].hist(land, bins=bins, color=LAND_COLOR, alpha=0.6, label=f"land/coast n={land.size:,}")
    if tc.size:
        axes[1].hist(tc, bins=bins, color=PRIMARY_COLOR, alpha=0.75, label=f"TC nature n={tc.size:,}")
    if other.size:
        axes[1].hist(other, bins=bins, color="#9ca3af", alpha=0.6, label=f"other nature n={other.size:,}")
    for ax in axes:
        ax.axvline(threshold_kt, color="#1e222a", linestyle="--", linewidth=1.2, label=f"{threshold_kt:g} kt")
        setup_axis(ax, "Fix count")
        ax.set_xlabel("Wind (kt)")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_title("Ocean Sensitivity", fontsize=10.5, fontweight="bold")
    axes[1].set_title("NATURE Sensitivity", fontsize=10.5, fontweight="bold")
    fig.suptitle(f"IBTrACS {basin} Sep+Oct Wind Distributions", fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def print_summary(rows: list[dict[str, object]], yearly_rows: list[dict[str, object]], stats: dict[str, int], args: argparse.Namespace) -> None:
    print("\nIBTrACS North Indian diagnostic summary")
    print(f"basin={args.basin}, years={args.years}, months={args.months}, wind={args.wind_var}, threshold={args.threshold_kt:g} kt")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value:,}")

    nature_counts = Counter(str(row["nature"]) for row in rows)
    basin_code_counts = Counter(str(row["basin_code"]) for row in rows)
    print("nature counts:", ", ".join(f"{key}={value}" for key, value in sorted(nature_counts.items())))
    print("IBTrACS basin-code counts:", ", ".join(f"{key}={value}" for key, value in sorted(basin_code_counts.items())))

    print("\nSep+Oct totals")
    print(f"{'scope':14s} {'fixes':>8s} {'TCfix':>8s} {'TSfix':>8s} {'storms':>8s} {'ACE':>10s} {'maxV':>8s}")
    for scope in ("all", "ocean", "land_or_coast"):
        row = next(item for item in yearly_rows if item["year"] == min(parse_years(args.years)) and item["month"] == "all" and item["scope"] == scope)
        # Recompute over all years for a compact total row.
        scope_rows = subset_for_scope(rows, scope)
        tc_rows = [item for item in scope_rows if int(item["is_tc_nature"]) == 1]
        ts_rows = [item for item in scope_rows if int(item["is_ts"]) == 1]
        max_wind = max([float(item["wind_kt"]) for item in scope_rows], default=float("nan"))
        print(
            f"{scope:14s} "
            f"{len(scope_rows):8d} "
            f"{len(tc_rows):8d} "
            f"{len(ts_rows):8d} "
            f"{len({str(item['sid']) for item in ts_rows}):8d} "
            f"{sum(float(item['ace']) for item in ts_rows):10.2f} "
            f"{max_wind:8.1f}"
        )

    ocean_years = [row for row in yearly_rows if row["month"] == "all" and row["scope"] == "ocean"]
    zero_ace = [int(row["year"]) for row in ocean_years if float(row["ace"]) == 0.0]
    low_sample = [int(row["year"]) for row in ocean_years if int(row["n_ts_fixes"]) <= 2]
    if zero_ace:
        print("Ocean-only years with zero TS ACE:", ", ".join(str(year) for year in zero_ace))
    if low_sample:
        print("Ocean-only years with <=2 TS fixes:", ", ".join(str(year) for year in low_sample))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create IBTrACS-only diagnostics for the North Indian Ocean basin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS)
    parser.add_argument("--years", default="1991:2024")
    parser.add_argument("--months", default="09:10")
    parser.add_argument("--basin", choices=tuple(BASINS), default="North Indian")
    parser.add_argument("--basin-method", choices=("boxes", "ibtracs_code"), default="boxes")
    parser.add_argument("--wind-var", default="usa_wind")
    parser.add_argument("--wind-vars", default="usa_wind,wmo_wind")
    parser.add_argument("--pressure-var", default="wmo_pres")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument(
        "--tc-natures",
        default="TS",
        help="NATURE codes treated as tropical cyclones for TS/ACE summaries.",
    )
    parser.add_argument(
        "--nature-filter",
        default="",
        help="Optional NATURE codes to keep before diagnostics. Empty keeps all selected basin fixes.",
    )
    parser.add_argument("--all-hours", action="store_false", dest="synoptic_only", help="Include non-synoptic IBTrACS times.")
    parser.set_defaults(synoptic_only=True)
    parser.add_argument("--ocean-mask-source", choices=("auto", "sfc", "cartopy", "none"), default="none")
    parser.add_argument("--ocean-mask-file", default="")
    parser.add_argument("--ocean-threshold", type=float, default=0.5)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table_dir = Path(args.table_dir)
    plot_dir = Path(args.plot_dir)
    years = parse_years(args.years)
    months = parse_months(args.months)

    rows, stats = read_fixes(args)
    if not rows:
        print("ERROR: no IBTrACS fixes matched the selected basin/year/month filters.", file=sys.stderr)
        return 1

    yearly_rows = build_yearly_rows(rows, years, months)
    nature_rows = build_nature_rows(rows, years, months)
    storm_rows = build_storm_rows(rows)

    fix_fields = [
        "sid", "name", "season", "time", "year", "month", "hour", "nature", "basin_code",
        "lat", "lon", "wind_kt", "pressure_hpa", "dist2land", "is_ocean", "ocean_source",
        "is_tc_nature", "is_ts", "ace",
    ]
    for wind_var in parse_list(args.wind_vars):
        key = f"{wind_var}_kt"
        if key in rows[0]:
            fix_fields.append(key)

    write_csv(table_dir / f"{args.prefix}_fixes.csv", rows, fix_fields)
    write_csv(table_dir / f"{args.prefix}_yearly.csv", yearly_rows)
    write_csv(table_dir / f"{args.prefix}_nature_by_year.csv", nature_rows)
    write_csv(table_dir / f"{args.prefix}_storms.csv", storm_rows)

    plot_ocean_sensitivity(yearly_rows, plot_dir / f"{args.prefix}_ocean_sensitivity.png", args.basin, args.dpi)
    plot_monthly_ace(yearly_rows, plot_dir / f"{args.prefix}_monthly_ace.png", args.basin, months, args.dpi)
    plot_nature_counts(nature_rows, plot_dir / f"{args.prefix}_nature_counts.png", args.basin, args.dpi)
    plot_wind_distribution(rows, plot_dir / f"{args.prefix}_wind_distribution.png", args.basin, args.threshold_kt, args.dpi)
    print_summary(rows, yearly_rows, stats, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
