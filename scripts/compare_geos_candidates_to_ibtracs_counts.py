#!/usr/bin/env python3
"""Compare cached GEOS TC-candidate counts with IBTrACS observed counts.

GEOS candidate CSV rows are one structurally selected center per basin/time, not
tracked storms. The closest observed count comparison is therefore the number of
IBTrACS basin/times with at least one tropical-storm fix. Raw IBTrACS 6-hour fix
counts and unique storms are still reported as context, but the default plots
use active basin-times.

By default, GEOS basin thresholds are recomputed from the candidate CSVs being
compared. That keeps the quantile-matching sanity check apples-to-apples when
multiple cached calibration files exist in the same directory.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import netCDF4
except ImportError:
    print("ERROR: netCDF4 package is required. Load the earth environment or install netCDF4.", file=sys.stderr)
    sys.exit(2)

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "geos_s2s_tc_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CANDIDATES = "data/calibration/*_candidates.csv"
DEFAULT_IBTRACS = "data/obs/ibtracs/IBTrACS.since1980.v04r01.nc"
DEFAULT_OUTPUT_DIR = "data/calibration/count_comparison"
DEFAULT_PLOT_DIR = "plots/geos_ibtracs_count_comparison"

BASINS = {
    "North Atlantic": {
        "codes": ("NA",),
        "lat_range": (0.0, 45.0),
        "lon_range": (-100.0, -10.0),
        "color": "#e55934",
    },
    "Northeast Pacific": {
        "codes": ("EP", "CP"),
        "lat_range": (0.0, 40.0),
        "lon_range": (-180.0, -100.0),
        "color": "#f3a712",
    },
    "Northwest Pacific": {
        "codes": ("WP",),
        "lat_range": (0.0, 45.0),
        "lon_range": (100.0, 180.0),
        "color": "#2ec4b6",
    },
    "North Indian": {
        "codes": ("NI",),
        "lat_range": (0.0, 40.0),
        "lon_range": (40.0, 100.0),
        "color": "#9b5de5",
    },
    "South Indian": {
        "codes": ("SI",),
        "lat_range": (-40.0, 0.0),
        "lon_range": (20.0, 135.0),
        "color": "#00bbf9",
    },
    "South Pacific": {
        "codes": ("SP",),
        "lat_range": (-40.0, 0.0),
        "lon_ranges": [(135.0, 180.0), (-180.0, -120.0)],
        "color": "#ff007f",
    },
}

BASIN_ORDER = list(BASINS)
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


def format_months_label(months_text: str) -> str:
    months = sorted(parse_months(months_text))
    if not months:
        return "all months"
    return "+".join(MONTH_LABELS.get(month, month) for month in months)


@dataclass
class GeosCandidate:
    init_date: str
    ens: str
    year: int
    month: str
    basin_name: str
    center_lat: float
    vmax_kt: float


@dataclass
class IbtracsFix:
    sid: str
    year: int
    month: str
    time_key: str
    basin_name: str
    lat: float
    wind_kt: float


def configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except (AttributeError, TypeError):
            pass


def observed_count_label(metric: str) -> str:
    if metric == "active_time":
        return "IBTrACS TS active basin-times/year"
    return "IBTrACS TS fixes/year"


def observed_count_title(metric: str) -> str:
    if metric == "active_time":
        return "IBTrACS Tropical-Storm Active Basin-Times"
    return "IBTrACS Tropical-Storm Fix Counts"


def ibtracs_count_for_metric(fixes: list[IbtracsFix], metric: str) -> int:
    if metric == "active_time":
        return len({fix.time_key for fix in fixes})
    return len(fixes)


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.replace(",", " ").replace(":", " ").split() if item]


def parse_months(value: str) -> set[str]:
    months: set[str] = set()
    for item in parse_list(value):
        month = int(item)
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {item}")
        months.add(f"{month:02d}")
    return months


def expand_paths(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).exists():
            matches = [pattern]
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
        if not candidate_path.name.endswith("_candidates.csv"):
            continue
        threshold_path = candidate_path.with_name(candidate_path.name.replace("_candidates.csv", ".csv"))
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


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


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
        if code in basin_def["codes"]:
            return basin_name
    return None


def finite_value(value) -> float | None:
    if np.ma.is_masked(value):
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def valid_year_month(year: int, month: str, start_year: int, end_year: int, months: set[str]) -> bool:
    return start_year <= year <= end_year and (not months or month in months)


def read_geos_candidates(paths: list[Path], args: argparse.Namespace) -> tuple[list[GeosCandidate], list[GeosCandidate]]:
    all_candidates: list[GeosCandidate] = []
    plotted_candidates: list[GeosCandidate] = []
    skipped = 0
    months = parse_months(args.months)

    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                basin_name = row.get("basin_name", "")
                if basin_name not in BASINS:
                    skipped += 1
                    continue
                valid_time = parse_datetime(row.get("valid_time"))
                if valid_time is not None:
                    year = valid_time.year
                    month = valid_time.strftime("%m")
                else:
                    init_date = row.get("init_date", "")
                    year = int(init_date[:4]) if len(init_date) >= 4 and init_date[:4].isdigit() else 0
                    month = row.get("forecast_month", "")
                if not valid_year_month(year, month, args.start_year, args.end_year, months):
                    continue

                center_lat = parse_float(row.get("center_lat"))
                vmax_kt = parse_float(row.get("vmax_kt"))
                if not (np.isfinite(center_lat) and np.isfinite(vmax_kt)):
                    skipped += 1
                    continue

                candidate = GeosCandidate(
                    init_date=row.get("init_date", ""),
                    ens=row.get("ens", ""),
                    year=year,
                    month=month,
                    basin_name=basin_name,
                    center_lat=center_lat,
                    vmax_kt=vmax_kt,
                )
                all_candidates.append(candidate)
                if args.min_lat <= center_lat <= args.max_lat:
                    plotted_candidates.append(candidate)

    if skipped:
        print(f"Skipped {skipped} incomplete/unrecognized GEOS candidate rows.")
    return all_candidates, plotted_candidates


def read_thresholds(paths: list[Path], observed_wind_var: str, basin_method: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                basin_name = row.get("basin_name", "")
                if basin_name not in BASINS:
                    continue
                if row.get("observed_wind_var") and row.get("observed_wind_var") != observed_wind_var:
                    continue
                if row.get("observed_basin_method") and row.get("observed_basin_method") != basin_method:
                    continue
                threshold = parse_float(row.get("geos_threshold_kt"))
                if np.isfinite(threshold):
                    thresholds[basin_name] = threshold
    return thresholds


def read_ibtracs(args: argparse.Namespace) -> tuple[list[IbtracsFix], dict[str, list[float]]]:
    months = parse_months(args.months)
    fixes: list[IbtracsFix] = []
    wind_samples: dict[str, list[float]] = defaultdict(list)
    skipped = 0

    with netCDF4.Dataset(args.ibtracs, "r") as ds:
        required = ["time", "lat", "lon", args.wind_var]
        if args.basin_method == "ibtracs_code":
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
        wind_values = ds.variables[args.wind_var][:]
        basin_values = ds.variables["basin"] if "basin" in ds.variables else None
        sid_values = ds.variables["sid"] if "sid" in ds.variables else None

        nstorm, ntime = time_values.shape
        for storm_index in range(nstorm):
            sid = decode_chars(sid_values[storm_index, :]) if sid_values is not None else ""
            sid = sid or f"storm_{storm_index}"
            for time_index in range(ntime):
                if time_mask[storm_index, time_index]:
                    continue
                date_value = dates[storm_index, time_index]
                year = int(getattr(date_value, "year", 0))
                month = f"{int(getattr(date_value, 'month', 0)):02d}"
                day = int(getattr(date_value, "day", 0))
                hour = int(getattr(date_value, "hour", 0))
                if args.synoptic_only and hour not in (0, 6, 12, 18):
                    continue
                if not valid_year_month(year, month, args.start_year, args.end_year, months):
                    continue

                lat = finite_value(lat_values[storm_index, time_index])
                lon = finite_value(lon_values[storm_index, time_index])
                wind = finite_value(wind_values[storm_index, time_index])
                if lat is None or lon is None or wind is None or wind < 0.0:
                    skipped += 1
                    continue

                if args.basin_method == "boxes":
                    basin_name = basin_from_boxes(lat, lon)
                else:
                    if basin_values is None:
                        skipped += 1
                        continue
                    basin_name = basin_from_code(decode_chars(basin_values[storm_index, time_index, :]))
                if basin_name is None:
                    skipped += 1
                    continue

                wind_samples[basin_name].append(float(wind))
                if not (args.min_lat <= lat <= args.max_lat):
                    continue

                if wind >= args.threshold_kt:
                    fixes.append(
                        IbtracsFix(
                            sid=sid,
                            year=year,
                            month=month,
                            time_key=f"{year:04d}-{month}-{day:02d} {hour:02d}:00",
                            basin_name=basin_name,
                            lat=float(lat),
                            wind_kt=float(wind),
                        )
                    )

    if skipped:
        print(f"Skipped {skipped} incomplete/unassigned IBTrACS samples.")
    return fixes, wind_samples


def quantile_matched_geos_thresholds(
    geos_candidates: list[GeosCandidate],
    wind_samples: dict[str, list[float]],
    observed_threshold_kt: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    values_by_basin: dict[str, list[float]] = defaultdict(list)
    for candidate in geos_candidates:
        values_by_basin[candidate.basin_name].append(candidate.vmax_kt)

    for basin_name in BASIN_ORDER:
        obs_values = np.asarray(wind_samples.get(basin_name, []), dtype="float64")
        geos_values = np.asarray(values_by_basin.get(basin_name, []), dtype="float64")
        obs_values = obs_values[np.isfinite(obs_values)]
        geos_values = geos_values[np.isfinite(geos_values)]
        if obs_values.size == 0 or geos_values.size == 0:
            continue
        observed_percentile = 100.0 * float(np.sum(obs_values <= observed_threshold_kt)) / float(obs_values.size)
        thresholds[basin_name] = float(np.nanpercentile(geos_values, min(100.0, max(0.0, observed_percentile))))
    return thresholds


def fill_missing_geos_thresholds(
    thresholds: dict[str, float],
    fallback_thresholds: dict[str, float],
) -> dict[str, float]:
    out = dict(thresholds)
    for basin_name, threshold in fallback_thresholds.items():
        out.setdefault(basin_name, threshold)
    return out


def active_members_by_year(candidates: list[GeosCandidate]) -> dict[int, set[tuple[str, str]]]:
    members: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for candidate in candidates:
        if candidate.init_date and candidate.ens:
            members[candidate.year].add((candidate.init_date, candidate.ens))
    return members


def safe_member_mean(count: int, year: int, members_by_year: dict[int, set[tuple[str, str]]], expected_members: int) -> float:
    denominator = expected_members if expected_members > 0 else len(members_by_year.get(year, set()))
    if denominator <= 0:
        return float("nan")
    return float(count) / float(denominator)


def year_has_geos_coverage(row: dict[str, object]) -> bool:
    return int(row["expected_members_per_year"]) > 0 or int(row["active_init_member_pairs"]) > 0


def build_yearly_rows(
    geos_candidates: list[GeosCandidate],
    ibtracs_fixes: list[IbtracsFix],
    geos_thresholds: dict[str, float],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    years = list(range(args.start_year, args.end_year + 1))
    members_by_year = active_members_by_year(geos_candidates)
    rows: list[dict[str, object]] = []

    for year in years:
        for basin_name in BASIN_ORDER:
            obs_fixes = [fix for fix in ibtracs_fixes if fix.year == year and fix.basin_name == basin_name]
            obs_storms = {(fix.sid, fix.basin_name) for fix in obs_fixes}
            obs_active_times = {fix.time_key for fix in obs_fixes}
            obs_comparison_count = ibtracs_count_for_metric(obs_fixes, args.observed_count_metric)
            geos_basin = [candidate for candidate in geos_candidates if candidate.year == year and candidate.basin_name == basin_name]
            threshold = geos_thresholds.get(basin_name, float("nan"))
            if np.isfinite(threshold):
                geos_threshold_count = sum(1 for candidate in geos_basin if candidate.vmax_kt >= threshold)
            else:
                geos_threshold_count = 0
            geos_structural_count = len(geos_basin)
            active_members = len(members_by_year.get(year, set()))
            rows.append(
                {
                    "year": year,
                    "basin_name": basin_name,
                    "ibtracs_count_metric": args.observed_count_metric,
                    "ibtracs_count_for_comparison": obs_comparison_count,
                    "ibtracs_fix_count": len(obs_fixes),
                    "ibtracs_active_time_count": len(obs_active_times),
                    "ibtracs_storm_count": len(obs_storms),
                    "geos_structural_candidate_count": geos_structural_count,
                    "geos_ts_equiv_candidate_count": geos_threshold_count,
                    "active_init_member_pairs": active_members,
                    "expected_members_per_year": args.expected_members_per_year,
                    "geos_structural_member_mean_count": safe_member_mean(
                        geos_structural_count,
                        year,
                        members_by_year,
                        args.expected_members_per_year,
                    ),
                    "geos_ts_equiv_member_mean_count": safe_member_mean(
                        geos_threshold_count,
                        year,
                        members_by_year,
                        args.expected_members_per_year,
                    ),
                    "geos_threshold_kt": threshold,
                }
            )
    return rows


def build_monthly_rows(
    geos_candidates: list[GeosCandidate],
    ibtracs_fixes: list[IbtracsFix],
    geos_thresholds: dict[str, float],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    years = list(range(args.start_year, args.end_year + 1))
    months = sorted(parse_months(args.months))
    members_by_year = active_members_by_year(geos_candidates)
    rows: list[dict[str, object]] = []

    for month in months:
        for basin_name in BASIN_ORDER:
            obs_all_year_counts: list[float] = []
            obs_all_fix_year_counts: list[float] = []
            obs_all_active_time_year_counts: list[float] = []
            obs_all_storm_year_counts: list[float] = []
            obs_comparison_year_counts: list[float] = []
            obs_comparison_fix_year_counts: list[float] = []
            obs_comparison_active_time_year_counts: list[float] = []
            obs_comparison_storm_year_counts: list[float] = []
            geos_struct_year_means: list[float] = []
            geos_ts_year_means: list[float] = []
            comparison_years: list[int] = []
            for year in years:
                obs_fixes = [
                    fix
                    for fix in ibtracs_fixes
                    if fix.year == year and fix.month == month and fix.basin_name == basin_name
                ]
                obs_count = float(ibtracs_count_for_metric(obs_fixes, args.observed_count_metric))
                obs_fix_count = float(len(obs_fixes))
                obs_active_time_count = float(len({fix.time_key for fix in obs_fixes}))
                obs_storm_count = float(len({fix.sid for fix in obs_fixes}))
                obs_all_year_counts.append(obs_count)
                obs_all_fix_year_counts.append(obs_fix_count)
                obs_all_active_time_year_counts.append(obs_active_time_count)
                obs_all_storm_year_counts.append(obs_storm_count)
                geos_basin = [
                    candidate
                    for candidate in geos_candidates
                    if candidate.year == year and candidate.month == month and candidate.basin_name == basin_name
                ]
                threshold = geos_thresholds.get(basin_name, float("nan"))
                geos_ts_count = sum(1 for candidate in geos_basin if np.isfinite(threshold) and candidate.vmax_kt >= threshold)
                geos_struct_mean = safe_member_mean(len(geos_basin), year, members_by_year, args.expected_members_per_year)
                geos_ts_mean = safe_member_mean(geos_ts_count, year, members_by_year, args.expected_members_per_year)
                if np.isfinite(geos_struct_mean):
                    comparison_years.append(year)
                    obs_comparison_year_counts.append(obs_count)
                    obs_comparison_fix_year_counts.append(obs_fix_count)
                    obs_comparison_active_time_year_counts.append(obs_active_time_count)
                    obs_comparison_storm_year_counts.append(obs_storm_count)
                    geos_struct_year_means.append(geos_struct_mean)
                    geos_ts_year_means.append(geos_ts_mean)

            rows.append(
                {
                    "month": month,
                    "basin_name": basin_name,
                    "ibtracs_count_metric": args.observed_count_metric,
                    "comparison_year_count": len(comparison_years),
                    "comparison_years": ",".join(str(year) for year in comparison_years),
                    "ibtracs_count_year_mean": float(np.nanmean(obs_comparison_year_counts)) if obs_comparison_year_counts else float("nan"),
                    "ibtracs_fix_year_mean": float(np.nanmean(obs_comparison_fix_year_counts)) if obs_comparison_fix_year_counts else float("nan"),
                    "ibtracs_active_time_year_mean": float(np.nanmean(obs_comparison_active_time_year_counts))
                    if obs_comparison_active_time_year_counts
                    else float("nan"),
                    "ibtracs_storm_year_mean": float(np.nanmean(obs_comparison_storm_year_counts)) if obs_comparison_storm_year_counts else float("nan"),
                    "ibtracs_count_year_mean_all_years": float(np.nanmean(obs_all_year_counts)),
                    "ibtracs_fix_year_mean_all_years": float(np.nanmean(obs_all_fix_year_counts)),
                    "ibtracs_active_time_year_mean_all_years": float(np.nanmean(obs_all_active_time_year_counts)),
                    "ibtracs_storm_year_mean_all_years": float(np.nanmean(obs_all_storm_year_counts)),
                    "geos_structural_member_year_mean": float(np.nanmean(geos_struct_year_means)) if geos_struct_year_means else float("nan"),
                    "geos_ts_equiv_member_year_mean": float(np.nanmean(geos_ts_year_means)) if geos_ts_year_means else float("nan"),
                }
            )
    return rows


def build_summary_rows(yearly_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for basin_name in BASIN_ORDER:
        basin_rows = [row for row in yearly_rows if row["basin_name"] == basin_name]
        comparison_rows = [row for row in basin_rows if year_has_geos_coverage(row)]
        metric = str(basin_rows[0].get("ibtracs_count_metric", "fixes")) if basin_rows else "fixes"
        obs_count_all_values = np.asarray([row["ibtracs_count_for_comparison"] for row in basin_rows], dtype="float64")
        obs_fix_all_values = np.asarray([row["ibtracs_fix_count"] for row in basin_rows], dtype="float64")
        obs_active_time_all_values = np.asarray([row["ibtracs_active_time_count"] for row in basin_rows], dtype="float64")
        obs_storm_all_values = np.asarray([row["ibtracs_storm_count"] for row in basin_rows], dtype="float64")
        obs_count_values = np.asarray([row["ibtracs_count_for_comparison"] for row in comparison_rows], dtype="float64")
        obs_fix_values = np.asarray([row["ibtracs_fix_count"] for row in comparison_rows], dtype="float64")
        obs_active_time_values = np.asarray([row["ibtracs_active_time_count"] for row in comparison_rows], dtype="float64")
        obs_storm_values = np.asarray([row["ibtracs_storm_count"] for row in comparison_rows], dtype="float64")
        geos_struct_values = np.asarray([row["geos_structural_member_mean_count"] for row in comparison_rows], dtype="float64")
        geos_ts_values = np.asarray([row["geos_ts_equiv_member_mean_count"] for row in comparison_rows], dtype="float64")
        geos_thresholds = [float(row["geos_threshold_kt"]) for row in basin_rows if np.isfinite(float(row["geos_threshold_kt"]))]
        obs_count_mean = float(np.nanmean(obs_count_values)) if obs_count_values.size else float("nan")
        geos_ts_mean = float(np.nanmean(geos_ts_values)) if geos_ts_values.size else float("nan")
        rows.append(
            {
                "basin_name": basin_name,
                "ibtracs_count_metric": metric,
                "comparison_year_count": len(comparison_rows),
                "comparison_years": ",".join(str(row["year"]) for row in comparison_rows),
                "ibtracs_count_total": int(np.nansum(obs_count_values)) if obs_count_values.size else 0,
                "ibtracs_fix_total": int(np.nansum(obs_fix_values)) if obs_fix_values.size else 0,
                "ibtracs_active_time_total": int(np.nansum(obs_active_time_values)) if obs_active_time_values.size else 0,
                "ibtracs_storm_total": int(np.nansum(obs_storm_values)) if obs_storm_values.size else 0,
                "ibtracs_count_total_all_years": int(np.nansum(obs_count_all_values)),
                "ibtracs_fix_total_all_years": int(np.nansum(obs_fix_all_values)),
                "ibtracs_active_time_total_all_years": int(np.nansum(obs_active_time_all_values)),
                "ibtracs_storm_total_all_years": int(np.nansum(obs_storm_all_values)),
                "ibtracs_count_year_mean": obs_count_mean,
                "ibtracs_fix_year_mean": float(np.nanmean(obs_fix_values)) if obs_fix_values.size else float("nan"),
                "ibtracs_active_time_year_mean": float(np.nanmean(obs_active_time_values)) if obs_active_time_values.size else float("nan"),
                "ibtracs_storm_year_mean": float(np.nanmean(obs_storm_values)) if obs_storm_values.size else float("nan"),
                "ibtracs_count_year_mean_all_years": float(np.nanmean(obs_count_all_values)),
                "ibtracs_fix_year_mean_all_years": float(np.nanmean(obs_fix_all_values)),
                "ibtracs_active_time_year_mean_all_years": float(np.nanmean(obs_active_time_all_values)),
                "ibtracs_storm_year_mean_all_years": float(np.nanmean(obs_storm_all_values)),
                "geos_structural_member_year_mean": float(np.nanmean(geos_struct_values)) if geos_struct_values.size else float("nan"),
                "geos_ts_equiv_member_year_mean": geos_ts_mean,
                "geos_to_ibtracs_count_ratio": geos_ts_mean / obs_count_mean if obs_count_mean > 0 else float("nan"),
                "geos_to_ibtracs_fix_ratio": geos_ts_mean / float(np.nanmean(obs_fix_values))
                if obs_fix_values.size and float(np.nanmean(obs_fix_values)) > 0
                else float("nan"),
                "geos_threshold_kt": geos_thresholds[0] if geos_thresholds else float("nan"),
            }
        )
    return rows


def build_percentile_rows(
    geos_candidates: list[GeosCandidate],
    wind_samples: dict[str, list[float]],
    geos_thresholds: dict[str, float],
    observed_threshold_kt: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values_by_basin: dict[str, list[float]] = defaultdict(list)
    for candidate in geos_candidates:
        values_by_basin[candidate.basin_name].append(candidate.vmax_kt)

    for basin_name in BASIN_ORDER:
        obs_values = np.asarray(wind_samples.get(basin_name, []), dtype="float64")
        geos_values = np.asarray(values_by_basin.get(basin_name, []), dtype="float64")
        obs_values = obs_values[np.isfinite(obs_values)]
        geos_values = geos_values[np.isfinite(geos_values)]
        threshold = geos_thresholds.get(basin_name, float("nan"))

        if obs_values.size:
            obs_le_count = int(np.sum(obs_values <= observed_threshold_kt))
            obs_ge_count = int(np.sum(obs_values >= observed_threshold_kt))
            obs_gt_count = int(np.sum(obs_values > observed_threshold_kt))
            obs_percentile_le = obs_le_count / float(obs_values.size)
            obs_exceedance = obs_ge_count / float(obs_values.size)
            obs_strict_exceedance = obs_gt_count / float(obs_values.size)
        else:
            obs_le_count = 0
            obs_ge_count = 0
            obs_gt_count = 0
            obs_percentile_le = float("nan")
            obs_exceedance = float("nan")
            obs_strict_exceedance = float("nan")

        if geos_values.size and np.isfinite(threshold):
            geos_le_count = int(np.sum(geos_values <= threshold))
            geos_ge_count = int(np.sum(geos_values >= threshold))
            geos_gt_count = int(np.sum(geos_values > threshold))
            geos_eq_count = int(np.sum(np.isclose(geos_values, threshold, rtol=1.0e-10, atol=1.0e-10)))
            geos_percentile_le = geos_le_count / float(geos_values.size)
            geos_exceedance = geos_ge_count / float(geos_values.size)
            geos_strict_exceedance = geos_gt_count / float(geos_values.size)
            geos_eq_fraction = geos_eq_count / float(geos_values.size)
            geos_upper_midrank = geos_strict_exceedance + 0.5 * geos_eq_fraction
        else:
            geos_le_count = 0
            geos_ge_count = 0
            geos_gt_count = 0
            geos_eq_count = 0
            geos_percentile_le = float("nan")
            geos_exceedance = float("nan")
            geos_strict_exceedance = float("nan")
            geos_eq_fraction = float("nan")
            geos_upper_midrank = float("nan")

        rows.append(
            {
                "basin_name": basin_name,
                "ibtracs_sample_count": int(obs_values.size),
                "ibtracs_count_le_34kt": obs_le_count,
                "ibtracs_count_ge_34kt": obs_ge_count,
                "ibtracs_count_gt_34kt": obs_gt_count,
                "ibtracs_fraction_le_34kt": obs_percentile_le,
                "ibtracs_fraction_ge_34kt": obs_exceedance,
                "ibtracs_fraction_gt_34kt": obs_strict_exceedance,
                "geos_candidate_count": int(geos_values.size),
                "geos_threshold_kt": threshold,
                "geos_count_le_threshold": geos_le_count,
                "geos_count_ge_threshold": geos_ge_count,
                "geos_count_gt_threshold": geos_gt_count,
                "geos_count_eq_threshold": geos_eq_count,
                "geos_fraction_le_threshold": geos_percentile_le,
                "geos_fraction_ge_threshold": geos_exceedance,
                "geos_fraction_gt_threshold": geos_strict_exceedance,
                "geos_fraction_eq_threshold": geos_eq_fraction,
                "geos_fraction_upper_midrank": geos_upper_midrank,
                "fraction_ge_difference": geos_exceedance - obs_exceedance
                if np.isfinite(geos_exceedance) and np.isfinite(obs_exceedance)
                else float("nan"),
                "fraction_gt_difference": geos_strict_exceedance - obs_strict_exceedance
                if np.isfinite(geos_strict_exceedance) and np.isfinite(obs_strict_exceedance)
                else float("nan"),
                "fraction_midrank_difference": geos_upper_midrank - obs_strict_exceedance
                if np.isfinite(geos_upper_midrank) and np.isfinite(obs_strict_exceedance)
                else float("nan"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  -> Wrote {path}")


def setup_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "#ffffff"


def save_figure(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_basin_summary(rows: list[dict[str, object]], path: Path, dpi: int, months_label: str, observed_label: str, observed_title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=dpi)
    x = np.arange(len(BASIN_ORDER))
    width = 0.26
    obs = np.asarray([next(row for row in rows if row["basin_name"] == basin)["ibtracs_count_year_mean"] for basin in BASIN_ORDER], dtype="float64")
    geos_struct = np.asarray([next(row for row in rows if row["basin_name"] == basin)["geos_structural_member_year_mean"] for basin in BASIN_ORDER], dtype="float64")
    geos_ts = np.asarray([next(row for row in rows if row["basin_name"] == basin)["geos_ts_equiv_member_year_mean"] for basin in BASIN_ORDER], dtype="float64")

    ax.bar(x - width, obs, width, color="#344e86", label=observed_label)
    ax.bar(x, geos_struct, width, color="#9aa7b1", label="GEOS structural candidates/member/year")
    ax.bar(x + width, geos_ts, width, color="#e55934", label="GEOS TS-equiv candidates/member/year")
    ax.set_xticks(x)
    ax.set_xticklabels(BASIN_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"GEOS Candidate Counts vs {observed_title} ({months_label})", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=1)
    save_figure(fig, path, dpi)


def plot_ratio(rows: list[dict[str, object]], path: Path, dpi: int, months_label: str, observed_title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=dpi)
    ratios = np.asarray([next(row for row in rows if row["basin_name"] == basin)["geos_to_ibtracs_count_ratio"] for basin in BASIN_ORDER], dtype="float64")
    colors = [BASINS[basin]["color"] for basin in BASIN_ORDER]
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.bar(np.arange(len(BASIN_ORDER)), np.nan_to_num(ratios, nan=0.0), color=colors, edgecolor="#ffffff", linewidth=0.8)
    ax.set_xticks(np.arange(len(BASIN_ORDER)))
    ax.set_xticklabels(BASIN_ORDER, rotation=20, ha="right")
    ax.set_ylabel("GEOS TS-equiv / IBTrACS count")
    ax.set_title(f"GEOS-to-{observed_title} Ratio by Basin ({months_label})", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, path, dpi)


def plot_percentile_sanity(rows: list[dict[str, object]], path: Path, dpi: int, months_label: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=dpi)
    x = np.arange(len(BASIN_ORDER))
    width = 0.35
    obs = np.asarray(
        [next(row for row in rows if row["basin_name"] == basin)["ibtracs_fraction_gt_34kt"] for basin in BASIN_ORDER],
        dtype="float64",
    )
    geos = np.asarray(
        [next(row for row in rows if row["basin_name"] == basin)["geos_fraction_upper_midrank"] for basin in BASIN_ORDER],
        dtype="float64",
    )
    ax.bar(x - width / 2, obs, width, color="#344e86", label="IBTrACS fraction > 34 kt")
    ax.bar(x + width / 2, geos, width, color="#e55934", label="GEOS upper-tail mid-rank at threshold")
    for x_value, basin_name in zip(x, BASIN_ORDER):
        row = next(item for item in rows if item["basin_name"] == basin_name)
        threshold = float(row["geos_threshold_kt"])
        obs_fraction = float(row["ibtracs_fraction_gt_34kt"])
        geos_fraction = float(row["geos_fraction_upper_midrank"])
        if np.isfinite(threshold) and np.isfinite(obs_fraction) and np.isfinite(geos_fraction):
            ax.text(
                x_value,
                max(obs_fraction, geos_fraction) + 0.025,
                f"T={threshold:.1f} kt",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1e222a",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(BASIN_ORDER, rotation=20, ha="right")
    finite_values = np.concatenate([obs[np.isfinite(obs)], geos[np.isfinite(geos)]])
    ymax = float(np.nanmax(finite_values)) if finite_values.size else 0.2
    ax.set_ylim(0.0, min(1.05, max(0.2, ymax + 0.12)))
    ax.set_ylabel("Fraction of Samples")
    ax.set_title(f"Quantile-Matching Sanity Check: Observed 34 kt vs GEOS Basin Threshold ({months_label})", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, path, dpi)


def plot_yearly_panels(yearly_rows: list[dict[str, object]], path: Path, dpi: int, months_label: str, observed_label: str, observed_title: str) -> None:
    years = sorted({int(row["year"]) for row in yearly_rows})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=dpi, sharex=True)
    for ax, basin_name in zip(axes.ravel(), BASIN_ORDER):
        rows = [row for row in yearly_rows if row["basin_name"] == basin_name]
        obs = np.asarray([row["ibtracs_count_for_comparison"] for row in rows], dtype="float64")
        geos = np.asarray([row["geos_ts_equiv_member_mean_count"] for row in rows], dtype="float64")
        ax.plot(years, obs, color="#344e86", linewidth=1.7, label=observed_label.replace("/year", ""))
        ax.plot(years, geos, color="#e55934", linewidth=1.7, label="GEOS TS-equiv/member")
        ax.set_title(basin_name, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Yearly {months_label} GEOS Candidate Counts vs {observed_title}", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def plot_monthly_summary(rows: list[dict[str, object]], path: Path, dpi: int, months_label: str, observed_label: str) -> None:
    months = sorted({row["month"] for row in rows})
    fig, axes = plt.subplots(1, max(1, len(months)), figsize=(7 * max(1, len(months)), 5.5), dpi=dpi, sharey=True)
    if len(months) == 1:
        axes = [axes]
    x = np.arange(len(BASIN_ORDER))
    width = 0.26
    for ax, month in zip(axes, months):
        month_rows = [row for row in rows if row["month"] == month]
        obs = np.asarray([next(row for row in month_rows if row["basin_name"] == basin)["ibtracs_count_year_mean"] for basin in BASIN_ORDER], dtype="float64")
        geos_struct = np.asarray([next(row for row in month_rows if row["basin_name"] == basin)["geos_structural_member_year_mean"] for basin in BASIN_ORDER], dtype="float64")
        geos = np.asarray([next(row for row in month_rows if row["basin_name"] == basin)["geos_ts_equiv_member_year_mean"] for basin in BASIN_ORDER], dtype="float64")
        ax.bar(x - width, obs, width, color="#344e86", label=observed_label)
        ax.bar(x, geos_struct, width, color="#9aa7b1", label="GEOS structural/member/year")
        ax.bar(x + width, geos, width, color="#e55934", label="GEOS TS-equiv/member/year")
        ax.set_title(MONTH_LABELS.get(month, month), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(BASIN_ORDER, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean Count per Year")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Monthly Count Comparison ({months_label})", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    save_figure(fig, path, dpi)


def print_summary(rows: list[dict[str, object]], observed_label: str) -> None:
    print("")
    print("GEOS vs IBTrACS count summary")
    observed_header = observed_label.replace("IBTrACS TS ", "").replace("/year", "/yr")
    print(f"{'basin':20s} {observed_header:>18s} {'GEOS TS/member/yr':>18s} {'ratio':>8s} {'T_geos':>8s}")
    for row in rows:
        print(
            f"{str(row['basin_name']):20s} "
            f"{float(row['ibtracs_count_year_mean']):18.2f} "
            f"{float(row['geos_ts_equiv_member_year_mean']):18.2f} "
            f"{float(row['geos_to_ibtracs_count_ratio']):8.2f} "
            f"{float(row['geos_threshold_kt']):8.2f}"
        )


def print_percentile_summary(rows: list[dict[str, object]]) -> None:
    print("")
    print("Quantile-matching sanity check")
    print(f"{'basin':20s} {'IB >34':>10s} {'GEOS mid':>10s} {'diff':>10s} {'T_geos':>8s}")
    for row in rows:
        print(
            f"{str(row['basin_name']):20s} "
            f"{float(row['ibtracs_fraction_gt_34kt']):10.3f} "
            f"{float(row['geos_fraction_upper_midrank']):10.3f} "
            f"{float(row['fraction_midrank_difference']):10.3f} "
            f"{float(row['geos_threshold_kt']):8.2f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=[DEFAULT_CANDIDATES])
    parser.add_argument("--thresholds", nargs="*", default=None, help="Optional GEOS threshold CSV path(s), used with --threshold-source csv.")
    parser.add_argument(
        "--threshold-source",
        choices=("recompute", "csv"),
        default="recompute",
        help="Use thresholds recomputed from the current candidate CSVs, or read threshold CSVs. Default: recompute.",
    )
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--months", default="09,10")
    parser.add_argument("--wind-var", default="usa_wind")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument("--basin-method", choices=("boxes", "ibtracs_code"), default="boxes")
    parser.add_argument(
        "--observed-count-metric",
        choices=("active_time", "fixes"),
        default="active_time",
        help="Observed count used in count plots. active_time counts each basin/time once, matching the one GEOS center per basin/time detector.",
    )
    parser.add_argument("--min-lat", type=float, default=-25.0)
    parser.add_argument("--max-lat", type=float, default=50.0)
    parser.add_argument("--expected-members-per-year", type=int, default=0, help="Use this denominator for GEOS member means. Default uses active members visible in candidate CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", default="geos_ibtracs_count_comparison")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--all-hours", action="store_false", dest="synoptic_only", help="Include non-synoptic IBTrACS times.")
    parser.set_defaults(synoptic_only=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_streams()
    setup_style()
    args = parse_args(argv)

    candidate_paths = expand_paths(args.candidates)
    if not candidate_paths:
        print(f"ERROR: no GEOS candidate CSV files matched: {', '.join(args.candidates)}", file=sys.stderr)
        return 1

    print("Reading GEOS candidate CSV files:")
    for path in candidate_paths:
        print(f"  - {path}")
    all_geos_candidates, geos_candidates = read_geos_candidates(candidate_paths, args)
    print(f"Loaded {len(all_geos_candidates):,} GEOS candidate rows in selected years/months.")
    print(f"Using {len(geos_candidates):,} GEOS candidate rows after latitude filter {args.min_lat:.1f} to {args.max_lat:.1f}.")

    print(f"Reading IBTrACS: {args.ibtracs}")
    ibtracs_fixes, wind_samples = read_ibtracs(args)

    recomputed_thresholds = quantile_matched_geos_thresholds(
        all_geos_candidates,
        wind_samples,
        args.threshold_kt,
    )
    if args.threshold_source == "csv":
        if args.thresholds is None:
            threshold_paths = infer_threshold_paths(candidate_paths)
        else:
            threshold_paths = expand_paths(args.thresholds)
        if threshold_paths:
            print("Reading GEOS threshold CSV files:")
            for path in threshold_paths:
                print(f"  - {path}")
        geos_thresholds = read_thresholds(threshold_paths, args.wind_var, args.basin_method)
        geos_thresholds = fill_missing_geos_thresholds(geos_thresholds, recomputed_thresholds)
        print("Using GEOS thresholds from CSV where available; recomputed thresholds fill missing basins.")
    else:
        geos_thresholds = recomputed_thresholds
        print("Using GEOS thresholds recomputed from the current candidate CSVs for this comparison.")

    missing_thresholds = [basin_name for basin_name in BASIN_ORDER if basin_name not in geos_thresholds]
    if missing_thresholds:
        print(f"WARNING: missing GEOS thresholds for: {', '.join(missing_thresholds)}")

    yearly_rows = build_yearly_rows(geos_candidates, ibtracs_fixes, geos_thresholds, args)
    monthly_rows = build_monthly_rows(geos_candidates, ibtracs_fixes, geos_thresholds, args)
    summary_rows = build_summary_rows(yearly_rows)
    percentile_rows = build_percentile_rows(all_geos_candidates, wind_samples, geos_thresholds, args.threshold_kt)

    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    months_label = format_months_label(args.months)
    observed_label = observed_count_label(args.observed_count_metric)
    observed_title = observed_count_title(args.observed_count_metric)
    write_csv(output_dir / f"{args.prefix}_yearly.csv", yearly_rows)
    write_csv(output_dir / f"{args.prefix}_monthly.csv", monthly_rows)
    write_csv(output_dir / f"{args.prefix}_summary.csv", summary_rows)
    write_csv(output_dir / f"{args.prefix}_percentile_sanity.csv", percentile_rows)

    plot_basin_summary(summary_rows, plot_dir / f"{args.prefix}_basin_mean_counts.png", args.dpi, months_label, observed_label, observed_title)
    plot_ratio(summary_rows, plot_dir / f"{args.prefix}_geos_to_ibtracs_ratio.png", args.dpi, months_label, observed_title)
    plot_percentile_sanity(percentile_rows, plot_dir / f"{args.prefix}_percentile_sanity.png", args.dpi, months_label)
    plot_yearly_panels(yearly_rows, plot_dir / f"{args.prefix}_yearly_counts_by_basin.png", args.dpi, months_label, observed_label, observed_title)
    plot_monthly_summary(monthly_rows, plot_dir / f"{args.prefix}_monthly_counts.png", args.dpi, months_label, observed_label)

    print_summary(summary_rows, observed_label)
    print_percentile_summary(percentile_rows)
    if args.expected_members_per_year <= 0:
        print("")
        print("NOTE: GEOS member means use active init/member pairs visible in the candidate CSV.")
        print("      Members with zero structural candidates are not recoverable from candidate-only CSVs.")
        print("      Pass --expected-members-per-year N if you want a fixed denominator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
