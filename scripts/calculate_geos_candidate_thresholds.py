#!/usr/bin/env python3
"""Calculate GEOS basin wind thresholds from structurally gated TC candidates.

This script is the GEOS-side companion to
``calculate_ibtracs_observed_percentiles.py``. It collects model candidate
intensities only after a TC-like structure gate based on
``calculate_tc_conditioned_ace.py``:

    SLP minimum + warm-core anomaly + QV850 anomaly + optional vorticity sign

Unlike the original single-center ACE workflow, this calibration inventory can
write multiple local SLP minima per basin/time. That lets the cached candidate
inventory represent simultaneous TCs in active basins such as the western North
Pacific.

For each basin, the script then maps the observed IBTrACS percentile of 34 kt
onto the GEOS candidate-centered Vmax distribution:

    T_geos_b = percentile(Vmax_geos_candidates_b, percentile_obs_34kt_b)

The result is a basin-dependent model wind threshold that is calibrated against
observed TC-track intensity samples without comparing IBTrACS to all GEOS winds.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def configure_line_buffered_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except (AttributeError, TypeError):
            pass


configure_line_buffered_streams()

try:
    import netCDF4
except ImportError:
    print("ERROR: netCDF4 package is required. Load the earth environment or install netCDF4.", file=sys.stderr)
    sys.exit(2)

# The shared TC-conditioned helper imports matplotlib for plotting. This
# calibration script does not plot, but setting MPLCONFIGDIR keeps remote/manual
# runs quiet when the default home cache is not writable.
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "geos_s2s_tc_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

from calculate_tc_conditioned_ace import (
    BASINS,
    DEFAULT_ATM_COLLECTION,
    DEFAULT_ATM_ROOT,
    DEFAULT_SFC_COLLECTION,
    DEFAULT_SFC_ROOT,
    MPS_TO_KNOTS,
    QV_CANDIDATES,
    SFCIndex,
    SLP_CANDIDATES,
    T_CANDIDATES,
    U_CANDIDATES,
    V_CANDIDATES,
    annulus_mean,
    basin_candidate_center,
    basin_lon_sets,
    basin_has_coverage,
    detect_vertical_dim,
    discover_collection_files,
    find_first_variable,
    forecast_yyyymm,
    index_radius,
    max_near_center,
    qv_to_gpkg,
    read_2d_field,
    relative_vorticity,
    slp_to_hpa,
    to_datetime,
)
from ocean_mask_utils import add_ocean_only_args, build_ocean_checker, row_over_ocean_value


DEFAULT_OBS_PERCENTILES = "data/obs/ibtracs/ibtracs_observed_percentiles.csv"
DEFAULT_OUTPUT_DIR = "data/calibration"


class RuntimeLimitReached(Exception):
    """Raised after a completed time step when --stop-after-hours is exceeded."""


@dataclass
class CandidateRow:
    init_date: str
    ens: str
    valid_time: datetime
    forecast_month: str
    atm_file: str
    atm_time_index: int
    sfc_file: str
    sfc_time_index: int
    sfc_valid_time: datetime
    sfc_delta_hours: float
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
    over_ocean: int


def normalize_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.replace(",", " ").replace(":", " ").split() if item]


def parse_months(value: str | None) -> set[str]:
    months = set()
    for item in parse_list(value):
        month = int(item)
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {item}")
        months.add(f"{month:02d}")
    return months


def read_init_dates(path: str | None, init_date: str | None) -> list[str]:
    if init_date:
        return parse_list(init_date)
    if not path:
        raise ValueError("Pass --init-date or --init-dates-file")

    dates: list[str] = []
    for item in Path(path).read_text().split():
        item = item.strip()
        if item and item not in dates:
            dates.append(item)
    return dates


def discover_ensembles(
    sfc_root: Path,
    atm_root: Path,
    init_date: str,
    requested: list[str],
    sfc_collection: str,
    atm_collection: str,
) -> list[str]:
    if requested and requested != ["all"]:
        return requested

    ensembles = set()
    for root, collection in ((sfc_root, sfc_collection), (atm_root, atm_collection)):
        init_dir = root / "GEOS_fcst" / init_date
        for path in init_dir.glob(f"ens*/{collection}"):
            if path.is_dir():
                ensembles.add(path.parent.name)
    return sorted(ensembles)


def load_observed_percentiles(path: Path, wind_vars: list[str], basin_method: str | None) -> dict[tuple[str, str], dict[str, float | int | str]]:
    rows: dict[tuple[str, str], dict[str, float | int | str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basin_name = row.get("basin_name", "")
            wind_var = row.get("wind_var", "")
            if basin_name not in BASINS:
                continue
            if wind_vars and wind_var not in wind_vars:
                continue
            if basin_method and row.get("basin_method") != basin_method:
                continue

            try:
                percentile = float(row["percentile_obs_threshold"])
                p_obs = float(row["p_obs_threshold"])
                n_samples = int(float(row["n_samples"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(percentile):
                continue

            rows[(basin_name, wind_var)] = {
                "basin_name": basin_name,
                "wind_var": wind_var,
                "observed_percentile": percentile,
                "observed_p": p_obs,
                "observed_n_samples": n_samples,
                "observed_threshold_kt": float(row.get("threshold_kt", "34")),
                "observed_basin_method": row.get("basin_method", ""),
            }
    return rows


def finite_percentile(values: list[float], percentile: float) -> float:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    percentile = min(100.0, max(0.0, percentile))
    return float(np.nanpercentile(array, percentile))


def lon_lat_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = lat1 - lat2
    dlon = abs(normalize_lon(lon1 - lon2))
    mean_lat = math.radians(0.5 * (lat1 + lat2))
    return math.sqrt(dlat * dlat + (dlon * math.cos(mean_lat)) ** 2)


def is_too_close_to_accepted(
    lat_idx: int,
    lon_idx: int,
    accepted_centers: list[tuple[int, int]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    min_separation_deg: float,
) -> bool:
    if min_separation_deg <= 0.0:
        return False
    lat = float(latitudes[lat_idx])
    lon = normalize_lon(float(longitudes[lon_idx]))
    for accepted_lat_idx, accepted_lon_idx in accepted_centers:
        accepted_lat = float(latitudes[accepted_lat_idx])
        accepted_lon = normalize_lon(float(longitudes[accepted_lon_idx]))
        if lon_lat_distance_deg(lat, lon, accepted_lat, accepted_lon) < min_separation_deg:
            return True
    return False


def is_local_minimum(field: np.ndarray, lat_idx: int, lon_idx: int, lat_radius: int, lon_radius: int) -> bool:
    value = float(field[lat_idx, lon_idx])
    if not np.isfinite(value):
        return False
    nlat, nlon = field.shape
    lat_lo = max(0, lat_idx - lat_radius)
    lat_hi = min(nlat, lat_idx + lat_radius + 1)
    lon_offsets = np.arange(-lon_radius, lon_radius + 1)
    lon_indices = (lon_idx + lon_offsets) % nlon
    patch = field[np.ix_(np.arange(lat_lo, lat_hi), lon_indices)]
    if not np.isfinite(patch).any():
        return False
    return value <= float(np.nanmin(patch)) + 1.0e-8 and value < float(np.nanmean(patch)) - 1.0e-8


def basin_low_slp_local_minima(
    field: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    basin_def: dict[str, object],
    max_search_points: int,
    local_min_lat_radius: int,
    local_min_lon_radius: int,
) -> list[tuple[int, int, float]]:
    lat_min, lat_max = basin_def["lat_range"]
    lat_idx = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
    if len(lat_idx) == 0:
        raise ValueError("No latitude points found for basin")

    point_lat_indices: list[np.ndarray] = []
    point_lon_indices: list[np.ndarray] = []
    point_values: list[np.ndarray] = []
    for lon_idx in basin_lon_sets(longitudes, basin_def):
        if len(lon_idx) == 0:
            continue
        patch = field[np.ix_(lat_idx, lon_idx)]
        finite_mask = np.isfinite(patch)
        if not finite_mask.any():
            continue
        patch_i, patch_j = np.where(finite_mask)
        point_lat_indices.append(lat_idx[patch_i])
        point_lon_indices.append(lon_idx[patch_j])
        point_values.append(patch[patch_i, patch_j])

    if not point_values:
        raise ValueError("No finite basin candidate center found")

    candidate_lats = np.concatenate(point_lat_indices)
    candidate_lons = np.concatenate(point_lon_indices)
    candidate_values = np.concatenate(point_values).astype("float64")
    search_count = min(candidate_values.size, max(1, max_search_points))
    if search_count < candidate_values.size:
        lowest_indices = np.argpartition(candidate_values, search_count - 1)[:search_count]
    else:
        lowest_indices = np.arange(candidate_values.size)
    lowest_indices = lowest_indices[np.argsort(candidate_values[lowest_indices])]

    centers: list[tuple[int, int, float]] = []
    for index in lowest_indices:
        lat_i = int(candidate_lats[index])
        lon_i = int(candidate_lons[index])
        if is_local_minimum(field, lat_i, lon_i, local_min_lat_radius, local_min_lon_radius):
            centers.append((lat_i, lon_i, float(candidate_values[index])))

    if centers:
        return centers

    center_lat_idx, center_lon_idx, center_slp = basin_candidate_center(field, latitudes, longitudes, basin_def)
    return [(center_lat_idx, center_lon_idx, center_slp)]


def summarize_values(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "geos_n_candidates": 0,
            "geos_min_vmax_kt": float("nan"),
            "geos_p10_vmax_kt": float("nan"),
            "geos_median_vmax_kt": float("nan"),
            "geos_p90_vmax_kt": float("nan"),
            "geos_max_vmax_kt": float("nan"),
        }
    return {
        "geos_n_candidates": int(array.size),
        "geos_min_vmax_kt": float(np.nanmin(array)),
        "geos_p10_vmax_kt": float(np.nanpercentile(array, 10)),
        "geos_median_vmax_kt": float(np.nanpercentile(array, 50)),
        "geos_p90_vmax_kt": float(np.nanpercentile(array, 90)),
        "geos_max_vmax_kt": float(np.nanmax(array)),
    }


def write_candidates_header(handle) -> csv.DictWriter:
    fieldnames = [
        "init_date",
        "ens",
        "valid_time",
        "forecast_month",
        "atm_file",
        "atm_time_index",
        "sfc_file",
        "sfc_time_index",
        "sfc_valid_time",
        "sfc_delta_hours",
        "basin_name",
        "center_lat",
        "center_lon",
        "vmax_kt",
        "slp_hpa",
        "slp_anom_hpa",
        "warm_core_anom_k",
        "qv850_anom_gpkg",
        "vort850_s1",
        "used_vorticity",
        "over_ocean",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    writer._flush_handle = handle
    return writer


def write_candidate(writer: csv.DictWriter, row: CandidateRow) -> None:
    writer.writerow(
        {
            "init_date": row.init_date,
            "ens": row.ens,
            "valid_time": row.valid_time.strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_month": row.forecast_month,
            "atm_file": row.atm_file,
            "atm_time_index": row.atm_time_index,
            "sfc_file": row.sfc_file,
            "sfc_time_index": row.sfc_time_index,
            "sfc_valid_time": row.sfc_valid_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sfc_delta_hours": row.sfc_delta_hours,
            "basin_name": row.basin_name,
            "center_lat": row.center_lat,
            "center_lon": row.center_lon,
            "vmax_kt": row.vmax_kt,
            "slp_hpa": row.slp_hpa,
            "slp_anom_hpa": row.slp_anom_hpa,
            "warm_core_anom_k": row.warm_core_anom_k,
            "qv850_anom_gpkg": row.qv850_anom_gpkg,
            "vort850_s1": row.vort850_s1,
            "used_vorticity": row.used_vorticity,
            "over_ocean": row.over_ocean,
        }
    )
    flush_handle = getattr(writer, "_flush_handle", None)
    if flush_handle is not None:
        flush_handle.flush()


def candidate_key_from_values(
    init_date: str,
    ens: str,
    valid_time: str,
    basin_name: str,
    center_lat: object,
    center_lon: object,
) -> tuple[str, str, str, str, str, str]:
    try:
        lat_text = f"{float(center_lat):.4f}"
        lon_text = f"{normalize_lon(float(center_lon)):.4f}"
    except (TypeError, ValueError):
        lat_text = str(center_lat)
        lon_text = str(center_lon)
    return (init_date, ens, valid_time, basin_name, lat_text, lon_text)


def candidate_key_from_row(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return candidate_key_from_values(
        row.get("init_date", ""),
        row.get("ens", ""),
        row.get("valid_time", ""),
        row.get("basin_name", ""),
        row.get("center_lat", ""),
        row.get("center_lon", ""),
    )


def candidate_key_from_candidate(row: CandidateRow) -> tuple[str, str, str, str, str, str]:
    return candidate_key_from_values(
        row.init_date,
        row.ens,
        row.valid_time.strftime("%Y-%m-%d %H:%M:%S"),
        row.basin_name,
        row.center_lat,
        row.center_lon,
    )


def progress_key(init_date: str, ens: str, atm_file: object, atm_time_index: object) -> tuple[str, str, str, str]:
    return (init_date, ens, str(atm_file), str(atm_time_index))


def progress_key_from_row(row: dict[str, str]) -> tuple[str, str, str, str]:
    return progress_key(row.get("init_date", ""), row.get("ens", ""), row.get("atm_file", ""), row.get("atm_time_index", ""))


def write_progress(
    writer: csv.DictWriter | None,
    completed_time_keys: set[tuple[str, str, str, str]],
    init_date: str,
    ens: str,
    atm_file: Path,
    atm_time_index: int,
    valid_time: datetime,
    status: str,
) -> None:
    key = progress_key(init_date, ens, atm_file, atm_time_index)
    if key in completed_time_keys:
        return
    completed_time_keys.add(key)
    if writer is None:
        return
    writer.writerow(
        {
            "init_date": init_date,
            "ens": ens,
            "atm_file": str(atm_file),
            "atm_time_index": atm_time_index,
            "valid_time": valid_time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
        }
    )
    flush_handle = getattr(writer, "_flush_handle", None)
    if flush_handle is not None:
        flush_handle.flush()


def check_runtime_limit(args: argparse.Namespace, init_date: str, ens: str, valid_time: datetime) -> None:
    stop_after_hours = getattr(args, "stop_after_hours", 0.0)
    if stop_after_hours <= 0.0:
        return
    elapsed_hours = (time.monotonic() - args._start_monotonic) / 3600.0
    if elapsed_hours >= stop_after_hours:
        raise RuntimeLimitReached(
            f"elapsed={elapsed_hours:.3f} h limit={stop_after_hours:.3f} h after {init_date} {ens} {valid_time:%Y-%m-%d %H:%M:%S}"
        )


def read_completed_time_keys(path: Path) -> set[tuple[str, str, str, str]]:
    completed: set[tuple[str, str, str, str]] = set()
    if not path.exists():
        return completed
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            completed.add(progress_key_from_row(row))
    return completed


def open_progress_writer(path: Path, append: bool) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() and path.stat().st_size > 0 else "w"
    handle = path.open(mode, newline="", buffering=1)
    fieldnames = ["init_date", "ens", "atm_file", "atm_time_index", "valid_time", "status"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()
    writer._flush_handle = handle
    return writer, handle


def load_existing_candidates(
    path: Path,
    geos_vmax_by_basin: dict[str, list[float]],
    basin_stats: dict[str, dict[str, int]],
    candidate_keys: set[tuple[str, str, str, str, str, str]],
    ocean_only: bool,
    ocean_checker=None,
) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0

    loaded = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basin_name = row.get("basin_name", "")
            if basin_name not in BASINS:
                continue
            key = candidate_key_from_row(row)
            if key in candidate_keys:
                continue
            try:
                vmax_kt = float(row.get("vmax_kt", "nan"))
            except ValueError:
                continue
            if not np.isfinite(vmax_kt):
                continue
            over_ocean = row_over_ocean_value(row)
            if ocean_only:
                is_ocean = over_ocean is not False
                if is_ocean and ocean_checker is not None:
                    try:
                        center_lat = float(row.get("center_lat", "nan"))
                        center_lon = normalize_lon(float(row.get("center_lon", "nan")))
                    except ValueError:
                        center_lat = float("nan")
                        center_lon = float("nan")
                    is_ocean = ocean_checker.is_ocean(center_lat, center_lon)
                if not is_ocean:
                    basin_stats[basin_name]["resumed_land_candidates_skipped"] += 1
                    continue
            candidate_keys.add(key)
            geos_vmax_by_basin[basin_name].append(vmax_kt)
            basin_stats[basin_name]["accepted_candidates"] += 1
            basin_stats[basin_name]["resumed_existing_candidates"] += 1
            loaded += 1
    return loaded


def open_candidate_writer(path: Path, resume: bool) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and path.exists() and path.stat().st_size > 0 else "w"
    handle = path.open(mode, newline="", buffering=1)
    writer = write_candidates_header(handle) if mode == "w" else csv.DictWriter(handle, fieldnames=write_candidate_fieldnames())
    if mode == "a":
        writer._flush_handle = handle
    return writer, handle


def write_candidate_fieldnames() -> list[str]:
    return [
        "init_date",
        "ens",
        "valid_time",
        "forecast_month",
        "atm_file",
        "atm_time_index",
        "sfc_file",
        "sfc_time_index",
        "sfc_valid_time",
        "sfc_delta_hours",
        "basin_name",
        "center_lat",
        "center_lon",
        "vmax_kt",
        "slp_hpa",
        "slp_anom_hpa",
        "warm_core_anom_k",
        "qv850_anom_gpkg",
        "vort850_s1",
        "used_vorticity",
        "over_ocean",
    ]


def collect_candidates_for_member(
    args: argparse.Namespace,
    init_date: str,
    ens: str,
    forecast_months: set[str],
    candidate_writer: csv.DictWriter,
    progress_writer: csv.DictWriter | None,
    candidate_keys: set[tuple[str, str, str, str, str, str]],
    completed_time_keys: set[tuple[str, str, str, str]],
    geos_vmax_by_basin: dict[str, list[float]],
    basin_stats: dict[str, dict[str, int]],
) -> None:
    sfc_root = Path(args.sfc_root)
    atm_root = Path(args.atm_root)
    sfc_files = discover_collection_files(sfc_root, init_date, ens, args.sfc_collection)
    atm_files = discover_collection_files(atm_root, init_date, ens, args.atm_collection)

    if not sfc_files:
        print(f"WARNING: no SFC files for {init_date} {ens}; skipping")
        return
    if not atm_files:
        print(f"WARNING: no ATM files for {init_date} {ens}; skipping")
        return

    try:
        sfc_index = SFCIndex(sfc_files, args.sfc_collection, forecast_months)
    except Exception as exc:
        print(f"WARNING: failed to index SFC for {init_date} {ens}: {exc}; skipping")
        return

    if getattr(sfc_index, "skipped_no_wind_files", 0):
        examples = ", ".join(getattr(sfc_index, "skipped_no_wind_examples", []))
        print(
            f"WARNING: skipped {sfc_index.skipped_no_wind_files} SFC files without recognized wind variables "
            f"for {init_date} {ens}. Examples: {examples}"
        )

    try:
        latitudes = sfc_index.latitudes
        longitudes = sfc_index.longitudes
        if latitudes is None or longitudes is None:
            print(f"WARNING: missing SFC coordinates for {init_date} {ens}; skipping")
            return

        basin_coverage = {name: basin_has_coverage(latitudes, longitudes, basin_def) for name, basin_def in BASINS.items()}
        lat_env_radius = index_radius(latitudes, args.environment_radius_deg)
        lon_env_radius = index_radius(longitudes, args.environment_radius_deg)
        lat_core_radius = index_radius(latitudes, args.inner_core_radius_deg)
        lon_core_radius = index_radius(longitudes, args.inner_core_radius_deg)
        lat_wind_radius = index_radius(latitudes, args.wind_search_radius_deg)
        lon_wind_radius = index_radius(longitudes, args.wind_search_radius_deg)
        lat_local_min_radius = index_radius(latitudes, args.local_min_radius_deg)
        lon_local_min_radius = index_radius(longitudes, args.local_min_radius_deg)
        ocean_checker = None
        if args.ocean_only:
            sfc_mask_path = sfc_index.entries[0].file_path if sfc_index.entries else None
            ocean_checker, ocean_warning = build_ocean_checker(
                args.ocean_mask_source,
                sfc_path=sfc_mask_path,
                mask_file=args.ocean_mask_file,
                threshold=args.ocean_threshold,
                require_mask=True,
            )
            print(f"Ocean-only filter enabled for {init_date} {ens}: source={ocean_checker.source}")
            if ocean_warning:
                print(f"WARNING: ocean mask fallback for {init_date} {ens}: {ocean_warning}")

        print(f"Processing {init_date} {ens}: SFC files={len(sfc_files)} ATM files={len(atm_files)}")

        for atm_path in atm_files:
            yyyymm = forecast_yyyymm(atm_path.name, args.atm_collection)
            if forecast_months and (yyyymm is None or yyyymm[-2:] not in forecast_months):
                continue

            try:
                with netCDF4.Dataset(atm_path, "r") as ds:
                    slp_name = find_first_variable(ds, SLP_CANDIDATES)
                    t_name = find_first_variable(ds, T_CANDIDATES)
                    qv_name = find_first_variable(ds, QV_CANDIDATES)
                    u_name = find_first_variable(ds, U_CANDIDATES)
                    v_name = find_first_variable(ds, V_CANDIDATES)
                    if slp_name is None or t_name is None or qv_name is None:
                        print(f"WARNING: {atm_path.name} missing SLP/T/QV; skipping")
                        continue

                    try:
                        level_dim, level_indices = detect_vertical_dim(ds, (850.0, 500.0, 200.0))
                    except Exception as exc:
                        print(f"WARNING: {atm_path.name} missing required pressure levels: {exc}; skipping")
                        continue

                    time_values = ds.variables["time"][:]
                    time_units = ds.variables["time"].units
                    atm_times = [to_datetime(value) for value in netCDF4.num2date(time_values, time_units)]

                    for time_index, valid_time in enumerate(atm_times):
                        if forecast_months and valid_time.strftime("%m") not in forecast_months:
                            continue

                        time_key = progress_key(init_date, ens, atm_path, time_index)
                        if time_key in completed_time_keys:
                            for basin_name in BASINS:
                                basin_stats[basin_name]["skipped_completed_times"] += 1
                            continue

                        sfc_match = sfc_index.nearest(valid_time, args.sfc_match_tolerance_hours)
                        if sfc_match is None:
                            for basin_name in BASINS:
                                basin_stats[basin_name]["missing_sfc_match"] += 1
                            write_progress(
                                progress_writer,
                                completed_time_keys,
                                init_date,
                                ens,
                                atm_path,
                                time_index,
                                valid_time,
                                "missing_sfc_match",
                            )
                            check_runtime_limit(args, init_date, ens, valid_time)
                            continue
                        sfc_delta_hours = abs((sfc_match.valid_time - valid_time).total_seconds()) / 3600.0

                        try:
                            us_sfc, vs_sfc = sfc_index.load_wind(sfc_match)
                        except Exception as exc:
                            print(f"WARNING: could not load SFC wind for {sfc_match.file_path}: {exc}; skipping time")
                            for basin_name in BASINS:
                                basin_stats[basin_name]["missing_sfc_wind"] += 1
                            write_progress(
                                progress_writer,
                                completed_time_keys,
                                init_date,
                                ens,
                                atm_path,
                                time_index,
                                valid_time,
                                "missing_sfc_wind",
                            )
                            check_runtime_limit(args, init_date, ens, valid_time)
                            continue
                        sfc_ws_kt = np.sqrt(us_sfc**2 + vs_sfc**2) * MPS_TO_KNOTS

                        slp_hpa = slp_to_hpa(
                            read_2d_field(ds.variables[slp_name], time_index=time_index),
                            getattr(ds.variables[slp_name], "units", ""),
                        )
                        t850 = read_2d_field(ds.variables[t_name], time_index=time_index, level_dim=level_dim, level_index=level_indices[850.0])
                        t500 = read_2d_field(ds.variables[t_name], time_index=time_index, level_dim=level_dim, level_index=level_indices[500.0])
                        t200 = read_2d_field(ds.variables[t_name], time_index=time_index, level_dim=level_dim, level_index=level_indices[200.0])
                        qv850 = qv_to_gpkg(
                            read_2d_field(ds.variables[qv_name], time_index=time_index, level_dim=level_dim, level_index=level_indices[850.0]),
                            getattr(ds.variables[qv_name], "units", ""),
                        )
                        warm_core_field = 0.5 * (t500 + t200) - t850

                        u850 = None
                        v850 = None
                        if not args.ignore_vorticity and u_name is not None and v_name is not None:
                            u850 = read_2d_field(ds.variables[u_name], time_index=time_index, level_dim=level_dim, level_index=level_indices[850.0])
                            v850 = read_2d_field(ds.variables[v_name], time_index=time_index, level_dim=level_dim, level_index=level_indices[850.0])

                        for basin_name, basin_def in BASINS.items():
                            basin_stats[basin_name]["evaluated_steps"] += 1
                            if not basin_coverage[basin_name]:
                                basin_stats[basin_name]["no_basin_coverage"] += 1
                                continue

                            try:
                                if args.max_centers_per_basin == 1:
                                    candidate_centers = [basin_candidate_center(slp_hpa, latitudes, longitudes, basin_def)]
                                else:
                                    candidate_centers = basin_low_slp_local_minima(
                                        slp_hpa,
                                        latitudes,
                                        longitudes,
                                        basin_def,
                                        args.center_search_points,
                                        lat_local_min_radius,
                                        lon_local_min_radius,
                                    )
                            except ValueError:
                                basin_stats[basin_name]["no_candidate_center"] += 1
                                continue

                            accepted_centers: list[tuple[int, int]] = []
                            for center_lat_idx, center_lon_idx, center_slp in candidate_centers:
                                if len(accepted_centers) >= args.max_centers_per_basin:
                                    basin_stats[basin_name]["max_centers_reached"] += 1
                                    break
                                basin_stats[basin_name]["candidate_centers_evaluated"] += 1

                                center_lat = float(latitudes[center_lat_idx])
                                center_lon = normalize_lon(float(longitudes[center_lon_idx]))
                                over_ocean = 1
                                if ocean_checker is not None:
                                    over_ocean = int(ocean_checker.is_ocean(center_lat, center_lon))
                                    if not over_ocean:
                                        basin_stats[basin_name]["rejected_land"] += 1
                                        continue

                                slp_env = annulus_mean(
                                    slp_hpa,
                                    center_lat_idx,
                                    center_lon_idx,
                                    lat_env_radius,
                                    lon_env_radius,
                                    lat_core_radius,
                                    lon_core_radius,
                                )
                                warm_env = annulus_mean(
                                    warm_core_field,
                                    center_lat_idx,
                                    center_lon_idx,
                                    lat_env_radius,
                                    lon_env_radius,
                                    lat_core_radius,
                                    lon_core_radius,
                                )
                                qv_env = annulus_mean(
                                    qv850,
                                    center_lat_idx,
                                    center_lon_idx,
                                    lat_env_radius,
                                    lon_env_radius,
                                    lat_core_radius,
                                    lon_core_radius,
                                )

                                slp_anom = float(center_slp - slp_env)
                                warm_anom = float(warm_core_field[center_lat_idx, center_lon_idx] - warm_env)
                                qv_anom = float(qv850[center_lat_idx, center_lon_idx] - qv_env)
                                criteria = [
                                    slp_anom < 0.0,
                                    warm_anom > 0.0,
                                    qv_anom > 0.0,
                                ]

                                vort850 = float("nan")
                                used_vorticity = 0
                                if u850 is not None and v850 is not None:
                                    vort850 = relative_vorticity(u850, v850, latitudes, longitudes, center_lat_idx, center_lon_idx)
                                    if np.isfinite(vort850):
                                        used_vorticity = 1
                                        criteria.append(vort850 > 0.0 if center_lat >= 0.0 else vort850 < 0.0)

                                if not all(criteria):
                                    basin_stats[basin_name]["rejected_structure"] += 1
                                    continue

                                if is_too_close_to_accepted(
                                    center_lat_idx,
                                    center_lon_idx,
                                    accepted_centers,
                                    latitudes,
                                    longitudes,
                                    args.min_center_separation_deg,
                                ):
                                    basin_stats[basin_name]["duplicate_center"] += 1
                                    continue

                                vmax_kt, _, _ = max_near_center(
                                    sfc_ws_kt,
                                    center_lat_idx,
                                    center_lon_idx,
                                    lat_wind_radius,
                                    lon_wind_radius,
                                )
                                if not np.isfinite(vmax_kt):
                                    basin_stats[basin_name]["missing_vmax"] += 1
                                    continue

                                candidate_row = CandidateRow(
                                    init_date=init_date,
                                    ens=ens,
                                    valid_time=valid_time,
                                    forecast_month=valid_time.strftime("%m"),
                                    atm_file=str(atm_path),
                                    atm_time_index=time_index,
                                    sfc_file=str(sfc_match.file_path),
                                    sfc_time_index=sfc_match.time_index,
                                    sfc_valid_time=sfc_match.valid_time,
                                    sfc_delta_hours=sfc_delta_hours,
                                    basin_name=basin_name,
                                    center_lat=center_lat,
                                    center_lon=center_lon,
                                    vmax_kt=float(vmax_kt),
                                    slp_hpa=float(center_slp),
                                    slp_anom_hpa=slp_anom,
                                    warm_core_anom_k=warm_anom,
                                    qv850_anom_gpkg=qv_anom,
                                    vort850_s1=float(vort850),
                                    used_vorticity=used_vorticity,
                                    over_ocean=over_ocean,
                                )
                                candidate_key = candidate_key_from_candidate(candidate_row)
                                if candidate_key in candidate_keys:
                                    accepted_centers.append((center_lat_idx, center_lon_idx))
                                    basin_stats[basin_name]["duplicate_resume_candidate"] += 1
                                    continue

                                candidate_keys.add(candidate_key)
                                accepted_centers.append((center_lat_idx, center_lon_idx))
                                basin_stats[basin_name]["accepted_candidates"] += 1
                                geos_vmax_by_basin[basin_name].append(float(vmax_kt))
                                write_candidate(candidate_writer, candidate_row)

                        write_progress(
                            progress_writer,
                            completed_time_keys,
                            init_date,
                            ens,
                            atm_path,
                            time_index,
                            valid_time,
                            "completed",
                        )
                        check_runtime_limit(args, init_date, ens, valid_time)
            except OSError as exc:
                print(f"WARNING: could not read {atm_path}: {exc}; skipping")
    finally:
        sfc_index.close()


def write_thresholds(
    path: Path,
    observed_rows: dict[tuple[str, str], dict[str, float | int | str]],
    observed_wind_vars: list[str],
    geos_vmax_by_basin: dict[str, list[float]],
    basin_stats: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "basin_name",
        "observed_wind_var",
        "observed_basin_method",
        "observed_threshold_kt",
        "observed_p",
        "observed_percentile",
        "observed_n_samples",
        "geos_threshold_kt",
        "geos_n_candidates",
        "geos_min_vmax_kt",
        "geos_p10_vmax_kt",
        "geos_median_vmax_kt",
        "geos_p90_vmax_kt",
        "geos_max_vmax_kt",
        "evaluated_steps",
        "candidate_centers_evaluated",
        "accepted_candidates",
        "resumed_existing_candidates",
        "duplicate_resume_candidate",
        "skipped_completed_times",
        "resumed_land_candidates_skipped",
        "rejected_structure",
        "rejected_land",
        "duplicate_center",
        "max_centers_reached",
        "missing_sfc_match",
        "missing_sfc_wind",
        "init_dates",
        "ensembles",
        "months",
        "environment_radius_deg",
        "inner_core_radius_deg",
        "wind_search_radius_deg",
        "max_centers_per_basin",
        "min_center_separation_deg",
        "local_min_radius_deg",
        "center_search_points",
        "resume",
        "progress_path",
        "stopped_early",
        "ocean_only",
        "ocean_mask_source",
        "ocean_threshold",
        "ignore_vorticity",
    ]

    init_dates_text = ",".join(args._resolved_init_dates)
    ensembles_text = ",".join(args._resolved_ensembles)
    months_text = ",".join(sorted(args._resolved_months))

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for basin_name in BASINS:
            stats = summarize_values(geos_vmax_by_basin[basin_name])
            for wind_var in observed_wind_vars:
                obs = observed_rows.get((basin_name, wind_var))
                if obs is None:
                    geos_threshold = float("nan")
                    obs_values = {
                        "observed_basin_method": "",
                        "observed_threshold_kt": float("nan"),
                        "observed_p": float("nan"),
                        "observed_percentile": float("nan"),
                        "observed_n_samples": 0,
                    }
                else:
                    observed_percentile = float(obs["observed_percentile"])
                    geos_threshold = finite_percentile(geos_vmax_by_basin[basin_name], observed_percentile)
                    obs_values = obs

                writer.writerow(
                    {
                        "basin_name": basin_name,
                        "observed_wind_var": wind_var,
                        "observed_basin_method": obs_values["observed_basin_method"],
                        "observed_threshold_kt": obs_values["observed_threshold_kt"],
                        "observed_p": obs_values["observed_p"],
                        "observed_percentile": obs_values["observed_percentile"],
                        "observed_n_samples": obs_values["observed_n_samples"],
                        "geos_threshold_kt": geos_threshold,
                        **stats,
                        "evaluated_steps": basin_stats[basin_name]["evaluated_steps"],
                        "candidate_centers_evaluated": basin_stats[basin_name]["candidate_centers_evaluated"],
                        "accepted_candidates": basin_stats[basin_name]["accepted_candidates"],
                        "resumed_existing_candidates": basin_stats[basin_name]["resumed_existing_candidates"],
                        "duplicate_resume_candidate": basin_stats[basin_name]["duplicate_resume_candidate"],
                        "skipped_completed_times": basin_stats[basin_name]["skipped_completed_times"],
                        "resumed_land_candidates_skipped": basin_stats[basin_name]["resumed_land_candidates_skipped"],
                        "rejected_structure": basin_stats[basin_name]["rejected_structure"],
                        "rejected_land": basin_stats[basin_name]["rejected_land"],
                        "duplicate_center": basin_stats[basin_name]["duplicate_center"],
                        "max_centers_reached": basin_stats[basin_name]["max_centers_reached"],
                        "missing_sfc_match": basin_stats[basin_name]["missing_sfc_match"],
                        "missing_sfc_wind": basin_stats[basin_name]["missing_sfc_wind"],
                        "init_dates": init_dates_text,
                        "ensembles": ensembles_text,
                        "months": months_text,
                        "environment_radius_deg": args.environment_radius_deg,
                        "inner_core_radius_deg": args.inner_core_radius_deg,
                        "wind_search_radius_deg": args.wind_search_radius_deg,
                        "max_centers_per_basin": args.max_centers_per_basin,
                        "min_center_separation_deg": args.min_center_separation_deg,
                        "local_min_radius_deg": args.local_min_radius_deg,
                        "center_search_points": args.center_search_points,
                        "resume": int(args.resume),
                        "progress_path": str(args._progress_path),
                        "stopped_early": int(getattr(args, "_stopped_early", False)),
                        "ocean_only": int(args.ocean_only),
                        "ocean_mask_source": args.ocean_mask_source,
                        "ocean_threshold": args.ocean_threshold,
                        "ignore_vorticity": int(args.ignore_vorticity),
                    }
                )


def print_threshold_table(threshold_path: Path) -> None:
    with threshold_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        print("")
        print("GEOS basin candidate wind thresholds")
        print(f"{'basin':20s} {'obs_wind':10s} {'obs_pct':>8s} {'n_geos':>8s} {'T_geos':>8s}")
        for row in reader:
            try:
                threshold = float(row["geos_threshold_kt"])
                obs_pct = float(row["observed_percentile"])
                n_geos = int(float(row["geos_n_candidates"]))
            except ValueError:
                threshold = float("nan")
                obs_pct = float("nan")
                n_geos = 0
            print(
                f"{row['basin_name']:20s} {row['observed_wind_var']:10s} "
                f"{obs_pct:8.2f} {n_geos:8d} {threshold:8.2f}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfc-root", default=DEFAULT_SFC_ROOT)
    parser.add_argument("--atm-root", default=DEFAULT_ATM_ROOT)
    parser.add_argument("--sfc-collection", default=DEFAULT_SFC_COLLECTION)
    parser.add_argument("--atm-collection", default=DEFAULT_ATM_COLLECTION)
    parser.add_argument("--init-date", default=None, help="One or more init dates separated by comma/colon/space.")
    parser.add_argument("--init-dates-file", default=None)
    parser.add_argument("--ens", default="ens1", help="One or more ensembles, or 'all'.")
    parser.add_argument("--months", default="09,10")
    parser.add_argument("--observed-percentiles", default=DEFAULT_OBS_PERCENTILES)
    parser.add_argument("--observed-wind-vars", default="usa_wind")
    parser.add_argument("--observed-basin-method", default="boxes")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--sfc-match-tolerance-hours", type=float, default=3.1)
    parser.add_argument("--environment-radius-deg", type=float, default=5.0)
    parser.add_argument("--inner-core-radius-deg", type=float, default=1.5)
    parser.add_argument("--wind-search-radius-deg", type=float, default=3.0)
    parser.add_argument(
        "--max-centers-per-basin",
        type=int,
        default=5,
        help="Maximum accepted candidate centers to write per basin/time. Use 1 to reproduce the old single-center behavior.",
    )
    parser.add_argument(
        "--min-center-separation-deg",
        type=float,
        default=6.0,
        help="Minimum approximate distance in degrees between accepted centers in the same basin/time.",
    )
    parser.add_argument(
        "--local-min-radius-deg",
        type=float,
        default=2.0,
        help="Neighborhood radius used to identify SLP local minima before structure checks.",
    )
    parser.add_argument(
        "--center-search-points",
        type=int,
        default=200,
        help="Number of lowest-SLP basin grid points inspected for local minima at each basin/time.",
    )
    parser.add_argument("--ignore-vorticity", action="store_true")
    add_ocean_only_args(parser)
    parser.add_argument("--max-init-dates", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing candidate CSV and skip time steps listed in the progress CSV.")
    parser.add_argument("--progress-path", default=None, help="Optional resume progress CSV path. Default is OUTPUT_PREFIX_progress.csv.")
    parser.add_argument(
        "--stop-after-hours",
        type=float,
        default=0.0,
        help="If positive, stop cleanly after this many elapsed hours, after finishing the current ATM time step.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_centers_per_basin < 1:
        print("ERROR: --max-centers-per-basin must be >= 1", file=sys.stderr)
        return 1
    if args.center_search_points < 1:
        print("ERROR: --center-search-points must be >= 1", file=sys.stderr)
        return 1
    if args.min_center_separation_deg < 0.0:
        print("ERROR: --min-center-separation-deg must be >= 0", file=sys.stderr)
        return 1
    if args.local_min_radius_deg <= 0.0:
        print("ERROR: --local-min-radius-deg must be > 0", file=sys.stderr)
        return 1
    if not (0.0 <= args.ocean_threshold <= 1.0):
        print("ERROR: --ocean-threshold must be between 0 and 1", file=sys.stderr)
        return 1
    if args.stop_after_hours < 0.0:
        print("ERROR: --stop-after-hours must be >= 0", file=sys.stderr)
        return 1

    args._start_monotonic = time.monotonic()
    init_dates = read_init_dates(args.init_dates_file, args.init_date)
    if args.max_init_dates > 0:
        init_dates = init_dates[: args.max_init_dates]
    if not init_dates:
        print("ERROR: no initialization dates selected", file=sys.stderr)
        return 1

    forecast_months = parse_months(args.months)
    observed_wind_vars = parse_list(args.observed_wind_vars)
    observed_rows = load_observed_percentiles(
        Path(args.observed_percentiles),
        observed_wind_vars,
        args.observed_basin_method,
    )
    if not observed_rows:
        print("ERROR: no matching observed percentile rows were found", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        if len(init_dates) == 1:
            init_label = init_dates[0]
        else:
            init_label = f"{init_dates[0]}_{init_dates[-1]}"
        prefix = f"geos_candidate_thresholds_{init_label}_{args.ens.replace(',', '-')}"

    candidates_path = output_dir / f"{prefix}_candidates.csv"
    thresholds_path = output_dir / f"{prefix}.csv"
    progress_path = Path(args.progress_path) if args.progress_path else output_dir / f"{prefix}_progress.csv"
    args._progress_path = progress_path
    print(f"Selected init dates: {len(init_dates)} ({init_dates[0]} to {init_dates[-1]})", flush=True)
    print(f"Selected months: {','.join(sorted(forecast_months))}", flush=True)
    print(f"Candidate inventory will be written to: {candidates_path}", flush=True)
    print(f"Threshold summary will be written to: {thresholds_path}", flush=True)
    print(f"Progress CSV will be written to: {progress_path}", flush=True)
    print(f"Resume mode: {int(args.resume)}", flush=True)
    print(f"Stop after hours: {args.stop_after_hours}", flush=True)
    print(f"Ocean-only: {int(args.ocean_only)} source={args.ocean_mask_source} threshold={args.ocean_threshold}", flush=True)

    geos_vmax_by_basin: dict[str, list[float]] = {basin_name: [] for basin_name in BASINS}
    basin_stats: dict[str, dict[str, int]] = {
        basin_name: defaultdict(int)
        for basin_name in BASINS
    }
    candidate_keys: set[tuple[str, str, str, str, str, str]] = set()
    completed_time_keys = read_completed_time_keys(progress_path) if args.resume else set()
    if args.resume and completed_time_keys:
        print(f"Loaded {len(completed_time_keys)} completed time steps from {progress_path}", flush=True)
    resume_ocean_checker = None
    if args.resume and args.ocean_only:
        resume_ocean_checker, resume_ocean_warning = build_ocean_checker(
            args.ocean_mask_source,
            mask_file=args.ocean_mask_file,
            threshold=args.ocean_threshold,
            require_mask=True,
        )
        print(f"Ocean-only resume filter enabled: source={resume_ocean_checker.source}", flush=True)
        if resume_ocean_warning:
            print(f"WARNING: ocean mask fallback while loading existing candidates: {resume_ocean_warning}", flush=True)
    loaded_existing = (
        load_existing_candidates(
            candidates_path,
            geos_vmax_by_basin,
            basin_stats,
            candidate_keys,
            args.ocean_only,
            resume_ocean_checker,
        )
        if args.resume
        else 0
    )
    if loaded_existing:
        print(f"Loaded {loaded_existing} existing candidate rows from {candidates_path}", flush=True)

    resolved_ensembles: list[str] = []
    stopped_early = False
    candidate_writer, candidate_handle = open_candidate_writer(candidates_path, args.resume)
    progress_writer, progress_handle = open_progress_writer(progress_path, args.resume)
    try:
        for init_date in init_dates:
            ensembles = discover_ensembles(
                Path(args.sfc_root),
                Path(args.atm_root),
                init_date,
                parse_list(args.ens),
                args.sfc_collection,
                args.atm_collection,
            )
            if not ensembles:
                print(f"WARNING: no ensembles discovered for {init_date}; skipping")
                continue
            for ens in ensembles:
                if ens not in resolved_ensembles:
                    resolved_ensembles.append(ens)
                collect_candidates_for_member(
                    args=args,
                    init_date=init_date,
                    ens=ens,
                    forecast_months=forecast_months,
                    candidate_writer=candidate_writer,
                    progress_writer=progress_writer,
                    candidate_keys=candidate_keys,
                    completed_time_keys=completed_time_keys,
                    geos_vmax_by_basin=geos_vmax_by_basin,
                    basin_stats=basin_stats,
                )
    except RuntimeLimitReached as exc:
        stopped_early = True
        print(f"Runtime limit reached; stopping cleanly: {exc}", flush=True)
    finally:
        candidate_handle.close()
        progress_handle.close()

    args._stopped_early = stopped_early
    args._resolved_init_dates = init_dates
    args._resolved_ensembles = resolved_ensembles or parse_list(args.ens)
    args._resolved_months = forecast_months
    write_thresholds(
        thresholds_path,
        observed_rows,
        observed_wind_vars,
        geos_vmax_by_basin,
        basin_stats,
        args,
    )

    print_threshold_table(thresholds_path)
    print("")
    print(f"Wrote accepted GEOS candidates: {candidates_path}")
    print(f"Wrote GEOS basin thresholds: {thresholds_path}")
    if stopped_early:
        print("Stopped early after --stop-after-hours; rerun with --resume to continue.")
        return 75
    return 0


if __name__ == "__main__":
    sys.exit(main())
