#!/usr/bin/env python3
"""Calculate TC-conditioned 2D Spatial and Temporal ACE from GEOS S2S3 winds and ATM structure.

This script is a unified master pipeline that:
1. Loads SFC surface winds and ATM structure variables (SLP, T, QV, U, V).
2. Performs robust longitude coordinate standardization ([0, 360] -> [-180, 180]).
3. Evaluates low-pressure candidate centers in each basin using a multivariate vertical structure gate:
   - Negative local SLP anomaly (isolated via an environmental annulus)
   - Positive warm-core anomaly from T(850/500/200 hPa)
   - Positive low-level moisture anomaly from QV(850 hPa)
   - Hemisphere-consistent 850-hPa relative vorticity sign (optional)
4. If a TC is structurally approved at a given time step:
   - Grid-point surface winds within a storm search radius (e.g. 3 degrees) of the center are
     accumulated into a 2D spatial local ACE field.
   - The maximum surface wind (vmax_kt) in the storm area is used to accumulate basin-wide
     cumulative ACE curves over time.
5. Caches calculations to a netCDF4 file.
6. Generates three publication-quality diagnostic plots (focused spatial map, cumulative curves,
   and global Pacific-centered multi-basin spatial map) using the clean TC-conditioned spatial ACE.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    print("ERROR: netCDF4 package is required. Install with: pip install netCDF4", file=sys.stderr)
    sys.exit(2)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: Cartopy not found. Plotting will fall back to standard coordinates.", file=sys.stderr)


DEFAULT_SFC_ROOT = "/nobackupp27/afahad/project/GEOS-S2S_TC/data"
DEFAULT_ATM_ROOT = "/nobackupp17/afahad/GEOSS2S3_atm"
DEFAULT_SFC_COLLECTION = "sfc_tavg_3hr_glo_L720x361_sfc"
DEFAULT_ATM_COLLECTION = "atm_inst_6hr_glo_L720x361_p49"
DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_PLOT_DIR = "plots"
DEFAULT_THRESHOLD_COLUMN = "geos_threshold_kt"

MPS_TO_KNOTS = 1.94384
EARTH_RADIUS_M = 6_371_000.0
TS_THRESHOLD_KNOTS_DEFAULT = 17.0

# Variable candidates for ATM structure gating
SLP_CANDIDATES = ("SLP", "slp", "PSL", "SeaLevelPressure")
T_CANDIDATES = ("T", "t", "TMP")
QV_CANDIDATES = ("QV", "qv", "Q", "QVAPOR")
U_CANDIDATES = ("U", "u", "UGRD")
V_CANDIDATES = ("V", "v", "VGRD")

# Variable candidates for SFC surface winds
US_CANDIDATES = ("US", "us", "U10M", "u10")
VS_CANDIDATES = ("VS", "vs", "V10M", "v10")

BASINS = {
    "North Atlantic": {
        "lat_range": (0.0, 45.0),
        "lon_range": (-100.0, -10.0),
        "color": "#e55934",
        "label_xy": (-55.0, 38.0)
    },
    "Northeast Pacific": {
        "lat_range": (0.0, 40.0),
        "lon_range": (-180.0, -100.0),
        "color": "#f3a712",
        "label_xy": (-135.0, 32.0)
    },
    "Northwest Pacific": {
        "lat_range": (0.0, 45.0),
        "lon_range": (100.0, 180.0),
        "color": "#2ec4b6",
        "label_xy": (140.0, 38.0)
    },
    "North Indian": {
        "lat_range": (0.0, 40.0),
        "lon_range": (40.0, 100.0),
        "color": "#9b5de5",
        "label_xy": (70.0, 32.0)
    },
    "South Indian": {
        "lat_range": (-40.0, 0.0),
        "lon_range": (20.0, 135.0),
        "color": "#00bbf9",
        "label_xy": (80.0, -32.0)
    },
    "South Pacific": {
        "lat_range": (-40.0, 0.0),
        "lon_ranges": [(135.0, 180.0), (-180.0, -120.0)],
        "color": "#ff007f",
        "label_xy": (-155.0, -32.0)
    },
}


@dataclass
class SFCMatch:
    valid_time: datetime
    file_path: Path
    time_index: int


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def basin_threshold_signature(thresholds: dict[str, float]) -> str:
    return ";".join(f"{basin_name}={float(thresholds.get(basin_name, np.nan)):.4f}" for basin_name in BASINS)


def load_basin_thresholds(
    path: Path | None,
    fallback_threshold: float,
    wind_var: str = "usa_wind",
    basin_method: str = "boxes",
    threshold_column: str = "auto",
) -> tuple[dict[str, float], str, str]:
    thresholds = {basin_name: float(fallback_threshold) for basin_name in BASINS}
    if path is None:
        return thresholds, "fixed", f"fixed:{fallback_threshold:.2f}kt"

    if not path.exists():
        raise FileNotFoundError(f"Threshold CSV does not exist: {path}")

    source_columns = (DEFAULT_THRESHOLD_COLUMN, "count_match_threshold_kt", "T_geos") if threshold_column == "auto" else (threshold_column,)
    loaded: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basin_name = row.get("basin_name", "")
            if basin_name not in BASINS:
                continue
            if row.get("observed_wind_var") and row.get("observed_wind_var") != wind_var:
                continue
            if row.get("observed_basin_method") and row.get("observed_basin_method") != basin_method:
                continue

            threshold = float("nan")
            for column in source_columns:
                threshold = parse_float(row.get(column))
                if np.isfinite(threshold):
                    break
            if not np.isfinite(threshold):
                continue

            thresholds[basin_name] = float(threshold)
            loaded.add(basin_name)

    missing = [basin_name for basin_name in BASINS if basin_name not in loaded]
    source = str(path)
    if missing:
        print(
            "WARNING: threshold CSV missing finite values for "
            f"{', '.join(missing)}; using fallback {fallback_threshold:.2f} kt for those basins."
        )
    return thresholds, "basin", source


def print_thresholds(thresholds: dict[str, float], mode: str, source: str) -> None:
    print(f"TC Wind Threshold Mode: {mode}")
    print(f"TC Wind Threshold Source: {source}")
    for basin_name in BASINS:
        print(f"  {basin_name:18s} {thresholds[basin_name]:6.2f} kt")


def cache_threshold_signature(cache_path: Path) -> str | None:
    if not cache_path.exists():
        return None
    try:
        with netCDF4.Dataset(cache_path, "r") as ds:
            value = getattr(ds, "ts_threshold_signature", None)
            return str(value) if value is not None else None
    except Exception:
        return None


def to_datetime(value) -> datetime:
    return datetime(
        int(value.year),
        int(value.month),
        int(value.day),
        int(getattr(value, "hour", 0)),
        int(getattr(value, "minute", 0)),
        int(getattr(value, "second", 0)),
    )


def forecast_yyyymm(name: str, collection: str) -> str | None:
    collection_index = name.find(collection)
    search_text = name[collection_index + len(collection) :] if collection_index >= 0 else name
    match = re.search(r"(?:^|\.)(?:daily\.)?(\d{6})(?:\d{2})?(?:[_\.]|$)", search_text)
    if match:
        return match.group(1)
    return None


def discover_collection_files(root: Path, init_date: str, ens: str, collection: str) -> list[Path]:
    collection_dir = root / "GEOS_fcst" / init_date / ens / collection
    return sorted(collection_dir.glob("*.nc4"))


def find_first_variable(dataset, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in dataset.variables:
            return name
    return None


def find_sfc_wind_var_names(dataset) -> tuple[str, str] | None:
    u_var_name = find_first_variable(dataset, US_CANDIDATES)
    v_var_name = find_first_variable(dataset, VS_CANDIDATES)
    if u_var_name is None or v_var_name is None:
        return None
    return u_var_name, v_var_name


def coord_values_hpa(var) -> list[float] | None:
    try:
        values = np.asarray(var[:], dtype="float64").reshape(-1)
    except Exception:
        return None

    if values.size == 0:
        return None

    units = str(getattr(var, "units", "")).lower()
    if "pa" in units and "hpa" not in units and "mbar" not in units and "millibar" not in units:
        values = values / 100.0
    elif float(np.nanmax(np.abs(values))) > 2000.0:
        values = values / 100.0

    return values.tolist()


def detect_vertical_dim(dataset, required_hpa: tuple[float, ...]) -> tuple[str, dict[float, int]]:
    best_dim_name = None
    best_indices = None

    for dim_name in dataset.dimensions:
        if dim_name not in dataset.variables:
            continue

        coord = dataset.variables[dim_name]
        if len(coord.dimensions) != 1 or coord.dimensions[0] != dim_name:
            continue

        values_hpa = coord_values_hpa(coord)
        if not values_hpa:
            continue

        values = np.asarray(values_hpa, dtype="float64")
        level_indices: dict[float, int] = {}
        ok = True

        for target in required_hpa:
            differences = np.abs(values - target)
            index = int(np.nanargmin(differences))
            if float(differences[index]) > 1.0:
                ok = False
                break
            level_indices[target] = index

        if ok:
            best_dim_name = dim_name
            best_indices = level_indices
            break

    if best_dim_name is None or best_indices is None:
        required_text = ", ".join(f"{level:g}" for level in required_hpa)
        raise ValueError(f"Could not find a pressure coordinate containing {required_text} hPa")

    return best_dim_name, best_indices


def read_2d_field(var, time_index: int | None = None, level_dim: str | None = None, level_index: int | None = None) -> np.ndarray:
    slices: list[object] = []
    for dim_name in var.dimensions:
        if dim_name == "time":
            if time_index is None:
                raise ValueError(f"time_index is required for variable {var.name}")
            slices.append(time_index)
        elif level_dim is not None and dim_name == level_dim:
            if level_index is None:
                raise ValueError(f"level_index is required for variable {var.name}")
            slices.append(level_index)
        else:
            slices.append(slice(None))

    data = np.asarray(var[tuple(slices)], dtype="float64")
    if data.ndim != 2:
        raise ValueError(f"Expected 2D slice for {var.name}; got shape {data.shape}")
    return data


def slp_to_hpa(field: np.ndarray, units: str | None) -> np.ndarray:
    units_lower = (units or "").lower()
    if "pa" in units_lower and "hpa" not in units_lower and "mbar" not in units_lower:
        return field / 100.0
    if np.nanmax(np.abs(field)) > 2_000.0:
        return field / 100.0
    return field


def qv_to_gpkg(field: np.ndarray, units: str | None) -> np.ndarray:
    units_lower = (units or "").lower()
    if "g/kg" in units_lower or "g kg-1" in units_lower:
        return field
    if "kg/kg" in units_lower or units_lower in {"", "1", "kg kg-1"}:
        return field * 1_000.0
    return field


def index_radius(values: np.ndarray, radius_deg: float) -> int:
    if len(values) < 2:
        return 0
    spacing = float(np.nanmedian(np.abs(np.diff(values))))
    if spacing <= 0:
        return 0
    return max(1, int(round(radius_deg / spacing)))


def rectangular_patch(field: np.ndarray, lat_idx: int, lon_idx: int, lat_radius: int, lon_radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nlat, nlon = field.shape
    lat_lo = max(0, lat_idx - lat_radius)
    lat_hi = min(nlat, lat_idx + lat_radius + 1)
    lat_indices = np.arange(lat_lo, lat_hi)
    lon_offsets = np.arange(-lon_radius, lon_radius + 1)
    lon_indices = (lon_idx + lon_offsets) % nlon
    patch = field[np.ix_(lat_indices, lon_indices)]
    return patch, lat_indices, lon_offsets


def annulus_mean(field: np.ndarray, lat_idx: int, lon_idx: int, outer_lat_radius: int, outer_lon_radius: int, inner_lat_radius: int, inner_lon_radius: int) -> float:
    patch, lat_indices, lon_offsets = rectangular_patch(field, lat_idx, lon_idx, outer_lat_radius, outer_lon_radius)
    lat_offsets = lat_indices - lat_idx
    lat_grid, lon_grid = np.meshgrid(lat_offsets, lon_offsets, indexing="ij")
    ring_mask = (np.abs(lat_grid) > inner_lat_radius) | (np.abs(lon_grid) > inner_lon_radius)
    ring_values = patch[ring_mask]
    if ring_values.size == 0:
        return float(np.nanmean(patch))
    return float(np.nanmean(ring_values))


def max_near_center(field: np.ndarray, lat_idx: int, lon_idx: int, lat_radius: int, lon_radius: int) -> tuple[float, int, int]:
    patch, lat_indices, lon_offsets = rectangular_patch(field, lat_idx, lon_idx, lat_radius, lon_radius)
    if not np.isfinite(patch).any():
        return float("nan"), lat_idx, lon_idx
    flat_index = int(np.nanargmax(patch))
    patch_i, patch_j = np.unravel_index(flat_index, patch.shape)
    return float(patch[patch_i, patch_j]), int(lat_indices[patch_i]), int((lon_idx + lon_offsets[patch_j]) % field.shape[1])


def basin_lon_sets(longitudes: np.ndarray, basin_def: dict[str, object]) -> list[np.ndarray]:
    # Standardize longitudes to [-180, 180] range dynamically
    lons_shifted = (longitudes + 180) % 360 - 180
    if "lon_range" in basin_def:
        lon_min, lon_max = basin_def["lon_range"]
        return [np.where((lons_shifted >= lon_min) & (lons_shifted <= lon_max))[0]]

    lon_sets: list[np.ndarray] = []
    for lon_min, lon_max in basin_def["lon_ranges"]:
        lon_sets.append(np.where((lons_shifted >= lon_min) & (lons_shifted <= lon_max))[0])
    return lon_sets


def basin_candidate_center(field: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray, basin_def: dict[str, object]) -> tuple[int, int, float]:
    lat_min, lat_max = basin_def["lat_range"]
    lat_idx = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
    if len(lat_idx) == 0:
        raise ValueError("No latitude points found for basin")

    best_value = None
    best_indices = None
    for lon_idx in basin_lon_sets(longitudes, basin_def):
        if len(lon_idx) == 0:
            continue
        patch = field[np.ix_(lat_idx, lon_idx)]
        if not np.isfinite(patch).any():
            continue
        flat_index = int(np.nanargmin(patch))
        patch_i, patch_j = np.unravel_index(flat_index, patch.shape)
        candidate_value = float(patch[patch_i, patch_j])
        if best_value is None or candidate_value < best_value:
            best_value = candidate_value
            best_indices = (int(lat_idx[patch_i]), int(lon_idx[patch_j]))

    if best_value is None or best_indices is None:
        raise ValueError("No finite basin candidate center found")

    return best_indices[0], best_indices[1], best_value


def relative_vorticity(u_field: np.ndarray, v_field: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray, lat_idx: int, lon_idx: int) -> float:
    if lat_idx <= 0 or lat_idx >= len(latitudes) - 1:
        return float("nan")

    dlat_deg = float(abs(latitudes[lat_idx + 1] - latitudes[lat_idx - 1]))
    if dlat_deg == 0:
        return float("nan")

    dlon_deg = float(np.nanmedian(np.abs(np.diff(longitudes))))
    if dlon_deg == 0:
        return float("nan")

    lat_rad = math.radians(float(latitudes[lat_idx]))
    dx = EARTH_RADIUS_M * math.cos(lat_rad) * math.radians(dlon_deg * 2.0)
    dy = EARTH_RADIUS_M * math.radians(dlat_deg)
    if dx == 0 or dy == 0:
        return float("nan")

    jm1 = (lon_idx - 1) % len(longitudes)
    jp1 = (lon_idx + 1) % len(longitudes)
    du_dy = (u_field[lat_idx + 1, lon_idx] - u_field[lat_idx - 1, lon_idx]) / dy
    dv_dx = (v_field[lat_idx, jp1] - v_field[lat_idx, jm1]) / dx
    return float(dv_dx - du_dy)


class SFCIndex:
    def __init__(self, sfc_files: list[Path], collection: str, forecast_months: set[str] | None):
        self.entries: list[SFCMatch] = []
        self.latitudes: np.ndarray | None = None
        self.longitudes: np.ndarray | None = None
        self._cached_path: Path | None = None
        self._cached_ds = None
        self.indexed_files = 0
        self.skipped_no_wind_files = 0
        self.skipped_no_wind_examples: list[str] = []

        for path in sfc_files:
            yyyymm = forecast_yyyymm(path.name, collection)
            if forecast_months and (yyyymm is None or yyyymm[-2:] not in forecast_months):
                continue

            with netCDF4.Dataset(path, "r") as ds:
                if find_sfc_wind_var_names(ds) is None:
                    self.skipped_no_wind_files += 1
                    if len(self.skipped_no_wind_examples) < 5:
                        self.skipped_no_wind_examples.append(path.name)
                    continue

                if self.latitudes is None:
                    self.latitudes = np.asarray(ds.variables["lat"][:], dtype="float64")
                    self.longitudes = np.asarray(ds.variables["lon"][:], dtype="float64")
                time_values = ds.variables["time"][:]
                time_units = ds.variables["time"].units
                dates = netCDF4.num2date(time_values, time_units)
                for index, date_value in enumerate(dates):
                    self.entries.append(SFCMatch(valid_time=to_datetime(date_value), file_path=path, time_index=index))
                self.indexed_files += 1

        self.entries.sort(key=lambda item: item.valid_time)
        if self.latitudes is None or self.longitudes is None:
            raise ValueError(
                "No wind-bearing SFC files were indexed. "
                f"Skipped {self.skipped_no_wind_files} files without one of {US_CANDIDATES}/{VS_CANDIDATES}."
            )

    def close(self) -> None:
        if self._cached_ds is not None:
            self._cached_ds.close()
            self._cached_ds = None
            self._cached_path = None

    def nearest(self, valid_time: datetime, tolerance_hours: float) -> SFCMatch | None:
        best_match = None
        best_delta = None
        for entry in self.entries:
            delta_hours = abs((entry.valid_time - valid_time).total_seconds()) / 3600.0
            if best_delta is None or delta_hours < best_delta:
                best_match = entry
                best_delta = delta_hours

        if best_match is None or best_delta is None or best_delta > tolerance_hours:
            return None
        return best_match

    def load_wind(self, match: SFCMatch) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_path != match.file_path:
            self.close()
            self._cached_ds = netCDF4.Dataset(match.file_path, "r")
            self._cached_path = match.file_path

        wind_names = find_sfc_wind_var_names(self._cached_ds)
        if wind_names is None:
            available = ", ".join(list(self._cached_ds.variables)[:30])
            raise KeyError(
                f"SFC file {match.file_path} does not contain a recognized wind pair "
                f"{US_CANDIDATES}/{VS_CANDIDATES}. Available variables start with: {available}"
            )
        u_var_name, v_var_name = wind_names
        us = np.asarray(self._cached_ds.variables[u_var_name][match.time_index, :, :], dtype="float64")
        vs = np.asarray(self._cached_ds.variables[v_var_name][match.time_index, :, :], dtype="float64")
        return us, vs


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def basin_has_coverage(latitudes: np.ndarray, longitudes: np.ndarray, basin_def: dict[str, object]) -> bool:
    lat_min, lat_max = basin_def["lat_range"]
    lat_idx = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
    if len(lat_idx) == 0:
        return False

    for lon_idx in basin_lon_sets(longitudes, basin_def):
        if len(lon_idx) > 0:
            return True
    return False


def accumulate_storm_ace(
    local_ace: np.ndarray,
    sfc_ws_kt: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    center_lat_idx: int,
    center_lon_idx: int,
    lat_radius: int,
    lon_radius: int,
    search_radius_deg: float,
    ts_threshold_knots: float,
    scale_step: float,
) -> float:
    """Accumulate grid-point surface winds squared into local_ace near active storm center."""
    nlat, nlon = local_ace.shape

    # Focus loop inside the patch bounding box for high performance
    patch_indices_lat = np.arange(
        max(0, center_lat_idx - lat_radius),
        min(nlat, center_lat_idx + lat_radius + 1)
    )
    lon_offsets = np.arange(-lon_radius, lon_radius + 1)
    patch_indices_lon = (center_lon_idx + lon_offsets) % nlon

    center_lat_val = float(latitudes[center_lat_idx])
    center_lon_val = float((longitudes[center_lon_idx] + 180) % 360 - 180)
    lons_shifted = (longitudes + 180) % 360 - 180

    max_ws_kt = 0.0
    for lat_i in patch_indices_lat:
        lat_val = float(latitudes[lat_i])
        d_lat = lat_val - center_lat_val

        for lon_j in patch_indices_lon:
            lon_val = float(lons_shifted[lon_j])
            d_lon = (lon_val - center_lon_val + 180) % 360 - 180
            dist_deg = math.sqrt(d_lat**2 + d_lon**2)

            if dist_deg <= search_radius_deg:
                ws_kt = float(sfc_ws_kt[lat_i, lon_j])
                if ws_kt > max_ws_kt:
                    max_ws_kt = ws_kt

                # Accumulate spatial grid ACE if wind speed exceeds adjustments
                if ws_kt >= ts_threshold_knots:
                    grid_step_ace = (ws_kt**2) * scale_step
                    local_ace[lat_i, lon_j] += grid_step_ace

    return max_ws_kt


def write_cache(
    output_path: Path,
    init_date: str,
    ens: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    valid_times: list[datetime],
    local_ace: np.ndarray,
    diagnostics: dict[str, dict[str, list[float]]],
    uses_vorticity: bool,
    local_ace_monthly: dict[str, np.ndarray] | None = None,
    basin_thresholds: dict[str, float] | None = None,
    threshold_mode: str = "fixed",
    threshold_source: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        init_dt = datetime.strptime(init_date, "%Y%m%d")
    except ValueError:
        # Robust fallback for lagged ensembles (e.g. "202008_lagged") or other formats
        match = re.match(r"^(\d{6})", init_date)
        if match:
            init_dt = datetime.strptime(match.group(1) + "01", "%Y%m%d")
        else:
            if valid_times:
                init_dt = datetime(valid_times[0].year, valid_times[0].month, 1)
            else:
                init_dt = datetime(2020, 1, 1)
    
    time_hours = np.array([(value - init_dt).total_seconds() / 3600.0 for value in valid_times], dtype="float32")

    with netCDF4.Dataset(output_path, "w", format="NETCDF4") as ds:
        # Create dimensions
        ds.createDimension("lat", len(latitudes))
        ds.createDimension("lon", len(longitudes))
        ds.createDimension("time", len(valid_times))

        # Create coordinate variables
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        time_var = ds.createVariable("time", "f4", ("time",))

        # Write coordinates
        lat_var[:] = latitudes
        lon_var[:] = longitudes
        time_var[:] = time_hours

        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        time_var.units = f"hours since {init_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        time_var.long_name = "valid time"

        # Create compressed spatial local ACE field
        ace_spatial_var = ds.createVariable("local_ace", "f4", ("lat", "lon"), zlib=True, complevel=4)
        ace_spatial_var.units = "10^4 kt^2"
        ace_spatial_var.long_name = "Local Accumulated Cyclone Energy spatial field"
        ace_spatial_var[:] = local_ace

        # Save monthly local ACE fields if provided
        if local_ace_monthly is not None:
            for month_str, monthly_ace in local_ace_monthly.items():
                monthly_var = ds.createVariable(f"local_ace_{month_str}", "f4", ("lat", "lon"), zlib=True, complevel=4)
                monthly_var.units = "10^4 kt^2"
                monthly_var.long_name = f"Local Accumulated Cyclone Energy spatial field for month {month_str}"
                monthly_var[:] = monthly_ace

        # Backward compatible integrated curves variable (North Atlantic)
        ace_time_var = ds.createVariable("cumulative_ace", "f4", ("time",), zlib=True, complevel=4)
        ace_time_var.units = "10^4 kt^2"
        ace_time_var.long_name = "Basin-wide Cumulative ACE over time (North Atlantic)"
        ace_time_var[:] = np.array(diagnostics["North Atlantic"]["cumulative_ace"], dtype="float32")

        # Save all other multi-basin curves
        for basin_name, basin_data in diagnostics.items():
            basin_key = safe_name(basin_name)
            for field_name, values in basin_data.items():
                array = np.asarray(values)
                dtype = "i1" if array.dtype.kind in {"b", "i"} and field_name == "tc_flag" else "f4"
                var = ds.createVariable(f"{field_name}_{basin_key}", dtype, ("time",), zlib=True, complevel=4)
                var[:] = array
                if field_name == "cumulative_ace":
                    var.units = "10^4 kt^2"
                    var.long_name = f"Cumulative TC-conditioned ACE for {basin_name}"
                elif field_name == "step_ace":
                    var.units = "10^4 kt^2"
                    var.long_name = f"Step TC-conditioned ACE for {basin_name}"
                elif field_name == "vmax_kt":
                    var.units = "kt"
                elif field_name == "center_lat":
                    var.units = "degrees_north"
                elif field_name == "center_lon":
                    var.units = "degrees_east"
                elif field_name == "slp_hpa":
                    var.units = "hPa"
                elif field_name == "slp_anom_hpa":
                    var.units = "hPa"
                elif field_name == "warm_core_anom_k":
                    var.units = "K"
                elif field_name == "qv850_anom_gpkg":
                    var.units = "g kg-1"
                elif field_name == "vort850_s1":
                    var.units = "s-1"
                elif field_name == "ts_threshold_kt":
                    var.units = "kt"
                    var.long_name = f"ACE wind threshold for {basin_name}"

        ds.title = "TC-conditioned ACE diagnostics"
        ds.source_initialization = init_date
        ds.source_ensemble = ens
        ds.uses_vorticity = "true" if uses_vorticity else "false"
        if basin_thresholds is not None:
            ds.ts_threshold_mode = threshold_mode
            ds.ts_threshold_source = threshold_source
            ds.ts_threshold_signature = basin_threshold_signature(basin_thresholds)
            for basin_name, threshold in basin_thresholds.items():
                ds.setncattr(f"ts_threshold_kt_{safe_name(basin_name)}", float(threshold))
        ds.comment = (
            "Unified ACE proxy using SFC wind intensity gated by ATM structure: "
            "SLP minimum, warm-core anomaly, low-level moisture anomaly, and optional 850-hPa vorticity sign."
        )


def read_cache(cache_path: Path) -> tuple[str, str, np.ndarray, np.ndarray, list[datetime], np.ndarray, dict[str, dict[str, np.ndarray]], bool, dict[str, np.ndarray]]:
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    local_ace_monthly: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(cache_path, "r") as ds:
        init_date = str(getattr(ds, "source_initialization"))
        ens = str(getattr(ds, "source_ensemble"))
        uses_vorticity = str(getattr(ds, "uses_vorticity", "false")).lower() == "true"
        
        latitudes = np.asarray(ds.variables["lat"][:], dtype="float64")
        longitudes = np.asarray(ds.variables["lon"][:], dtype="float64")
        
        time_var = ds.variables["time"]
        times = [to_datetime(value) for value in netCDF4.num2date(time_var[:], time_var.units)]
        
        local_ace = np.asarray(ds.variables["local_ace"][:], dtype="float64")

        # Load monthly local ACE fields if they exist
        for var_name in ds.variables:
            if var_name.startswith("local_ace_"):
                month_str = var_name.split("_")[-1]
                local_ace_monthly[month_str] = np.asarray(ds.variables[var_name][:], dtype="float64")

        for basin_name in BASINS:
            basin_key = safe_name(basin_name)
            diagnostics[basin_name] = {}
            for field_name in (
                "cumulative_ace",
                "step_ace",
                "vmax_kt",
                "tc_flag",
                "center_lat",
                "center_lon",
                "slp_hpa",
                "slp_anom_hpa",
                "warm_core_anom_k",
                "qv850_anom_gpkg",
                "vort850_s1",
                "ts_threshold_kt",
            ):
                var_name = f"{field_name}_{basin_key}"
                if var_name in ds.variables:
                    diagnostics[basin_name][field_name] = np.asarray(ds.variables[var_name][:])
                else:
                    fill_value = float(getattr(ds, f"ts_threshold_kt_{basin_key}", np.nan)) if field_name == "ts_threshold_kt" else np.nan
                    diagnostics[basin_name][field_name] = np.full(len(times), fill_value, dtype="float64")

    return init_date, ens, latitudes, longitudes, times, local_ace, diagnostics, uses_vorticity, local_ace_monthly


def thresholds_from_diagnostics(diagnostics: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for basin_name in BASINS:
        values = np.asarray(diagnostics.get(basin_name, {}).get("ts_threshold_kt", []), dtype="float64")
        values = values[np.isfinite(values)]
        if values.size:
            thresholds[basin_name] = float(values[0])
    return thresholds


def plot_ace_diagnostics(
    local_ace: np.ndarray,
    cumulative_ace_time: np.ndarray,
    time_dates: list[datetime],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    init_date: str,
    ens: str,
    plot_dir: Path,
    basin_cumulative_ace: dict[str, np.ndarray] | None = None,
    local_ace_monthly: dict[str, np.ndarray] | None = None,
    basin_thresholds: dict[str, float] | None = None,
) -> None:
    """Generate and save premium, publication-quality diagnostic plots.

    1. A spatial Mercator projection map showing local TC-conditioned ACE tracks (NATL).
    2. A temporal line chart showing accumulation curves.
    3. A global Pacific-centered map showing all basins and integrated ACE.
    4. A two-panel plot showing Month 1 (September) and Month 2 (October) forecast.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    print("Generating unified diagnostic plots...")

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#4d924d", "#95d5b2", "#ffeb3b", "#ffa726", "#e65100", "#c2185b", "#ffffff"]
    ace_cmap = LinearSegmentedColormap.from_list("wmo_ace", colors, N=256)

    # Standardize longitudes to [-180, 180] for maps plotting
    lons_shifted = (longitudes + 180) % 360 - 180
    sorted_idx = np.argsort(lons_shifted)
    lons_plot = lons_shifted[sorted_idx]
    ace_plot = local_ace[:, sorted_idx]
    threshold_caption = "Basin-dependent GEOS thresholds" if basin_thresholds else "Fixed wind threshold"

    # --------------------------------------------------------------------------
    # PLOT 1: SPATIAL TC-CONDITIONED ACE MAP (NORTH ATLANTIC FOCUS)
    # --------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 7), dpi=300)
    
    if HAS_CARTOPY:
        projection = ccrs.Mercator(central_longitude=-55.0, min_latitude=0.0, max_latitude=45.0)
        ax = fig1.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent([-98.0, -15.0, 5.0, 42.0], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.OCEAN, facecolor="#daeefb", zorder=0)
        
        levels = np.linspace(0.005, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.005 else 5.0, 100)
        contour = ax.contourf(
            lons_plot, latitudes, ace_plot,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=ace_cmap,
            zorder=1,
            alpha=0.9
        )
        
        ax.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=2)
        ax.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.4, linestyle=":", zorder=3)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--", zorder=4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8, "color": "#555555"}
        gl.ylabel_style = {"size": 8, "color": "#555555"}
    else:
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_facecolor("#daeefb")
        levels = np.linspace(0.005, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.005 else 5.0, 100)
        contour = ax.contourf(longitudes, latitudes, local_ace, levels=levels, cmap=ace_cmap)
        ax.set_xlim([-98.0, -15.0])
        ax.set_ylim([5.0, 42.0])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--")

    cbar = fig1.colorbar(contour, ax=ax, orientation="horizontal", pad=0.08, aspect=40, shrink=0.8)
    cbar.set_label("TC-Conditioned Spatial ACE Index (10$^4$ kt$^2$)", fontsize=9, color="#333333", fontweight="bold", labelpad=6)
    cbar.ax.tick_params(labelsize=8, color="#555555", labelcolor="#333333")
    cbar.outline.set_visible(False)
    
    plt.title(
        f"GEOS S2S3 TC-Conditioned Local Accumulated Cyclone Energy (ACE) Map\n"
        f"Initialization: {init_date}  |  Member: {ens}  |  Season: Sep-Nov  |  {threshold_caption}",
        fontsize=12, fontweight="bold", pad=15, color="#1e222a"
    )
    
    spatial_plot_path = plot_dir / f"local_ace_map_{init_date}_{ens}.png"
    plt.savefig(spatial_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved spatial map to: {spatial_plot_path}")

    # --------------------------------------------------------------------------
    # PLOT 2: TEMPORAL CUMULATIVE ACE
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=300)
    ax2.plot(time_dates, cumulative_ace_time, color="#e55934", linewidth=2.5, label="Cumulative ACE")
    ax2.fill_between(time_dates, cumulative_ace_time, color="#e55934", alpha=0.1)
    
    ax2.set_facecolor("#fafafa")
    fig2.patch.set_facecolor("#ffffff")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#cccccc")
    ax2.spines["bottom"].set_color("#cccccc")
    
    ax2.set_ylabel("Basin-wide Cumulative ACE (10$^4$ kt$^2$)", fontsize=10, fontweight="bold", labelpad=8)
    ax2.set_xlabel("Forecast Lead Time (Date)", fontsize=10, fontweight="bold", labelpad=8)
    ax2.tick_params(axis="both", labelsize=9, colors="#555555")
    
    ax2.grid(True, linestyle="--", alpha=0.4, color="#cccccc", linewidth=0.5)
    fig2.autofmt_xdate()
    
    ax2.set_title(
        f"S2S3 North Atlantic Cumulative TC-Conditioned ACE Curve\n"
        f"Initialization: {init_date}  |  Ensemble: {ens}"
        + (
            f"  |  T={basin_thresholds['North Atlantic']:.1f} kt"
            if basin_thresholds and "North Atlantic" in basin_thresholds
            else ""
        ),
        fontsize=12, fontweight="bold", pad=15, color="#1e222a"
    )
    
    peak_ace = cumulative_ace_time[-1]
    ax2.annotate(
        f"Total ACE: {peak_ace:.2f}",
        xy=(time_dates[-1], peak_ace),
        xytext=(time_dates[-int(len(time_dates)*0.15)], peak_ace * 0.85),
        arrowprops=dict(facecolor="#333333", shrink=0.08, width=1, headwidth=6, headlength=6),
        fontsize=9, fontweight="bold", color="#e55934",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#e55934", lw=1, alpha=0.9)
    )

    temporal_plot_path = plot_dir / f"ace_accumulation_{init_date}_{ens}.png"
    plt.savefig(temporal_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved accumulation curve to: {temporal_plot_path}")

    # --------------------------------------------------------------------------
    # PLOT 3: GLOBAL MULTI-BASIN TC-CONDITIONED ACE MAP
    # --------------------------------------------------------------------------
    fig3 = plt.figure(figsize=(14, 8), dpi=300)
    
    if HAS_CARTOPY:
        projection = ccrs.PlateCarree(central_longitude=180.0)
        ax3 = fig3.add_subplot(1, 1, 1, projection=projection)
        ax3.set_global()
        ax3.add_feature(cfeature.OCEAN, facecolor="#daeefb", zorder=0)
        
        levels = np.linspace(0.005, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.005 else 5.0, 100)
        contour3 = ax3.contourf(
            lons_plot, latitudes, ace_plot,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=ace_cmap,
            zorder=1,
            alpha=0.9
        )
        
        ax3.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=2)
        ax3.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.5, zorder=3)
        ax3.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.3, linestyle=":", zorder=3)
        
        gl3 = ax3.gridlines(draw_labels=True, linewidth=0.2, color="#aaaaaa", alpha=0.4, linestyle="--", zorder=4)
        gl3.top_labels = False
        gl3.right_labels = False
        gl3.xlabel_style = {"size": 8, "color": "#555555"}
        gl3.ylabel_style = {"size": 8, "color": "#555555"}
    else:
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.set_facecolor("#daeefb")
        levels = np.linspace(0.005, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.005 else 5.0, 100)
        contour3 = ax3.contourf(longitudes, latitudes, local_ace, levels=levels, cmap=ace_cmap)
        ax3.set_xlim([-180.0, 180.0])
        ax3.set_ylim([-60.0, 60.0])
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Latitude")
        ax3.grid(True, linewidth=0.2, color="#aaaaaa", alpha=0.4, linestyle="--")

    for name, b_def in BASINS.items():
        color = b_def["color"]
        lat_min, lat_max = b_def["lat_range"]
        
        total_ace = 0.0
        if basin_cumulative_ace and name in basin_cumulative_ace:
            total_ace = float(basin_cumulative_ace[name][-1])
            
        if "lon_range" in b_def:
            lon_min, lon_max = b_def["lon_range"]
            lons_rect = [lon_min, lon_max, lon_max, lon_min, lon_min]
            lats_rect = [lat_min, lat_min, lat_max, lat_max, lat_min]
            if HAS_CARTOPY:
                ax3.plot(lons_rect, lats_rect, color=color, linewidth=1.8, linestyle="-", transform=ccrs.PlateCarree(), zorder=5)
            else:
                ax3.plot(lons_rect, lats_rect, color=color, linewidth=1.8, linestyle="-", zorder=5)
        else:
            for lon_min, lon_max in b_def["lon_ranges"]:
                lons_rect = [lon_min, lon_max, lon_max, lon_min, lon_min]
                lats_rect = [lat_min, lat_min, lat_max, lat_max, lat_min]
                if HAS_CARTOPY:
                    ax3.plot(lons_rect, lats_rect, color=color, linewidth=1.8, linestyle="-", transform=ccrs.PlateCarree(), zorder=5)
                else:
                    ax3.plot(lons_rect, lats_rect, color=color, linewidth=1.8, linestyle="-", zorder=5)
                    
        label_lon, label_lat = b_def["label_xy"]
        bbox_props = dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=color, lw=1.2, alpha=0.9)
        
        if HAS_CARTOPY:
            threshold_text = f"\nT: {basin_thresholds[name]:.1f} kt" if basin_thresholds and name in basin_thresholds else ""
            ax3.text(
                label_lon, label_lat, f"{name}\nACE: {total_ace:.2f}{threshold_text}",
                transform=ccrs.PlateCarree(),
                color="#1e222a", fontsize=8, fontweight="bold",
                ha="center", va="center", bbox=bbox_props, zorder=6
            )
        else:
            threshold_text = f"\nT: {basin_thresholds[name]:.1f} kt" if basin_thresholds and name in basin_thresholds else ""
            ax3.text(
                label_lon, label_lat, f"{name}\nACE: {total_ace:.2f}{threshold_text}",
                color="#1e222a", fontsize=8, fontweight="bold",
                ha="center", va="center", bbox=bbox_props, zorder=6
            )

    cbar3 = fig3.colorbar(contour3, ax=ax3, orientation="horizontal", pad=0.08, aspect=45, shrink=0.75)
    cbar3.set_label("TC-Conditioned Spatial ACE Index (10$^4$ kt$^2$)", fontsize=9, color="#333333", fontweight="bold", labelpad=6)
    cbar3.ax.tick_params(labelsize=8, color="#555555", labelcolor="#333333")
    cbar3.outline.set_visible(False)
    
    plt.title(
        f"GEOS S2S3 Global Multi-Basin TC-Conditioned Accumulated Cyclone Energy (ACE) Map\n"
        f"Initialization: {init_date}  |  Ensemble: {ens}  |  Season: Sep-Nov  |  {threshold_caption}",
        fontsize=13, fontweight="bold", pad=15, color="#1e222a"
    )
    
    global_plot_path = plot_dir / f"global_ace_map_{init_date}_{ens}.png"
    plt.savefig(global_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved global multi-basin map to: {global_plot_path}\n")

    # --------------------------------------------------------------------------
    # PLOT 4: TWO-PANEL MONTHLY COMPARISON (SEPTEMBER & OCTOBER)
    # --------------------------------------------------------------------------
    if local_ace_monthly:
        print("Generating monthly comparison plots...")
        ace_sep = local_ace_monthly.get("09", np.zeros_like(local_ace))
        ace_oct = local_ace_monthly.get("10", np.zeros_like(local_ace))

        ace_plot_sep = ace_sep[:, sorted_idx]
        ace_plot_oct = ace_oct[:, sorted_idx]

        fig4 = plt.figure(figsize=(16, 8), dpi=300)
        
        max_ace_val = max(np.max(ace_sep), np.max(ace_oct))
        if max_ace_val < 0.005:
            max_ace_val = 5.0
        levels = np.linspace(0.005, max_ace_val * 1.05, 100)

        # September Panel
        if HAS_CARTOPY:
            proj = ccrs.Mercator(central_longitude=-55.0, min_latitude=0.0, max_latitude=45.0)
            ax_sep = fig4.add_subplot(1, 2, 1, projection=proj)
            ax_sep.set_extent([-98.0, -15.0, 5.0, 42.0], crs=ccrs.PlateCarree())
            ax_sep.add_feature(cfeature.OCEAN, facecolor="#daeefb", zorder=0)
            
            contour_sep = ax_sep.contourf(
                lons_plot, latitudes, ace_plot_sep,
                levels=levels,
                transform=ccrs.PlateCarree(),
                cmap=ace_cmap,
                zorder=1,
                alpha=0.9
            )
            ax_sep.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=2)
            ax_sep.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.6, zorder=3)
            ax_sep.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.4, linestyle=":", zorder=3)
            
            gl_s = ax_sep.gridlines(draw_labels=True, linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--", zorder=4)
            gl_s.top_labels = False
            gl_s.right_labels = False
            gl_s.xlabel_style = {"size": 8, "color": "#555555"}
            gl_s.ylabel_style = {"size": 8, "color": "#555555"}
        else:
            ax_sep = fig4.add_subplot(1, 2, 1)
            ax_sep.set_facecolor("#daeefb")
            contour_sep = ax_sep.contourf(longitudes, latitudes, ace_sep, levels=levels, cmap=ace_cmap)
            ax_sep.set_xlim([-98.0, -15.0])
            ax_sep.set_ylim([5.0, 42.0])
            ax_sep.set_xlabel("Longitude")
            ax_sep.set_ylabel("Latitude")
            ax_sep.grid(True, linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--")

        ax_sep.set_title("September - Month 1 Forecast", fontsize=11, fontweight="bold", pad=10, color="#1e222a")

        # October Panel
        if HAS_CARTOPY:
            ax_oct = fig4.add_subplot(1, 2, 2, projection=proj)
            ax_oct.set_extent([-98.0, -15.0, 5.0, 42.0], crs=ccrs.PlateCarree())
            ax_oct.add_feature(cfeature.OCEAN, facecolor="#daeefb", zorder=0)
            
            contour_oct = ax_oct.contourf(
                lons_plot, latitudes, ace_plot_oct,
                levels=levels,
                transform=ccrs.PlateCarree(),
                cmap=ace_cmap,
                zorder=1,
                alpha=0.9
            )
            ax_oct.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=2)
            ax_oct.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.6, zorder=3)
            ax_oct.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.4, linestyle=":", zorder=3)
            
            gl_o = ax_oct.gridlines(draw_labels=True, linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--", zorder=4)
            gl_o.top_labels = False
            gl_o.right_labels = False
            gl_o.xlabel_style = {"size": 8, "color": "#555555"}
            gl_o.ylabel_style = {"size": 8, "color": "#555555"}
        else:
            ax_oct = fig4.add_subplot(1, 2, 2)
            ax_oct.set_facecolor("#daeefb")
            contour_oct = ax_oct.contourf(longitudes, latitudes, ace_oct, levels=levels, cmap=ace_cmap)
            ax_oct.set_xlim([-98.0, -15.0])
            ax_oct.set_ylim([5.0, 42.0])
            ax_oct.set_xlabel("Longitude")
            ax_oct.set_ylabel("Latitude")
            ax_oct.grid(True, linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--")

        ax_oct.set_title("October - Month 2 Forecast", fontsize=11, fontweight="bold", pad=10, color="#1e222a")

        # Shared colorbar at the bottom
        cbar_ax = fig4.add_axes([0.25, 0.08, 0.5, 0.03])
        cbar4 = fig4.colorbar(contour_sep, cax=cbar_ax, orientation="horizontal")
        cbar4.set_label("TC-Conditioned Spatial ACE Index (10$^4$ kt$^2$)", fontsize=9, color="#333333", fontweight="bold", labelpad=6)
        cbar4.ax.tick_params(labelsize=8, color="#555555", labelcolor="#333333")
        cbar4.outline.set_visible(False)

        # Super title
        fig4.suptitle(
            f"GEOS S2S3 TC-Conditioned Local ACE Monthly Comparison (North Atlantic)\n"
            f"Initialization: {init_date}  |  Member: {ens}"
            + (
                f"  |  T={basin_thresholds['North Atlantic']:.1f} kt"
                if basin_thresholds and "North Atlantic" in basin_thresholds
                else ""
            ),
            fontsize=13, fontweight="bold", y=0.96, color="#1e222a"
        )
        
        monthly_plot_path = plot_dir / f"monthly_local_ace_comparison_{init_date}_{ens}.png"
        plt.savefig(monthly_plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  -> Saved monthly comparison map to: {monthly_plot_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfc-root", default=os.environ.get("SFC_ROOT", DEFAULT_SFC_ROOT))
    parser.add_argument("--atm-root", default=os.environ.get("ATM_ROOT", DEFAULT_ATM_ROOT))
    parser.add_argument("--init-date", required=True, help="Forecast initialization date, e.g. 20200824")
    parser.add_argument(
        "--ens",
        default="all",
        help="Ensemble member directory name, comma-separated list, or 'all' to process all discovered members (default: all)",
    )
    parser.add_argument("--sfc-collection", default=DEFAULT_SFC_COLLECTION)
    parser.add_argument("--atm-collection", default=DEFAULT_ATM_COLLECTION)
    parser.add_argument("--months", default="09,10", help="Forecast months to use, separated by commas")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--plot-only-cache",
        default=None,
        help="Read an existing NetCDF cache file and create plots without reprocessing SFC/ATM data.",
    )
    parser.add_argument(
        "--sfc-match-tolerance-hours",
        type=float,
        default=3.1,
        help="Maximum allowed time mismatch between ATM and SFC valid times",
    )
    parser.add_argument(
        "--environment-radius-deg",
        type=float,
        default=5.0,
        help="Radius used to define the local background environment",
    )
    parser.add_argument(
        "--inner-core-radius-deg",
        type=float,
        default=1.5,
        help="Inner-core radius excluded from the environment mean",
    )
    parser.add_argument(
        "--wind-search-radius-deg",
        type=float,
        default=3.0,
        help="Radius around the candidate center used to search for max SFC wind and accumulate spatial ACE",
    )
    parser.add_argument(
        "--ts-threshold",
        type=float,
        default=TS_THRESHOLD_KNOTS_DEFAULT,
        help=(
            f"Wind speed threshold (knots) for ACE accumulation (default: {TS_THRESHOLD_KNOTS_DEFAULT}). "
            "GEOS 0.5deg models underestimate TC intensity; the model-equivalent of the "
            "observed 34-kt TS threshold is ~17-24 kt (Garcia-Franco et al. 2024). "
            "Use 34.0 for observational/reanalysis data."
        )
    )
    parser.add_argument(
        "--geos-thresholds",
        default="",
        help="Optional basin-dependent GEOS threshold CSV from calculate_geos_candidate_thresholds.py.",
    )
    parser.add_argument(
        "--threshold-wind-var",
        default="usa_wind",
        help="Observed wind variable selector for threshold CSV rows. Default: usa_wind.",
    )
    parser.add_argument(
        "--threshold-basin-method",
        default="boxes",
        help="Observed basin method selector for threshold CSV rows. Default: boxes.",
    )
    parser.add_argument(
        "--threshold-column",
        default="auto",
        help="Threshold column to read. auto tries geos_threshold_kt, count_match_threshold_kt, then T_geos.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute ACE caches even if matching cache files already exist.",
    )
    parser.add_argument(
        "--plot-individual",
        action="store_true",
        help="Generate visual diagnostic plots for each individual ensemble member (default: False)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plot_dir = Path(args.plot_dir)
    threshold_csv = Path(args.geos_thresholds) if args.geos_thresholds else None
    basin_thresholds, threshold_mode, threshold_source = load_basin_thresholds(
        threshold_csv,
        fallback_threshold=args.ts_threshold,
        wind_var=args.threshold_wind_var,
        basin_method=args.threshold_basin_method,
        threshold_column=args.threshold_column,
    )
    threshold_signature = basin_threshold_signature(basin_thresholds)
    
    if args.plot_only_cache:
        init_date, ens, latitudes, longitudes, times, local_ace, diagnostics, uses_vorticity, local_ace_monthly = read_cache(Path(args.plot_only_cache))
        cache_thresholds = thresholds_from_diagnostics(diagnostics)
        if threshold_csv is not None and not cache_thresholds:
            print(
                "ERROR: --plot-only-cache has no threshold metadata. "
                "Rerun without --plot-only-cache so ACE can be rebuilt with --geos-thresholds.",
                file=sys.stderr,
            )
            return 1
        if threshold_csv is not None and basin_threshold_signature(basin_thresholds) != basin_threshold_signature(cache_thresholds):
            print(
                "ERROR: --plot-only-cache cannot recalculate ACE with new thresholds. "
                "Rerun without --plot-only-cache so the cache can be rebuilt.",
                file=sys.stderr,
            )
            return 1
        if not cache_thresholds:
            print("WARNING: cache has no basin threshold metadata; plot labels will not show calibrated thresholds.")
        basin_cumulative_ace = {basin_name: data["cumulative_ace"] for basin_name, data in diagnostics.items()}
        plot_ace_diagnostics(
            local_ace=local_ace,
            cumulative_ace_time=diagnostics["North Atlantic"]["cumulative_ace"],
            time_dates=times,
            latitudes=latitudes,
            longitudes=longitudes,
            init_date=init_date,
            ens=ens,
            plot_dir=plot_dir,
            basin_cumulative_ace=basin_cumulative_ace,
            local_ace_monthly=local_ace_monthly,
            basin_thresholds=cache_thresholds or None,
        )
        return 0

    forecast_months = set(parse_list(args.months))

    sfc_root = Path(args.sfc_root)
    atm_root = Path(args.atm_root)
    cache_dir = Path(args.cache_dir)

    # 1. Determine list of ensemble members to process
    ens_input = args.ens.strip()
    if ens_input.lower() == "all":
        sfc_fcst_dir = sfc_root / "GEOS_fcst" / args.init_date
        if not sfc_fcst_dir.is_dir():
            print(f"ERROR: Initialization directory {sfc_fcst_dir} does not exist", file=sys.stderr)
            return 1
        ens_list = sorted([d.name for d in sfc_fcst_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        if not ens_list:
            print(f"ERROR: No ensemble directories found in {sfc_fcst_dir}", file=sys.stderr)
            return 1
    else:
        ens_list = [e.strip() for e in ens_input.split(",") if e.strip()]

    print("=" * 80)
    print("GEOS S2S3 UNIFIED TC-CONDITIONED ACE DIAGNOSTICS")
    print(f"Initialization: {args.init_date}")
    print(f"Target Ensembles: {', '.join(ens_list)}")
    print_thresholds(basin_thresholds, threshold_mode, threshold_source)
    print("=" * 80)

    processed_count = 0
    all_member_results = []

    for ens_member in ens_list:
        print("\n" + "-" * 80)
        print(f"PROCESSING ENSEMBLE MEMBER: {ens_member}")
        print("-" * 80)

        # Check if cache file already exists and load it if it does
        member_cache_path = cache_dir / f"tc_conditioned_ace_{args.init_date}_{ens_member}.nc4"
        cache_matches = False
        if member_cache_path.is_file() and not args.force_recompute:
            cache_signature = cache_threshold_signature(member_cache_path)
            cache_matches = cache_signature == threshold_signature
            if cache_signature is None and threshold_mode == "fixed" and args.ts_threshold == TS_THRESHOLD_KNOTS_DEFAULT:
                cache_matches = True
            if not cache_matches:
                print(f"[Cache] Existing cache threshold signature differs for '{ens_member}'; recomputing {member_cache_path.name}")
            else:
                print(f"[Cache] Found matching cache for '{ens_member}': {member_cache_path.name}")
        elif member_cache_path.is_file() and args.force_recompute:
            print(f"[Cache] Forcing recompute for '{ens_member}': {member_cache_path.name}")
        if member_cache_path.is_file() and cache_matches:
            try:
                c_init, c_ens, c_lats, c_lons, c_times, c_local_ace, c_diag, c_vort, c_monthly = read_cache(member_cache_path)
                cache_thresholds = thresholds_from_diagnostics(c_diag)
                if args.plot_individual:
                    print(f"[Cache] Successfully loaded cached diagnostics. Regenerating plots to ensure completeness...")
                    # Regenerate plots in case styling updated
                    basin_cumulative_ace = {name: np.asarray(data["cumulative_ace"]) for name, data in c_diag.items()}
                    plot_ace_diagnostics(
                        local_ace=c_local_ace,
                        cumulative_ace_time=basin_cumulative_ace["North Atlantic"],
                        time_dates=c_times,
                        latitudes=c_lats,
                        longitudes=c_lons,
                        init_date=c_init,
                        ens=c_ens,
                        plot_dir=plot_dir,
                        basin_cumulative_ace=basin_cumulative_ace,
                        local_ace_monthly=c_monthly,
                        basin_thresholds=cache_thresholds or basin_thresholds,
                    )
                else:
                    print(f"[Cache] Successfully loaded cached diagnostics for '{ens_member}'. Skipping individual plots.")
                
                # Make sure the diagnostics dict values are np.asarray for ensmean consistency
                np_diagnostics = {}
                for b_name, b_data in c_diag.items():
                    np_diagnostics[b_name] = {k: np.asarray(v) for k, v in b_data.items()}
                
                all_member_results.append((ens_member, c_lats, c_lons, c_times, c_local_ace, np_diagnostics, c_vort, c_monthly))
                processed_count += 1
                print(f"[Cache] Completed member '{ens_member}' using cached diagnostics.")
                continue
            except Exception as e:
                print(f"[Cache Warning] Failed to read existing cache '{member_cache_path.name}': {e}. Reverting to raw data processing.")

        sfc_files = discover_collection_files(sfc_root, args.init_date, ens_member, args.sfc_collection)
        atm_files = discover_collection_files(atm_root, args.init_date, ens_member, args.atm_collection)

        if not sfc_files:
            print(f"WARNING: Skipping member '{ens_member}' as no SFC files were found under {sfc_root}.")
            continue
        if not atm_files:
            print(f"WARNING: Skipping member '{ens_member}' as no ATM files were found under {atm_root}.")
            continue

        try:
            sfc_index = SFCIndex(sfc_files, args.sfc_collection, forecast_months)
        except Exception as e:
            print(f"WARNING: Skipping member '{ens_member}' due to indexing failure: {e}")
            continue

        try:
            latitudes = sfc_index.latitudes
            longitudes = sfc_index.longitudes
            if latitudes is None or longitudes is None:
                print(f"WARNING: Skipping member '{ens_member}' due to missing latitude/longitude coordinates.")
                continue
            
            basin_coverage = {name: basin_has_coverage(latitudes, longitudes, basin_def) for name, basin_def in BASINS.items()}

            # Compute coordinate indices for patch operations
            lat_env_radius = index_radius(latitudes, args.environment_radius_deg)
            lon_env_radius = index_radius(longitudes, args.environment_radius_deg)
            lat_core_radius = index_radius(latitudes, args.inner_core_radius_deg)
            lon_core_radius = index_radius(longitudes, args.inner_core_radius_deg)
            lat_wind_radius = index_radius(latitudes, args.wind_search_radius_deg)
            lon_wind_radius = index_radius(longitudes, args.wind_search_radius_deg)

            # Initialize 2D spatial local ACE field
            nlat_g, nlon_g = len(latitudes), len(longitudes)
            local_ace = np.zeros((nlat_g, nlon_g), dtype="float64")
            local_ace_monthly = {
                m: np.zeros((nlat_g, nlon_g), dtype="float64")
                for m in forecast_months
            }

            diagnostics: dict[str, dict[str, list[float]]] = {
                basin_name: {
                    "cumulative_ace": [],
                    "step_ace": [],
                    "vmax_kt": [],
                    "tc_flag": [],
                    "center_lat": [],
                    "center_lon": [],
                    "slp_hpa": [],
                    "slp_anom_hpa": [],
                    "warm_core_anom_k": [],
                    "qv850_anom_gpkg": [],
                    "vort850_s1": [],
                    "ts_threshold_kt": [],
                }
                for basin_name in BASINS
            }
            basin_totals = {basin_name: 0.0 for basin_name in BASINS}
            valid_times: list[datetime] = []
            uses_vorticity = False

            print(f"SFC files: {len(sfc_files)} | ATM files: {len(atm_files)}")

            for atm_path in atm_files:
                yyyymm = forecast_yyyymm(atm_path.name, args.atm_collection)
                if forecast_months and (yyyymm is None or yyyymm[-2:] not in forecast_months):
                    continue

                print(f"  Reading ATM file: {atm_path.name}")
                with netCDF4.Dataset(atm_path, "r") as ds:
                    slp_name = find_first_variable(ds, SLP_CANDIDATES)
                    t_name = find_first_variable(ds, T_CANDIDATES)
                    qv_name = find_first_variable(ds, QV_CANDIDATES)
                    u_name = find_first_variable(ds, U_CANDIDATES)
                    v_name = find_first_variable(ds, V_CANDIDATES)
                    
                    if slp_name is None or t_name is None or qv_name is None:
                        raise ValueError(
                            f"ATM file {atm_path.name} is missing one of the required variables: "
                            f"SLP={slp_name}, T={t_name}, QV={qv_name}"
                        )

                    level_dim, level_indices = detect_vertical_dim(ds, (850.0, 500.0, 200.0))
                    time_values = ds.variables["time"][:]
                    time_units = ds.variables["time"].units
                    atm_times = [to_datetime(value) for value in netCDF4.num2date(time_values, time_units)]

                    # Compute dynamic sampling interval for correct scaling
                    time_diff_hours = 6.0
                    if len(atm_times) > 1:
                        time_diff_hours = abs((atm_times[1] - atm_times[0]).total_seconds()) / 3600.0
                    scale_step = 1e-4 * (time_diff_hours / 6.0)

                    for time_index, valid_time in enumerate(atm_times):
                        if forecast_months and valid_time.strftime("%m") not in forecast_months:
                            continue

                        sfc_match = sfc_index.nearest(valid_time, args.sfc_match_tolerance_hours)
                        if sfc_match is None:
                            print(f"    Skipping {valid_time}: no matching SFC time within tolerance")
                            continue

                        us_sfc, vs_sfc = sfc_index.load_wind(sfc_match)
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
                        if u_name is not None and v_name is not None:
                            u850 = read_2d_field(ds.variables[u_name], time_index=time_index, level_dim=level_dim, level_index=level_indices[850.0])
                            v850 = read_2d_field(ds.variables[v_name], time_index=time_index, level_dim=level_dim, level_index=level_indices[850.0])
                            uses_vorticity = True

                        valid_times.append(valid_time)

                        for basin_name, basin_def in BASINS.items():
                            basin_ts_threshold = float(basin_thresholds[basin_name])
                            if not basin_coverage[basin_name]:
                                basin_diag = diagnostics[basin_name]
                                basin_diag["cumulative_ace"].append(basin_totals[basin_name])
                                basin_diag["step_ace"].append(0.0)
                                basin_diag["vmax_kt"].append(float("nan"))
                                basin_diag["tc_flag"].append(0)
                                basin_diag["center_lat"].append(float("nan"))
                                basin_diag["center_lon"].append(float("nan"))
                                basin_diag["slp_hpa"].append(float("nan"))
                                basin_diag["slp_anom_hpa"].append(float("nan"))
                                basin_diag["warm_core_anom_k"].append(float("nan"))
                                basin_diag["qv850_anom_gpkg"].append(float("nan"))
                                basin_diag["vort850_s1"].append(float("nan"))
                                basin_diag["ts_threshold_kt"].append(basin_ts_threshold)
                                continue

                            # Robust candidate evaluation with exception handling to prevent crashes
                            try:
                                center_lat_idx, center_lon_idx, center_slp = basin_candidate_center(slp_hpa, latitudes, longitudes, basin_def)
                                center_lat = float(latitudes[center_lat_idx])
                                # Map saved center longitude to [-180, 180]
                                center_lon = float((longitudes[center_lon_idx] + 180) % 360 - 180)
                            except ValueError:
                                # Safely ignore and fill step with NaNs
                                basin_diag = diagnostics[basin_name]
                                basin_diag["cumulative_ace"].append(basin_totals[basin_name])
                                basin_diag["step_ace"].append(0.0)
                                basin_diag["vmax_kt"].append(float("nan"))
                                basin_diag["tc_flag"].append(0)
                                basin_diag["center_lat"].append(float("nan"))
                                basin_diag["center_lon"].append(float("nan"))
                                basin_diag["slp_hpa"].append(float("nan"))
                                basin_diag["slp_anom_hpa"].append(float("nan"))
                                basin_diag["warm_core_anom_k"].append(float("nan"))
                                basin_diag["qv850_anom_gpkg"].append(float("nan"))
                                basin_diag["vort850_s1"].append(float("nan"))
                                basin_diag["ts_threshold_kt"].append(basin_ts_threshold)
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

                            slp_anom = center_slp - slp_env
                            warm_anom = float(warm_core_field[center_lat_idx, center_lon_idx] - warm_env)
                            qv_anom = float(qv850[center_lat_idx, center_lon_idx] - qv_env)

                            criteria = [
                                slp_anom < 0.0,
                                warm_anom > 0.0,
                                qv_anom > 0.0,
                            ]

                            vort850 = float("nan")
                            if u850 is not None and v850 is not None:
                                vort850 = relative_vorticity(u850, v850, latitudes, longitudes, center_lat_idx, center_lon_idx)
                                if np.isfinite(vort850):
                                    criteria.append(vort850 > 0.0 if center_lat >= 0.0 else vort850 < 0.0)

                            tc_flag = int(all(criteria))
                            
                            vmax_kt = float("nan")
                            step_ace = 0.0
                            if tc_flag:
                                # Accumulate grid-point winds into spatial local_ace field near approved storm
                                vmax_kt = accumulate_storm_ace(
                                    local_ace=local_ace,
                                    sfc_ws_kt=sfc_ws_kt,
                                    latitudes=latitudes,
                                    longitudes=longitudes,
                                    center_lat_idx=center_lat_idx,
                                    center_lon_idx=center_lon_idx,
                                    lat_radius=lat_wind_radius,
                                    lon_radius=lon_wind_radius,
                                    search_radius_deg=args.wind_search_radius_deg,
                                    ts_threshold_knots=basin_ts_threshold,
                                    scale_step=scale_step,
                                )
                                
                                # Also accumulate into monthly field
                                current_month = valid_time.strftime("%m")
                                if current_month in local_ace_monthly:
                                    _ = accumulate_storm_ace(
                                        local_ace=local_ace_monthly[current_month],
                                        sfc_ws_kt=sfc_ws_kt,
                                        latitudes=latitudes,
                                        longitudes=longitudes,
                                        center_lat_idx=center_lat_idx,
                                        center_lon_idx=center_lon_idx,
                                        lat_radius=lat_wind_radius,
                                        lon_radius=lon_wind_radius,
                                        search_radius_deg=args.wind_search_radius_deg,
                                        ts_threshold_knots=basin_ts_threshold,
                                        scale_step=scale_step,
                                    )
                                
                                # Increment integrated temporal curves using vmax within storm radius
                                if np.isfinite(vmax_kt) and vmax_kt >= basin_ts_threshold:
                                    step_ace = float(vmax_kt**2 * scale_step)
                            else:
                                # Evaluated peak wind for metadata diagnostics, even if rejected by structure gate
                                raw_vmax, _, _ = max_near_center(
                                    sfc_ws_kt,
                                    center_lat_idx,
                                    center_lon_idx,
                                    lat_wind_radius,
                                    lon_wind_radius,
                                )
                                vmax_kt = float(raw_vmax)

                            basin_totals[basin_name] += step_ace

                            basin_diag = diagnostics[basin_name]
                            basin_diag["cumulative_ace"].append(basin_totals[basin_name])
                            basin_diag["step_ace"].append(step_ace)
                            basin_diag["vmax_kt"].append(vmax_kt)
                            basin_diag["tc_flag"].append(tc_flag)
                            basin_diag["center_lat"].append(center_lat)
                            basin_diag["center_lon"].append(center_lon)
                            basin_diag["slp_hpa"].append(float(center_slp))
                            basin_diag["slp_anom_hpa"].append(float(slp_anom))
                            basin_diag["warm_core_anom_k"].append(float(warm_anom))
                            basin_diag["qv850_anom_gpkg"].append(float(qv_anom))
                            basin_diag["vort850_s1"].append(float(vort850))
                            basin_diag["ts_threshold_kt"].append(basin_ts_threshold)

            if not valid_times:
                print(f"WARNING: No valid ATM/SFC time matches were processed for '{ens_member}'. Skipping cache/plot generation.")
                continue

            output_path = cache_dir / f"tc_conditioned_ace_{args.init_date}_{ens_member}.nc4"
            write_cache(
                output_path,
                args.init_date,
                ens_member,
                latitudes,
                longitudes,
                valid_times,
                local_ace,
                diagnostics,
                uses_vorticity,
                local_ace_monthly,
                basin_thresholds=basin_thresholds,
                threshold_mode=threshold_mode,
                threshold_source=threshold_source,
            )
            
            # Load curves to plot
            basin_cumulative_ace = {name: np.array(data["cumulative_ace"]) for name, data in diagnostics.items()}
            if args.plot_individual:
                plot_ace_diagnostics(
                    local_ace=local_ace,
                    cumulative_ace_time=basin_cumulative_ace["North Atlantic"],
                    time_dates=valid_times,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    init_date=args.init_date,
                    ens=ens_member,
                    plot_dir=plot_dir,
                    basin_cumulative_ace=basin_cumulative_ace,
                    local_ace_monthly=local_ace_monthly,
                    basin_thresholds=basin_thresholds,
                )

            # Convert diagnostics to np.asarray values for consistency with c_diag loaded from NetCDF
            np_diagnostics = {}
            for b_name, b_data in diagnostics.items():
                np_diagnostics[b_name] = {k: np.asarray(v) for k, v in b_data.items()}

            all_member_results.append((ens_member, latitudes, longitudes, valid_times, local_ace, np_diagnostics, uses_vorticity, local_ace_monthly))

            print("=" * 80)
            print(f"Wrote cache: {output_path}")
            for basin_name in BASINS:
                total_ace = diagnostics[basin_name]["cumulative_ace"][-1]
                hits = int(np.sum(diagnostics[basin_name]["tc_flag"]))
                print(f"{basin_name:18s} total_ace={total_ace:8.2f}  tc_hits={hits}")
            print("=" * 80)
            processed_count += 1

        except Exception as e:
            print(f"ERROR: Failed processing ensemble member '{ens_member}': {e}")
            import traceback
            traceback.print_exc()
        finally:
            sfc_index.close()

    if processed_count == 0:
        print("ERROR: No ensemble members were successfully processed.", file=sys.stderr)
        return 1

    # 4. Generate Ensemble Mean (ensmean) Diagnostics and Plots
    print("\n" + "=" * 80)
    print("GENERATING ENSEMBLE MEAN (ENSMEAN) DIAGNOSTICS")
    print(f"Aggregating {len(all_member_results)} successful ensemble member(s)...")
    print("=" * 80)

    # Use first successful member's grids as reference (unpack first 7 values)
    ref_member, ref_lats, ref_lons, ref_times, _, _, ref_vort = all_member_results[0][:7]
    
    # Average 2D spatial local ACE
    mean_local_ace = np.mean([res[4] for res in all_member_results], axis=0)
    
    # Average monthly 2D spatial local ACE
    mean_local_ace_monthly = {}
    all_months = set()
    for res in all_member_results:
        if len(res) > 7 and res[7]:
            all_months.update(res[7].keys())
    for m in all_months:
        mean_local_ace_monthly[m] = np.mean([res[7][m] for res in all_member_results if m in res[7]], axis=0)

    # Average integrated diagnostic curves
    mean_diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for basin_name in BASINS:
        mean_diagnostics[basin_name] = {}
        ref_basin = all_member_results[0][5][basin_name]
        for field_name in ref_basin.keys():
            field_arrays = []
            for res in all_member_results:
                member_val = res[5][basin_name].get(field_name)
                if member_val is not None:
                    field_arrays.append(np.asarray(member_val))
            
            if field_arrays:
                min_len = min(arr.shape[0] for arr in field_arrays)
                aligned_arrays = [arr[:min_len] for arr in field_arrays]
                with np.errstate(all="ignore"):
                    mean_field = np.nanmean(aligned_arrays, axis=0)
                mean_diagnostics[basin_name][field_name] = mean_field
            else:
                mean_diagnostics[basin_name][field_name] = np.zeros(len(ref_times))

    any_uses_vorticity = any(res[6] for res in all_member_results)
    
    ensmean_cache_path = cache_dir / f"tc_conditioned_ace_{args.init_date}_ensmean.nc4"
    write_cache(
        ensmean_cache_path,
        args.init_date,
        "ensmean",
        ref_lats,
        ref_lons,
        ref_times,
        mean_local_ace,
        mean_diagnostics,
        any_uses_vorticity,
        mean_local_ace_monthly,
        basin_thresholds=basin_thresholds,
        threshold_mode=threshold_mode,
        threshold_source=threshold_source,
    )
    
    # Load cumulative curves to plot
    basin_cumulative_ace = {name: mean_diagnostics[name]["cumulative_ace"] for name in BASINS}
    plot_ace_diagnostics(
        local_ace=mean_local_ace,
        cumulative_ace_time=basin_cumulative_ace["North Atlantic"],
        time_dates=ref_times,
        latitudes=ref_lats,
        longitudes=ref_lons,
        init_date=args.init_date,
        ens="ensmean",
        plot_dir=plot_dir,
        basin_cumulative_ace=basin_cumulative_ace,
        local_ace_monthly=mean_local_ace_monthly,
        basin_thresholds=basin_thresholds,
    )

    print("=" * 80)
    print(f"Wrote ensemble mean cache: {ensmean_cache_path}")
    for basin_name in BASINS:
        total_ace = mean_diagnostics[basin_name]["cumulative_ace"][-1]
        strike_prob_sum = float(np.sum(mean_diagnostics[basin_name]["tc_flag"]))
        print(f"{basin_name:18s} mean_total_ace={total_ace:8.2f}  mean_tc_hits={strike_prob_sum:8.2f}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
