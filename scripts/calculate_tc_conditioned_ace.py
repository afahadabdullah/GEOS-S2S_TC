#!/usr/bin/env python3
"""Calculate TC-conditioned ACE from GEOS S2S3 SFC winds and ATM structure.

This script is an experimental alternative to the fixed wind-threshold ACE
workflow in ``calculate_ace_diagnostics.py``.

Instead of accumulating ACE whenever a basin contains winds above one chosen
threshold, this script uses a simple multivariate structural gate:

1. Find the sea-level-pressure minimum in each tropical cyclone basin.
2. Test whether the candidate center looks tropical-cyclone-like using:
   - negative local SLP anomaly
   - positive warm-core anomaly from T(850/500/200 hPa)
   - positive low-level moisture anomaly from QV(850 hPa)
   - hemisphere-consistent 850-hPa vorticity sign when U/V are available
3. If the structure is TC-like, accumulate ACE using the maximum nearby surface
   wind from the SFC files at the matching valid time.

The goal is not to be a full tracker. It is a basin-level ACE proxy with a
more physical storm gate than a single model-specific wind cutoff.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

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

MPS_TO_KNOTS = 1.94384
EARTH_RADIUS_M = 6_371_000.0

SLP_CANDIDATES = ("SLP", "slp", "PSL", "SeaLevelPressure")
T_CANDIDATES = ("T", "t", "TMP")
QV_CANDIDATES = ("QV", "qv", "Q", "QVAPOR")
U_CANDIDATES = ("U", "u", "UGRD")
V_CANDIDATES = ("V", "v", "VGRD")

BASINS = {
    "North Atlantic": {
        "lat_range": (0.0, 45.0),
        "lon_range": (-100.0, -10.0),
        "color": "#e55934",
    },
    "Northeast Pacific": {
        "lat_range": (0.0, 40.0),
        "lon_range": (-180.0, -100.0),
        "color": "#f3a712",
    },
    "Northwest Pacific": {
        "lat_range": (0.0, 45.0),
        "lon_range": (100.0, 180.0),
        "color": "#2ec4b6",
    },
    "North Indian": {
        "lat_range": (0.0, 40.0),
        "lon_range": (40.0, 100.0),
        "color": "#9b5de5",
    },
    "South Indian": {
        "lat_range": (-40.0, 0.0),
        "lon_range": (20.0, 135.0),
        "color": "#00bbf9",
    },
    "South Pacific": {
        "lat_range": (-40.0, 0.0),
        "lon_ranges": [(135.0, 180.0), (-180.0, -120.0)],
        "color": "#ff007f",
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
    if "lon_range" in basin_def:
        lon_min, lon_max = basin_def["lon_range"]
        return [np.where((longitudes >= lon_min) & (longitudes <= lon_max))[0]]

    lon_sets: list[np.ndarray] = []
    for lon_min, lon_max in basin_def["lon_ranges"]:
        lon_sets.append(np.where((longitudes >= lon_min) & (longitudes <= lon_max))[0])
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

        for path in sfc_files:
            yyyymm = forecast_yyyymm(path.name, collection)
            if forecast_months and (yyyymm is None or yyyymm[-2:] not in forecast_months):
                continue

            with netCDF4.Dataset(path, "r") as ds:
                if self.latitudes is None:
                    self.latitudes = np.asarray(ds.variables["lat"][:], dtype="float64")
                    self.longitudes = np.asarray(ds.variables["lon"][:], dtype="float64")
                time_values = ds.variables["time"][:]
                time_units = ds.variables["time"].units
                dates = netCDF4.num2date(time_values, time_units)
                for index, date_value in enumerate(dates):
                    self.entries.append(SFCMatch(valid_time=to_datetime(date_value), file_path=path, time_index=index))

        self.entries.sort(key=lambda item: item.valid_time)
        if self.latitudes is None or self.longitudes is None:
            raise ValueError("No SFC files were indexed")

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

        us = np.asarray(self._cached_ds.variables["US"][match.time_index, :, :], dtype="float64")
        vs = np.asarray(self._cached_ds.variables["VS"][match.time_index, :, :], dtype="float64")
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


def write_cache(
    output_path: Path,
    init_date: str,
    ens: str,
    valid_times: list[datetime],
    diagnostics: dict[str, dict[str, list[float]]],
    uses_vorticity: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    init_dt = datetime.strptime(init_date, "%Y%m%d")
    time_hours = np.array([(value - init_dt).total_seconds() / 3600.0 for value in valid_times], dtype="float32")

    with netCDF4.Dataset(output_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(valid_times))

        time_var = ds.createVariable("time", "f4", ("time",))
        time_var.units = f"hours since {init_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        time_var.long_name = "valid time"
        time_var[:] = time_hours

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

        ds.title = "TC-conditioned ACE diagnostics"
        ds.source_initialization = init_date
        ds.source_ensemble = ens
        ds.uses_vorticity = "true" if uses_vorticity else "false"
        ds.comment = (
            "Experimental ACE proxy using SFC wind intensity gated by ATM structure: "
            "SLP minimum, warm-core anomaly, low-level moisture anomaly, and optional 850-hPa vorticity sign."
        )


def read_cache(cache_path: Path) -> tuple[str, str, list[datetime], dict[str, dict[str, np.ndarray]], bool]:
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    with netCDF4.Dataset(cache_path, "r") as ds:
        init_date = str(getattr(ds, "source_initialization"))
        ens = str(getattr(ds, "source_ensemble"))
        uses_vorticity = str(getattr(ds, "uses_vorticity", "false")).lower() == "true"
        time_var = ds.variables["time"]
        times = [to_datetime(value) for value in netCDF4.num2date(time_var[:], time_var.units)]

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
            ):
                var_name = f"{field_name}_{basin_key}"
                diagnostics[basin_name][field_name] = np.asarray(ds.variables[var_name][:])

    return init_date, ens, times, diagnostics, uses_vorticity


def plot_tc_conditioned_ace_from_cache(cache_path: Path, plot_dir: Path) -> None:
    init_date, ens, times, diagnostics, uses_vorticity = read_cache(cache_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    print(f"Generating plots from cache: {cache_path}")

    # ------------------------------------------------------------------
    # Plot 1: Multi-basin cumulative TC-conditioned ACE
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    for basin_name, basin_def in BASINS.items():
        curve = diagnostics[basin_name]["cumulative_ace"]
        ax1.plot(times, curve, linewidth=2.2, color=basin_def["color"], label=basin_name)

    ax1.set_facecolor("#fafafa")
    fig1.patch.set_facecolor("#ffffff")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, linestyle="--", alpha=0.35, color="#cccccc", linewidth=0.5)
    ax1.tick_params(axis="both", labelsize=9, colors="#555555")
    ax1.set_ylabel("Cumulative TC-Conditioned ACE (10$^4$ kt$^2$)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Forecast Lead Time (Date)", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"GEOS S2S3 TC-Conditioned ACE by Basin\nInitialization: {init_date}  |  Ensemble: {ens}",
        fontsize=12,
        fontweight="bold",
        pad=12,
        color="#1e222a",
    )
    ax1.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig1.autofmt_xdate()
    curve_plot_path = plot_dir / f"tc_conditioned_ace_curves_{init_date}_{ens}.png"
    plt.savefig(curve_plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig1)
    print(f"  -> Saved basin ACE curves to: {curve_plot_path}")

    # ------------------------------------------------------------------
    # Plot 2: Global map of accepted candidate centers
    # ------------------------------------------------------------------
    fig2 = plt.figure(figsize=(14, 8), dpi=300)
    if HAS_CARTOPY:
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180.0))
        ax2.set_global()
        ax2.add_feature(cfeature.OCEAN, facecolor="#daeefb", zorder=0)
        ax2.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=1)
        ax2.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.5, zorder=2)
        ax2.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.3, linestyle=":", zorder=2)
        gl = ax2.gridlines(draw_labels=True, linewidth=0.2, color="#aaaaaa", alpha=0.4, linestyle="--", zorder=3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8, "color": "#555555"}
        gl.ylabel_style = {"size": 8, "color": "#555555"}
    else:
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.set_facecolor("#daeefb")
        ax2.set_xlim([-180.0, 180.0])
        ax2.set_ylim([-45.0, 50.0])
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.grid(True, linewidth=0.2, color="#aaaaaa", alpha=0.4, linestyle="--")

    for basin_name, basin_def in BASINS.items():
        basin_diag = diagnostics[basin_name]
        accepted = (basin_diag["tc_flag"] > 0) & np.isfinite(basin_diag["center_lat"]) & np.isfinite(basin_diag["center_lon"])
        if not np.any(accepted):
            continue

        lats = basin_diag["center_lat"][accepted]
        lons = basin_diag["center_lon"][accepted]
        sizes = 25.0 + 120.0 * np.sqrt(np.maximum(basin_diag["step_ace"][accepted], 0.0))
        color = basin_def["color"]

        if HAS_CARTOPY:
            ax2.scatter(
                lons,
                lats,
                s=sizes,
                c=color,
                alpha=0.75,
                edgecolors="#222222",
                linewidths=0.3,
                transform=ccrs.PlateCarree(),
                label=basin_name,
                zorder=4,
            )
        else:
            ax2.scatter(
                lons,
                lats,
                s=sizes,
                c=color,
                alpha=0.75,
                edgecolors="#222222",
                linewidths=0.3,
                label=basin_name,
                zorder=4,
            )

    ax2.set_title(
        f"GEOS S2S3 TC-Conditioned Candidate Centers\nInitialization: {init_date}  |  Ensemble: {ens}",
        fontsize=12,
        fontweight="bold",
        pad=12,
        color="#1e222a",
    )
    ax2.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    map_plot_path = plot_dir / f"tc_conditioned_centers_{init_date}_{ens}.png"
    plt.savefig(map_plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print(f"  -> Saved accepted-center map to: {map_plot_path}")

    # ------------------------------------------------------------------
    # Plot 3: North Atlantic gate diagnostics
    # ------------------------------------------------------------------
    basin_name = "North Atlantic"
    natl = diagnostics[basin_name]
    fig3, axes = plt.subplots(4, 1, figsize=(11, 9), dpi=300, sharex=True)
    fig3.patch.set_facecolor("#ffffff")
    gate_specs = [
        ("slp_anom_hpa", "SLP Anomaly (hPa)", "#33658a"),
        ("warm_core_anom_k", "Warm-Core Anomaly (K)", "#f26419"),
        ("qv850_anom_gpkg", "QV850 Anomaly (g kg$^{-1}$)", "#2a9d8f"),
        ("vmax_kt", "Nearby Max SFC Wind (kt)", "#c1121f"),
    ]

    tc_mask = natl["tc_flag"] > 0
    for ax, (field_name, ylabel, color) in zip(axes, gate_specs):
        series = natl[field_name]
        ax.plot(times, series, color=color, linewidth=1.8)
        if np.any(tc_mask):
            ax.scatter(np.asarray(times)[tc_mask], series[tc_mask], s=16, color="#111111", zorder=3)
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
        ax.set_facecolor("#fafafa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=8, colors="#555555")
        if field_name != "vmax_kt":
            ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle=":")

    if uses_vorticity:
        ax_vort = axes[-1].twinx()
        ax_vort.plot(times, natl["vort850_s1"], color="#6a4c93", linewidth=1.2, alpha=0.8)
        ax_vort.set_ylabel("850-hPa Vorticity (s$^{-1}$)", fontsize=8, color="#6a4c93")
        ax_vort.tick_params(axis="y", labelsize=8, colors="#6a4c93")

    axes[0].set_title(
        f"North Atlantic TC Gate Diagnostics\nInitialization: {init_date}  |  Ensemble: {ens}",
        fontsize=12,
        fontweight="bold",
        pad=12,
        color="#1e222a",
    )
    axes[-1].set_xlabel("Forecast Lead Time (Date)", fontsize=10, fontweight="bold")
    fig3.autofmt_xdate()
    gate_plot_path = plot_dir / f"tc_conditioned_gate_natl_{init_date}_{ens}.png"
    plt.savefig(gate_plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig3)
    print(f"  -> Saved North Atlantic gate diagnostics to: {gate_plot_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfc-root", default=os.environ.get("SFC_ROOT", DEFAULT_SFC_ROOT))
    parser.add_argument("--atm-root", default=os.environ.get("ATM_ROOT", DEFAULT_ATM_ROOT))
    parser.add_argument("--init-date", required=True, help="Forecast initialization date, e.g. 20200824")
    parser.add_argument("--ens", default="ens1", help="Ensemble member directory name (default: ens1)")
    parser.add_argument("--sfc-collection", default=DEFAULT_SFC_COLLECTION)
    parser.add_argument("--atm-collection", default=DEFAULT_ATM_COLLECTION)
    parser.add_argument("--months", default="09,10,11", help="Forecast months to use, separated by commas")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--plot-only-cache",
        default=None,
        help="Read an existing tc_conditioned_ace_*.nc4 cache file and create plots without reprocessing SFC/ATM data.",
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
        help="Radius around the candidate center used to search for max SFC wind",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plot_dir = Path(args.plot_dir)
    if args.plot_only_cache:
        plot_tc_conditioned_ace_from_cache(Path(args.plot_only_cache), plot_dir)
        return 0

    forecast_months = set(parse_list(args.months))

    sfc_root = Path(args.sfc_root)
    atm_root = Path(args.atm_root)
    cache_dir = Path(args.cache_dir)

    sfc_files = discover_collection_files(sfc_root, args.init_date, args.ens, args.sfc_collection)
    atm_files = discover_collection_files(atm_root, args.init_date, args.ens, args.atm_collection)

    if not sfc_files:
        print("ERROR: No SFC files found for the requested init/ens/collection", file=sys.stderr)
        return 1
    if not atm_files:
        print("ERROR: No ATM files found for the requested init/ens/collection", file=sys.stderr)
        return 1

    sfc_index = SFCIndex(sfc_files, args.sfc_collection, forecast_months)
    latitudes = sfc_index.latitudes
    longitudes = sfc_index.longitudes
    assert latitudes is not None
    assert longitudes is not None
    basin_coverage = {name: basin_has_coverage(latitudes, longitudes, basin_def) for name, basin_def in BASINS.items()}

    lat_env_radius = index_radius(latitudes, args.environment_radius_deg)
    lon_env_radius = index_radius(longitudes, args.environment_radius_deg)
    lat_core_radius = index_radius(latitudes, args.inner_core_radius_deg)
    lon_core_radius = index_radius(longitudes, args.inner_core_radius_deg)
    lat_wind_radius = index_radius(latitudes, args.wind_search_radius_deg)
    lon_wind_radius = index_radius(longitudes, args.wind_search_radius_deg)

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
        }
        for basin_name in BASINS
    }
    basin_totals = {basin_name: 0.0 for basin_name in BASINS}
    valid_times: list[datetime] = []
    uses_vorticity = False

    print("=" * 80)
    print("GEOS S2S3 TC-CONDITIONED ACE DIAGNOSTICS")
    print(f"Initialization: {args.init_date}  Ensemble: {args.ens}")
    print(f"SFC files found: {len(sfc_files)}")
    print(f"ATM files found: {len(atm_files)}")
    print("=" * 80)

    try:
        for atm_path in atm_files:
            yyyymm = forecast_yyyymm(atm_path.name, args.atm_collection)
            if forecast_months and (yyyymm is None or yyyymm[-2:] not in forecast_months):
                continue

            print(f"Reading ATM file: {atm_path.name}")
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

                for time_index, valid_time in enumerate(atm_times):
                    if forecast_months and valid_time.strftime("%m") not in forecast_months:
                        continue

                    sfc_match = sfc_index.nearest(valid_time, args.sfc_match_tolerance_hours)
                    if sfc_match is None:
                        print(f"  Skipping {valid_time}: no matching SFC time within tolerance")
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
                            continue

                        center_lat_idx, center_lon_idx, center_slp = basin_candidate_center(slp_hpa, latitudes, longitudes, basin_def)
                        center_lat = float(latitudes[center_lat_idx])
                        center_lon = float(longitudes[center_lon_idx])

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
                        vmax_kt, _, _ = max_near_center(
                            sfc_ws_kt,
                            center_lat_idx,
                            center_lon_idx,
                            lat_wind_radius,
                            lon_wind_radius,
                        )

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

                        tc_flag = int(all(criteria) and np.isfinite(vmax_kt))
                        step_ace = float(vmax_kt**2 * 1e-4) if tc_flag else 0.0
                        basin_totals[basin_name] += step_ace

                        basin_diag = diagnostics[basin_name]
                        basin_diag["cumulative_ace"].append(basin_totals[basin_name])
                        basin_diag["step_ace"].append(step_ace)
                        basin_diag["vmax_kt"].append(float(vmax_kt))
                        basin_diag["tc_flag"].append(tc_flag)
                        basin_diag["center_lat"].append(center_lat)
                        basin_diag["center_lon"].append(center_lon)
                        basin_diag["slp_hpa"].append(float(center_slp))
                        basin_diag["slp_anom_hpa"].append(float(slp_anom))
                        basin_diag["warm_core_anom_k"].append(float(warm_anom))
                        basin_diag["qv850_anom_gpkg"].append(float(qv_anom))
                        basin_diag["vort850_s1"].append(float(vort850))

        if not valid_times:
            print("ERROR: No valid ATM/SFC time matches were processed", file=sys.stderr)
            return 1

        output_path = cache_dir / f"tc_conditioned_ace_{args.init_date}_{args.ens}.nc4"
        write_cache(output_path, args.init_date, args.ens, valid_times, diagnostics, uses_vorticity)
        plot_tc_conditioned_ace_from_cache(output_path, plot_dir)

        print("=" * 80)
        print(f"Wrote cache: {output_path}")
        for basin_name in BASINS:
            total_ace = diagnostics[basin_name]["cumulative_ace"][-1]
            hits = int(np.sum(diagnostics[basin_name]["tc_flag"]))
            print(f"{basin_name:18s} total_ace={total_ace:8.2f}  tc_hits={hits}")
        print("=" * 80)
        return 0
    finally:
        sfc_index.close()


if __name__ == "__main__":
    sys.exit(main())
