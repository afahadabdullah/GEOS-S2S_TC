#!/usr/bin/env python3
"""Calculate and Plot Accumulated Cyclone Energy (ACE) Diagnostics from GEOS S2S3.

This script calculates the Local Grid-Point ACE index from surface winds (US, VS)
for a specific forecast initialization. It is restart-safe, saves calculations to
a NetCDF4 cache file, and creates high-quality spatial maps and temporal curves.
If the raw dataset is missing (e.g. running locally), it can automatically generate
a physically realistic mock dataset to verify the pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Set backend to Agg for non-interactive environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try importing netCDF4 and Xarray
try:
    import netCDF4
except ImportError:
    print("ERROR: netCDF4 package is required. Install with: pip install netCDF4", file=sys.stderr)
    sys.exit(2)

# Try importing Cartopy for maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: Cartopy not found. Plotting will fall back to standard coordinates.", file=sys.stderr)


# ==============================================================================
# 1. PHYSICAL CONSTANTS & CORE ACE FUNCTIONS
# ==============================================================================
MPS_TO_KNOTS = 1.94384
TS_THRESHOLD_KNOTS = 34.0  # Tropical Storm threshold (34 knots / 17.5 m/s)

# Standard Global Tropical Cyclone Basin Boundaries
BASINS = {
    "North Atlantic": {
        "lat_range": (0.0, 60.0),
        "lon_range": (-100.0, -10.0),
        "color": "#e55934",
        "label_xy": (-55.0, 30.0)
    },
    "Northeast Pacific": {
        "lat_range": (0.0, 40.0),
        "lon_range": (-180.0, -100.0),
        "color": "#f3a712",
        "label_xy": (-140.0, 20.0)
    },
    "Northwest Pacific": {
        "lat_range": (0.0, 60.0),
        "lon_range": (100.0, 180.0),
        "color": "#2ec4b6",
        "label_xy": (140.0, 30.0)
    },
    "North Indian": {
        "lat_range": (0.0, 40.0),
        "lon_range": (40.0, 100.0),
        "color": "#9b5de5",
        "label_xy": (70.0, 20.0)
    },
    "South Indian": {
        "lat_range": (-40.0, 0.0),
        "lon_range": (20.0, 135.0),
        "color": "#00bbf9",
        "label_xy": (77.0, -20.0)
    },
    "South Pacific": {
        "lat_range": (-40.0, 0.0),
        "lon_ranges": [(135.0, 180.0), (-180.0, -120.0)],
        "color": "#ff007f",
        "label_xy": (-160.0, -20.0)
    }
}


def calculate_local_ace(us: np.ndarray, vs: np.ndarray, sampling_hours: int = 6) -> np.ndarray:
    """Calculate the Local Grid-Point ACE index.

    ACE is defined as the sum of squared 10m wind speeds (in knots) above 34 knots.
    NOAA's standard calculation is based on 6-hourly intervals.
    If the input data has a different sampling frequency (e.g. 3-hourly), we adjust
    the accumulation by scaling by (sampling_hours / 6.0) to match NOAA's standard.

    Parameters
    ----------
    us : np.ndarray
        Eastward 10m wind component (time, lat, lon) in m/s.
    vs : np.ndarray
        Northward 10m wind component (time, lat, lon) in m/s.
    sampling_hours : int
        Temporal sampling interval of the input data in hours.

    Returns
    -------
    np.ndarray
        2D field (lat, lon) of Accumulated Cyclone Energy (10^4 kt^2).
    """
    # 1. Compute wind speed magnitude in m/s
    wind_speed_mps = np.sqrt(us**2 + vs**2)
    
    # 2. Convert to knots
    wind_speed_knots = wind_speed_mps * MPS_TO_KNOTS
    
    # 3. Apply the Tropical Storm threshold (keep winds >= 34 knots, set others to 0)
    ts_winds = np.where(wind_speed_knots >= TS_THRESHOLD_KNOTS, wind_speed_knots, 0.0)
    
    # 4. Sum the squared wind speeds over time
    squared_winds_sum = np.sum(ts_winds**2, axis=0)
    
    # 5. Convert to standard ACE units (10^-4 kt^2) and scale for sampling interval
    scale_factor = 1e-4 * (sampling_hours / 6.0)
    local_ace = squared_winds_sum * scale_factor
    
    return local_ace


# ==============================================================================
# 2. AUTOMATED MOCK DATA GENERATOR
# ==============================================================================
def generate_mock_geos_dataset(
    sfc_root: Path,
    init_date: str,
    ens: str,
    collection: str,
    forecast_months: list[str],
) -> None:
    """Generate high-fidelity, physically consistent mock SFC NetCDF files.

    Creates moving low-pressure/high-wind vortex patterns representing tropical
    cyclones traveling across the warm tropical Atlantic. This enables full local
    testing of the ACE pipeline on personal machines.
    """
    print(f"\n[Mock Data] Generating mock GEOS S2S3 SFC dataset for {init_date} ({ens})...")
    
    # Target collection directory
    collection_dir = sfc_root / "GEOS_fcst" / init_date / ens / collection
    collection_dir.mkdir(parents=True, exist_ok=True)
    
    # Coordinate grid: standard 0.5-degree grid matching GEOS 720x361 resolution
    nlon, nlat = 720, 361
    lons = np.linspace(-180, 180, nlon, dtype=np.float32)
    lats = np.linspace(-90, 90, nlat, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Define realistic global tropical cyclone trajectories for Sept, Oct, Nov
    # Each storm has: (start_day, end_day, start_lat, start_lon, end_lat, end_lon, max_wind, radius, hemi)
    tc_tracks = {
        "09": [
            {  # September NATL Storm: Major Cape Verde hurricane moving toward the US East Coast
                "start": 5, "end": 22,
                "start_lat": 12.0, "start_lon": -30.0,
                "end_lat": 32.0, "end_lon": -78.0,
                "max_wind": 62.0, "radius": 2.0, "hemi": "N"
            },
            {  # September WPAC Typhoon: Super typhoon heading toward Taiwan/China
                "start": 8, "end": 25,
                "start_lat": 10.0, "start_lon": 150.0,
                "end_lat": 24.0, "end_lon": 122.0,
                "max_wind": 75.0, "radius": 2.5, "hemi": "N"
            },
            {  # September EPAC Hurricane: Out to sea
                "start": 12, "end": 22,
                "start_lat": 13.0, "start_lon": -100.0,
                "end_lat": 22.0, "end_lon": -135.0,
                "max_wind": 50.0, "radius": 2.2, "hemi": "N"
            }
        ],
        "10": [
            {  # October NATL Storm: Western Caribbean storm moving northeast across Florida
                "start": 8, "end": 18,
                "start_lat": 13.0, "start_lon": -82.0,
                "end_lat": 29.0, "end_lon": -60.0,
                "max_wind": 45.0, "radius": 2.5, "hemi": "N"
            },
            {  # October NIND Cyclone: Bay of Bengal storm hitting India
                "start": 12, "end": 20,
                "start_lat": 9.0, "start_lon": 90.0,
                "end_lat": 19.0, "end_lon": 82.0,
                "max_wind": 42.0, "radius": 2.0, "hemi": "N"
            }
        ],
        "11": [
            {  # November NATL Storm: Late-season storm curving out to sea in the Atlantic
                "start": 2, "end": 12,
                "start_lat": 15.0, "start_lon": -65.0,
                "end_lat": 38.0, "end_lon": -40.0,
                "max_wind": 30.0, "radius": 3.0, "hemi": "N"
            },
            {  # November SIND Cyclone: Southern Indian Ocean storm (Southern Hemisphere - Clockwise flow!)
                "start": 5, "end": 18,
                "start_lat": -10.0, "start_lon": 90.0,
                "end_lat": -22.0, "end_lon": 68.0,
                "max_wind": 48.0, "radius": 2.4, "hemi": "S"
            },
            {  # November SPAC Cyclone: South Pacific storm crossing the dateline (Southern Hemisphere)
                "start": 10, "end": 22,
                "start_lat": -12.0, "start_lon": 165.0,
                "end_lat": -25.0, "end_lon": -165.0,
                "max_wind": 52.0, "radius": 2.6, "hemi": "S"
            }
        ]
    }
    
    init_year = init_date[:4]
    
    for month in forecast_months:
        yyyymm = f"{init_year}{month}"
        filename = f"{init_date}.{collection}.daily.{yyyymm}.nc4"
        file_path = collection_dir / filename
        
        # 6-hourly temporal resolution (4 steps per day) to keep file sizes optimal
        steps_per_day = 4
        days_in_month = 30 if month in ["09", "11"] else 31
        n_times = days_in_month * steps_per_day
        
        print(f"  -> Creating {filename} ({days_in_month} days, {n_times} steps)...")
        
        # Initialize background trade winds: Easterlies in the deep tropics, Westerlies in mid-latitudes
        us_background = np.zeros((n_times, nlat, nlon), dtype=np.float32)
        vs_background = np.zeros((n_times, nlat, nlon), dtype=np.float32)
        
        # Set easterly trades (approx -6 m/s) in tropics (5N-20N) and westerlies in mid-latitudes (30N-50N)
        for t in range(n_times):
            us_background[t, :, :] = -6.0 * np.exp(-((lat_grid - 12.0) / 8.0)**2) + 8.0 * np.exp(-((lat_grid - 40.0) / 10.0)**2)
            vs_background[t, :, :] = -1.0 * np.exp(-((lat_grid - 12.0) / 8.0)**2)
            
        # Inject storm vortices if there are active TC tracks for this month
        if month in tc_tracks:
            for track in tc_tracks[month]:
                for step in range(n_times):
                    day = step / steps_per_day
                    if track["start"] <= day <= track["end"]:
                        # Interpolate current storm position
                        frac = (day - track["start"]) / (track["end"] - track["start"])
                        c_lat = track["start_lat"] + frac * (track["end_lat"] - track["start_lat"])
                        
                        # Handle longitude interpolation, especially when crossing the 180 meridian
                        start_lon = track["start_lon"]
                        end_lon = track["end_lon"]
                        if abs(end_lon - start_lon) > 180:
                            if end_lon < start_lon:
                                end_lon += 360
                            else:
                                start_lon += 360
                        c_lon = start_lon + frac * (end_lon - start_lon)
                        c_lon = (c_lon + 180) % 360 - 180
                        
                        # Maximum wind speed for current day (ramps up, then decays)
                        intensity_envelope = 4.0 * frac * (1.0 - frac)
                        v_max = track["max_wind"] * (0.3 + 0.7 * intensity_envelope)
                        r_max = track["radius"]
                        
                        # Compute distance to storm center on sphere (approximate degrees)
                        d_lon = (lon_grid - c_lon + 180) % 360 - 180
                        d_lat = lat_grid - c_lat
                        dist = np.sqrt(d_lon**2 + d_lat**2)
                        
                        # Cyclonic vortex wind profiles: V(r) = Vmax * (r/Rmax) * exp(1 - r/Rmax)
                        r_eps = dist + 1e-5
                        v_theta = v_max * (r_eps / r_max) * np.exp(1.0 - r_eps / r_max)
                        
                        # Compute wind components (Northern = counter-clockwise, Southern = clockwise)
                        hemi_sign = 1.0 if track.get("hemi", "N") == "N" else -1.0
                        u_vortex = -hemi_sign * v_theta * (d_lat / r_eps)
                        v_vortex = hemi_sign * v_theta * (d_lon / r_eps)
                        
                        # Superimpose onto background wind field
                        us_background[step, :, :] += u_vortex
                        vs_background[step, :, :] += v_vortex
        
        # Write to compressed NetCDF4 file
        with netCDF4.Dataset(file_path, "w", format="NETCDF4") as nc:
            # Create dimensions
            nc.createDimension("time", None)  # Unlimited dimension
            nc.createDimension("lat", nlat)
            nc.createDimension("lon", nlon)
            
            # Create variables
            lat_var = nc.createVariable("lat", "f4", ("lat",))
            lon_var = nc.createVariable("lon", "f4", ("lon",))
            time_var = nc.createVariable("time", "i4", ("time",))
            
            # Write coordinates
            lat_var[:] = lats
            lon_var[:] = lons
            
            base_time = datetime(int(init_year), int(month), 1)
            time_steps = []
            for step in range(n_times):
                dt = base_time + timedelta(hours=step * (24 // steps_per_day))
                # Hours since base initialization
                time_steps.append(int((dt - datetime(int(init_year), 8, 24)).total_seconds() // 3600))
            time_var[:] = time_steps
            
            # Set variable attributes
            lat_var.units = "degrees_north"
            lat_var.long_name = "latitude"
            lon_var.units = "degrees_east"
            lon_var.long_name = "longitude"
            time_var.units = f"hours since {init_year}-08-24 00:00:00"
            time_var.long_name = "time"
            
            # Create compressed scientific variables (zlib level 4 for speed & size)
            us_var = nc.createVariable("US", "f4", ("time", "lat", "lon"), zlib=True, complevel=4)
            vs_var = nc.createVariable("VS", "f4", ("time", "lat", "lon"), zlib=True, complevel=4)
            
            us_var.units = "m s-1"
            us_var.long_name = "10-meter eastward wind"
            vs_var.units = "m s-1"
            vs_var.long_name = "10-meter northward wind"
            
            # Write data
            us_var[:] = us_background
            vs_var[:] = vs_background
            
            # Global attributes
            nc.title = "Mock GEOS S2S3 SFC daily thinned forecast variables"
            nc.institution = "NASA GSFC / GMU S2S TC Evaluation"
            nc.comment = "Generated automatically by calculate_ace_diagnostics.py for local testing."
            nc.source = "GEOS-S2S3 Forecast Model (Synthetic)"
            nc.creation_date = datetime.utcnow().isoformat() + "Z"

    print("[Mock Data] Generation complete. Files saved under:")
    print(f"  {collection_dir}/\n")


# ==============================================================================
# 3. DIAGNOSTICS & CACHING PIPELINE
# ==============================================================================
def process_ace_diagnostics(
    sfc_root: Path,
    init_date: str,
    ens: str,
    collection: str,
    forecast_months: list[str],
    cache_dir: Path,
    mock_if_missing: bool,
) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    """Load surface wind files, calculate ACE diagnostics, and cache results.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[datetime]]
        local_ace (lat, lon), cumulative_ace_time (time), time_steps
    """
    collection_dir = sfc_root / "GEOS_fcst" / init_date / ens / collection
    init_year = init_date[:4]
    
    # 1. Handle missing directories / files
    nc_files = sorted(collection_dir.glob(f"{init_date}.{collection}.*.nc4"))
    if not nc_files:
        if mock_if_missing:
            generate_mock_geos_dataset(sfc_root, init_date, ens, collection, forecast_months)
            nc_files = sorted(collection_dir.glob(f"{init_date}.{collection}.*.nc4"))
        else:
            print(f"ERROR: No NetCDF SFC files found in {collection_dir}", file=sys.stderr)
            print("Run with '--mock-if-missing' to automatically generate test data.", file=sys.stderr)
            sys.exit(1)
            
    print(f"Processing ACE diagnostics for Initialization: {init_date}, Ensemble: {ens}...")
    print(f"Found {len(nc_files)} NetCDF monthly files: {[f.name for f in nc_files]}")

    # 2. Iterate through files and accumulate spatial winds
    latitudes = None
    longitudes = None
    all_us = []
    all_vs = []
    time_units = None
    raw_times = []

    for file_path in nc_files:
        print(f"  Reading {file_path.name}...")
        with netCDF4.Dataset(file_path, "r") as nc:
            # Extract spatial coordinates if not done yet
            if latitudes is None:
                latitudes = nc.variables["lat"][:]
                longitudes = nc.variables["lon"][:]
            
            # Read variables
            all_us.append(nc.variables["US"][:])
            all_vs.append(nc.variables["VS"][:])
            raw_times.extend(nc.variables["time"][:])
            if time_units is None:
                time_units = nc.variables["time"].units
                
    # Concatenate monthly blocks along time axis
    us_full = np.concatenate(all_us, axis=0)
    vs_full = np.concatenate(all_vs, axis=0)
    
    # Determine actual sampling hours between time steps robustly
    time_diff_hours = 6.0  # standard fallback
    if len(raw_times) > 1:
        try:
            # Attempt to decode using netCDF4's standard num2date
            dates = netCDF4.num2date(raw_times[:2], units=time_units)
            dt_seconds = (dates[1] - dates[0]).total_seconds()
            time_diff_hours = float(dt_seconds / 3600.0)
        except Exception as e:
            # Fallback heuristic using the time units string
            diff_val = float(raw_times[1] - raw_times[0])
            units_lower = time_units.lower() if time_units else ""
            if "day" in units_lower:
                time_diff_hours = diff_val * 24.0
            elif "hour" in units_lower:
                time_diff_hours = diff_val
            elif "min" in units_lower:
                time_diff_hours = diff_val / 60.0
            elif "sec" in units_lower:
                time_diff_hours = diff_val / 3600.0
            else:
                time_diff_hours = diff_val

    # Ensure we don't have 0 or negative sampling hours
    if time_diff_hours <= 0:
        time_diff_hours = 6.0
        
    print(f"Dataset summary:")
    print(f"  Shape: {us_full.shape} (time, lat, lon)")
    print(f"  Temporal Sampling: every {time_diff_hours} hours")
    print(f"  Latitude bounds: {latitudes.min()} to {latitudes.max()} (N={len(latitudes)})")
    print(f"  Longitude bounds: {longitudes.min()} to {longitudes.max()} (N={len(longitudes)})")

    # 3. Calculate 2D Local Spatial ACE
    print("  Calculating Local ACE map...")
    local_ace = calculate_local_ace(us_full, vs_full, sampling_hours=time_diff_hours)
    
    # 4. Calculate Time Series of Cumulative ACE for All Basins
    print("  Calculating temporal cumulative ACE for all global basins...")
    basin_cumulative_ace = {name: [] for name in BASINS}
    basin_totals = {name: 0.0 for name in BASINS}
    scale_step = 1e-4 * (time_diff_hours / 6.0)
    
    # Pre-calculate masks or coordinate indices for each basin
    basin_masks = {}
    for name, b_def in BASINS.items():
        lat_min, lat_max = b_def["lat_range"]
        lat_idx = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
        
        if "lon_range" in b_def:
            lon_min, lon_max = b_def["lon_range"]
            lon_idx = np.where((longitudes >= lon_min) & (longitudes <= lon_max))[0]
            basin_masks[name] = (lat_idx, lon_idx, None)
        else:
            lon_idx_list = []
            for lon_min, lon_max in b_def["lon_ranges"]:
                lon_idx_list.append(np.where((longitudes >= lon_min) & (longitudes <= lon_max))[0])
            basin_masks[name] = (lat_idx, None, lon_idx_list)
            
    for t in range(len(raw_times)):
        us_t = us_full[t]
        vs_t = vs_full[t]
        
        for name, (lat_idx, lon_idx, lon_idx_list) in basin_masks.items():
            if len(lat_idx) == 0:
                step_ace = 0.0
            elif lon_idx is not None:
                if len(lon_idx) == 0:
                    step_ace = 0.0
                else:
                    us_sub = us_t[np.ix_(lat_idx, lon_idx)]
                    vs_sub = vs_t[np.ix_(lat_idx, lon_idx)]
                    ws_kt = np.sqrt(us_sub**2 + vs_sub**2) * MPS_TO_KNOTS
                    active_winds = np.where(ws_kt >= TS_THRESHOLD_KNOTS, ws_kt, 0.0)
                    step_ace = np.sum(active_winds**2) * scale_step
            else:
                step_ace = 0.0
                for l_idx in lon_idx_list:
                    if len(l_idx) > 0:
                        us_sub = us_t[np.ix_(lat_idx, l_idx)]
                        vs_sub = vs_t[np.ix_(lat_idx, l_idx)]
                        ws_kt = np.sqrt(us_sub**2 + vs_sub**2) * MPS_TO_KNOTS
                        active_winds = np.where(ws_kt >= TS_THRESHOLD_KNOTS, ws_kt, 0.0)
                        step_ace += np.sum(active_winds**2) * scale_step
            
            basin_totals[name] += step_ace
            basin_cumulative_ace[name].append(basin_totals[name])
            
    # For backward compatibility, map cumulative_ace_time to North Atlantic
    cumulative_ace_time = np.array(basin_cumulative_ace["North Atlantic"])
    
    # Parse datetimes for plotting
    epoch = datetime.strptime(time_units.split("since ")[1], "%Y-%m-%d %H:%M:%00" if "%H:%M:%00" in time_units else "%Y-%m-%d %H:%M:%S")
    time_dates = [epoch + timedelta(hours=int(h)) for h in raw_times]

    # 5. Save/Cache data to NetCDF4
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ace_cache_{init_date}_{ens}.nc4"
    print(f"  Caching calculation results to {cache_path}...")
    
    with netCDF4.Dataset(cache_path, "w", format="NETCDF4") as cache_nc:
        # Create dimensions
        cache_nc.createDimension("lat", len(latitudes))
        cache_nc.createDimension("lon", len(longitudes))
        cache_nc.createDimension("time", len(raw_times))
        
        # Create coordinate variables
        lat_var = cache_nc.createVariable("lat", "f4", ("lat",))
        lon_var = cache_nc.createVariable("lon", "f4", ("lon",))
        time_var = cache_nc.createVariable("time", "i4", ("time",))
        
        # Write coordinates
        lat_var[:] = latitudes
        lon_var[:] = longitudes
        time_var[:] = raw_times
        
        # Set attributes
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        time_var.units = time_units
        
        # Create scientific diagnostic variables
        ace_spatial_var = cache_nc.createVariable("local_ace", "f4", ("lat", "lon"), zlib=True, complevel=4)
        ace_spatial_var.units = "10^4 kt^2"
        ace_spatial_var.long_name = "Local Accumulated Cyclone Energy spatial field"
        ace_spatial_var[:] = local_ace
        
        # Backward compatible variable
        ace_time_var = cache_nc.createVariable("cumulative_ace", "f4", ("time",), zlib=True, complevel=4)
        ace_time_var.units = "10^4 kt^2"
        ace_time_var.long_name = "Basin-wide Cumulative ACE over time (North Atlantic)"
        ace_time_var[:] = cumulative_ace_time
        
        # Write separate cumulative ACE for each basin
        for name in BASINS:
            safe_name = name.lower().replace(" ", "_")
            b_var = cache_nc.createVariable(f"cumulative_ace_{safe_name}", "f4", ("time",), zlib=True, complevel=4)
            b_var.units = "10^4 kt^2"
            b_var.long_name = f"Cumulative ACE over time for {name} basin"
            b_var[:] = np.array(basin_cumulative_ace[name])
        
        # Global attributes
        cache_nc.title = "Cached ACE Diagnostic Fields"
        cache_nc.source_initialization = init_date
        cache_nc.source_ensemble = ens
        cache_nc.source_collection = collection
        cache_nc.calculation_date = datetime.utcnow().isoformat() + "Z"
        cache_nc.comment = "Generated by calculate_ace_diagnostics.py. Caches 2D maps and temporal curves globally."

    print(f"ACE Diagnostics processing & caching complete! (Total Basin ACE reached: North Atlantic = {cumulative_ace_time[-1]:.2f})\n")
    return local_ace, cumulative_ace_time, time_dates, basin_cumulative_ace


# ==============================================================================
# 4. PREMIUM DIAGNOSTIC PLOTTING SUITE
# ==============================================================================
def plot_ace_diagnostics(
    local_ace: np.ndarray,
    cumulative_ace_time: np.ndarray,
    time_dates: list[datetime],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    init_date: str,
    ens: str,
    plot_dir: Path,
    basin_cumulative_ace: dict | None = None,
) -> None:
    """Generate and save premium, publication-quality diagnostic plots.

    1. A spatial Mercator projection map showing local ACE centers and storm tracks.
    2. A temporal line chart showing accumulation curves during September–November.
    3. A global Pacific-centered map showing all basins and their respective integrated ACE.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    print("Generating diagnostic plots...")

    # Set up styling parameters for premium aesthetics
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    from matplotlib.colors import LinearSegmentedColormap
    # Define custom ACE colormap matching the WMO reference style (not darker starting, green -> yellow -> orange -> red -> pink -> white)
    colors = ["#4d924d", "#95d5b2", "#ffeb3b", "#ffa726", "#e65100", "#c2185b", "#ffffff"]
    ace_cmap = LinearSegmentedColormap.from_list("wmo_ace", colors, N=256)

    # --------------------------------------------------------------------------
    # PLOT 1: SPATIAL ACE MAP (NORTH ATLANTIC FOCUS)
    # --------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 7), dpi=300)
    
    if HAS_CARTOPY:
        # Beautiful Mercator Projection focused on Tropical North Atlantic / Caribbean
        projection = ccrs.Mercator(central_longitude=-55.0, min_latitude=0.0, max_latitude=45.0)
        ax = fig1.add_subplot(1, 1, 1, projection=projection)
        
        # Extent bounds: [West Lon, East Lon, South Lat, North Lat]
        ax.set_extent([-98.0, -15.0, 5.0, 42.0], crs=ccrs.PlateCarree())
        
        # Add high-quality features (ocean first, zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#daeefb", zorder=0)  # Light pastel blue ocean
        
        # Plot spatial ACE contours under the land
        lons_shifted = (longitudes + 180) % 360 - 180
        sorted_idx = np.argsort(lons_shifted)
        lons_plot = lons_shifted[sorted_idx]
        ace_plot = local_ace[:, sorted_idx]
        
        # Levels for ACE (only plot active storm energy > 0.05)
        levels = np.linspace(0.05, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.05 else 10.0, 100)
        
        contour = ax.contourf(
            lons_plot, latitudes, ace_plot,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=ace_cmap,
            zorder=1,   # Contours drawn on top of ocean, under land
            alpha=0.9
        )
        
        # Add land on top of contours to mask anything over land (zorder=2)
        ax.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=2)   # Premium light-beige land
        ax.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.4, linestyle=":", zorder=3)
        
        # Add grid lines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--", zorder=4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8, "color": "#555555"}
        gl.ylabel_style = {"size": 8, "color": "#555555"}
        
    else:
        # Standard axes fallback if Cartopy is not available
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_facecolor("#daeefb")
        levels = np.linspace(0.05, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.05 else 10.0, 100)
        contour = ax.contourf(longitudes, latitudes, local_ace, levels=levels, cmap=ace_cmap)
        ax.set_xlim([-98.0, -15.0])
        ax.set_ylim([5.0, 42.0])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.3, color="#aaaaaa", alpha=0.4, linestyle="--")

    # Add a beautiful colorbar
    cbar = fig1.colorbar(contour, ax=ax, orientation="horizontal", pad=0.08, aspect=40, shrink=0.8)
    cbar.set_label("Accumulated Cyclone Energy Index (10$^4$ kt$^2$)", fontsize=9, color="#333333", fontweight="bold", labelpad=6)
    cbar.ax.tick_params(labelsize=8, color="#555555", labelcolor="#333333")
    cbar.outline.set_visible(False)
    
    # Custom titles and captions
    plt.title(
        f"GEOS S2S3 Local Accumulated Cyclone Energy (ACE) Map\n"
        f"Initialization: {init_date}  |  Member: {ens}  |  Season: Sep-Nov",
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
    
    # Plot accumulation curve
    ax2.plot(time_dates, cumulative_ace_time, color="#e55934", linewidth=2.5, label="Cumulative ACE")
    
    # Shading under the curve
    ax2.fill_between(time_dates, cumulative_ace_time, color="#e55934", alpha=0.1)
    
    # Premium axes styling
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
    
    # Format x-axis dates beautifully
    fig2.autofmt_xdate()
    
    # Title
    ax2.set_title(
        f"S2S3 North Atlantic Cumulative ACE Curve\n"
        f"Initialization: {init_date}  |  Ensemble: {ens}",
        fontsize=12, fontweight="bold", pad=15, color="#1e222a"
    )
    
    # Final annotation showing peak seasonal ACE reached
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
    # PLOT 3: GLOBAL MULTI-BASIN ACE MAP
    # --------------------------------------------------------------------------
    fig3 = plt.figure(figsize=(14, 8), dpi=300)
    
    if HAS_CARTOPY:
        # Pacific-centered Cylindrical Projection (central_longitude=180.0)
        projection = ccrs.PlateCarree(central_longitude=180.0)
        ax3 = fig3.add_subplot(1, 1, 1, projection=projection)
        
        ax3.set_global()
        
        # Add high-quality features (ocean first, zorder=0)
        ax3.add_feature(cfeature.OCEAN, facecolor="#daeefb", zorder=0)  # Light pastel blue ocean
        
        # Plot spatial ACE contours globally under the land
        lons_shifted = (longitudes + 180) % 360 - 180
        sorted_idx = np.argsort(lons_shifted)
        lons_plot = lons_shifted[sorted_idx]
        ace_plot = local_ace[:, sorted_idx]
        
        levels = np.linspace(0.05, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.05 else 10.0, 100)
        
        contour3 = ax3.contourf(
            lons_plot, latitudes, ace_plot,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=ace_cmap,
            zorder=1,   # Contours drawn on top of ocean, under land
            alpha=0.9
        )
        
        # Add land on top of contours to mask anything over land (zorder=2)
        ax3.add_feature(cfeature.LAND, facecolor="#eae6df", zorder=2)   # Premium light-beige land
        ax3.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.5, zorder=3)
        ax3.add_feature(cfeature.BORDERS, edgecolor="#bbbbbb", linewidth=0.3, linestyle=":", zorder=3)
        
        # Add grid lines
        gl3 = ax3.gridlines(draw_labels=True, linewidth=0.2, color="#aaaaaa", alpha=0.4, linestyle="--", zorder=4)
        gl3.top_labels = False
        gl3.right_labels = False
        gl3.xlabel_style = {"size": 8, "color": "#555555"}
        gl3.ylabel_style = {"size": 8, "color": "#555555"}
        
    else:
        # Standard axes fallback
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.set_facecolor("#daeefb")
        levels = np.linspace(0.05, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.05 else 10.0, 100)
        contour3 = ax3.contourf(longitudes, latitudes, local_ace, levels=levels, cmap=ace_cmap)
        ax3.set_xlim([-180.0, 180.0])
        ax3.set_ylim([-60.0, 60.0])
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Latitude")
        ax3.grid(True, linewidth=0.2, color="#aaaaaa", alpha=0.4, linestyle="--")

    # Draw basin boundary rectangles and labels
    for name, b_def in BASINS.items():
        color = b_def["color"]
        lat_min, lat_max = b_def["lat_range"]
        
        # Get total ACE for this basin (use final timestep)
        total_ace = 0.0
        if basin_cumulative_ace and name in basin_cumulative_ace:
            total_ace = basin_cumulative_ace[name][-1]
            
        # Draw rectangular boundaries (using solid line '-' for neat appearance)
        if "lon_range" in b_def:
            lon_min, lon_max = b_def["lon_range"]
            lons_rect = [lon_min, lon_max, lon_max, lon_min, lon_min]
            lats_rect = [lat_min, lat_min, lat_max, lat_max, lat_min]
            if HAS_CARTOPY:
                ax3.plot(lons_rect, lats_rect, color=color, linewidth=1.8, linestyle="-", transform=ccrs.PlateCarree(), zorder=5)
            else:
                ax3.plot(lons_rect, lats_rect, color=color, linewidth=1.8, linestyle="-", zorder=5)
        else:
            # Split-meridian case (South Pacific)
            for lon_min, lon_max in b_def["lon_ranges"]:
                lons_rect = [lon_min, lon_max, lon_max, lon_min, lon_min]
                lats_rect = [lat_min, lat_min, lat_max, lat_max, lat_min]
                if HAS_CARTOPY:
                    ax3.plot(lons_rect, lats_rect, color=color, linewidth=1.8, linestyle="-", transform=ccrs.PlateCarree(), zorder=5)
                else:
                    ax3.plot(lons_rect, lats_rect, color=color, linewidth=1.8, linestyle="-", zorder=5)
                    
        # Place label inside or near the box (using premium light theme labels)
        label_lon, label_lat = b_def["label_xy"]
        bbox_props = dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=color, lw=1.2, alpha=0.9)
        
        if HAS_CARTOPY:
            ax3.text(
                label_lon, label_lat, f"{name}\nACE: {total_ace:.2f}",
                transform=ccrs.PlateCarree(),
                color="#1e222a", fontsize=8, fontweight="bold",
                ha="center", va="center", bbox=bbox_props, zorder=6
            )
        else:
            ax3.text(
                label_lon, label_lat, f"{name}\nACE: {total_ace:.2f}",
                color="#1e222a", fontsize=8, fontweight="bold",
                ha="center", va="center", bbox=bbox_props, zorder=6
            )

    # Add global colorbar
    cbar3 = fig3.colorbar(contour3, ax=ax3, orientation="horizontal", pad=0.08, aspect=45, shrink=0.75)
    cbar3.set_label("Accumulated Cyclone Energy Index (10$^4$ kt$^2$)", fontsize=9, color="#333333", fontweight="bold", labelpad=6)
    cbar3.ax.tick_params(labelsize=8, color="#555555", labelcolor="#333333")
    cbar3.outline.set_visible(False)
    
    # Custom title
    plt.title(
        f"GEOS S2S3 Global Multi-Basin Accumulated Cyclone Energy (ACE) Map\n"
        f"Initialization: {init_date}  |  Ensemble: {ens}  |  Season: Sep-Nov",
        fontsize=13, fontweight="bold", pad=15, color="#1e222a"
    )
    
    global_plot_path = plot_dir / f"global_ace_map_{init_date}_{ens}.png"
    plt.savefig(global_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved global multi-basin map to: {global_plot_path}\n")


# ==============================================================================
# 5. ENTRY POINT & ARGUMENT PARSING
# ==============================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sfc-root",
        default=os.environ.get("SFC_ROOT", "data"),
        help="Local surface data root directory containing GEOS_fcst/ (defaults to 'data')."
    )
    parser.add_argument(
        "--init-date",
        default=None,
        help="Forecast initialization date (e.g. 20240824). If omitted, picks a random available date."
    )
    parser.add_argument(
        "--ens",
        default="ens1",
        help="Ensemble member directory name (default: ens1)."
    )
    parser.add_argument(
        "--collection",
        default="sfc_tavg_3hr_glo_L720x361_sfc",
        help="SFC NetCDF collection folder name."
    )
    parser.add_argument(
        "--months",
        default="09,10,11",
        help="Forecast months to compute ACE over, separated by commas (default: 09,10,11)."
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Directory to save the processed NetCDF4 diagnostic cache files (default: data/cache)."
    )
    parser.add_argument(
        "--plot-dir",
        default="plots",
        help="Directory to save the premium output diagnostics plots (default: plots)."
    )
    parser.add_argument(
        "--mock-if-missing",
        action="store_true",
        help="If raw SFC NetCDF forecast files are missing, automatically generate a realistic mock dataset."
    )

    args = parser.parse_args(argv)
    
    sfc_root = Path(args.sfc_root)
    cache_dir = Path(args.cache_dir)
    plot_dir = Path(args.plot_dir)
    forecast_months = [m.strip() for m in args.months.split(",") if m.strip()]
    
    # Resolve initialization date
    init_date = args.init_date
    if not init_date:
        # Try finding available init dates in config file or generate default
        config_init_file = Path("config/init_dates_late_aug_1991_2024.txt")
        if config_init_file.exists():
            dates = [d.strip() for d in config_init_file.read_text().splitlines() if d.strip()]
            if dates:
                # Select a random year for demonstration
                # We'll use 20240824 as a stable and representative default
                init_date = "20240824" if "20240824" in dates else dates[-1]
        
        # Ultimate fallback
        if not init_date:
            init_date = "20240824"
            
    print("=" * 80)
    print(f"GEOS S2S3 ACCUMULATED CYCLONE ENERGY (ACE) DIAGNOSTICS GENERATOR")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run the diagnostics pipeline
    local_ace, cumulative_ace_time, time_dates, basin_cumulative_ace = process_ace_diagnostics(
        sfc_root=sfc_root,
        init_date=init_date,
        ens=args.ens,
        collection=args.collection,
        forecast_months=forecast_months,
        cache_dir=cache_dir,
        mock_if_missing=args.mock_if_missing,
    )
    
    # Load coordinates back for plotting
    cache_path = cache_dir / f"ace_cache_{init_date}_{args.ens}.nc4"
    with netCDF4.Dataset(cache_path, "r") as cached:
        latitudes = cached.variables["lat"][:]
        longitudes = cached.variables["lon"][:]
        
    # Generate spatial maps and time-series plots
    plot_ace_diagnostics(
        local_ace=local_ace,
        cumulative_ace_time=cumulative_ace_time,
        time_dates=time_dates,
        latitudes=latitudes,
        longitudes=longitudes,
        init_date=init_date,
        ens=args.ens,
        plot_dir=plot_dir,
        basin_cumulative_ace=basin_cumulative_ace,
    )
    
    print("=" * 80)
    print("All tasks completed successfully!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
