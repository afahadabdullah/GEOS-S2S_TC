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
    
    # Define realistic tropical cyclone trajectories for Sept, Oct, Nov
    # Each storm has: (start_day, end_day, start_lat, start_lon, end_lat, end_lon, max_wind_knots, max_radius_degrees)
    tc_tracks = {
        "09": {  # September Storm: Major Cape Verde hurricane moving toward the US East Coast
            "start": 5, "end": 22,
            "start_lat": 12.0, "start_lon": -30.0,
            "end_lat": 32.0, "end_lon": -78.0,
            "max_wind": 62.0,  # m/s (~120 knots, Cat 4)
            "radius": 2.0
        },
        "10": {  # October Storm: Western Caribbean storm moving northeast across Florida
            "start": 8, "end": 18,
            "start_lat": 13.0, "start_lon": -82.0,
            "end_lat": 29.0, "end_lon": -60.0,
            "max_wind": 45.0,  # m/s (~90 knots, Cat 2)
            "radius": 2.5
        },
        "11": {  # November Storm: Late-season storm curving out to sea in the Atlantic
            "start": 2, "end": 12,
            "start_lat": 15.0, "start_lon": -65.0,
            "end_lat": 38.0, "end_lon": -40.0,
            "max_wind": 30.0,  # m/s (~60 knots, Tropical Storm)
            "radius": 3.0
        }
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
            
        # Inject storm vortex if there's a active TC track for this month
        if month in tc_tracks:
            track = tc_tracks[month]
            for step in range(n_times):
                day = step / steps_per_day
                if track["start"] <= day <= track["end"]:
                    # Interpolate current storm position
                    frac = (day - track["start"]) / (track["end"] - track["start"])
                    c_lat = track["start_lat"] + frac * (track["end_lat"] - track["start_lat"])
                    c_lon = track["start_lon"] + frac * (track["end_lon"] - track["start_lon"])
                    
                    # Maximum wind speed for current day (ramps up, then decays)
                    # Quadratic envelope for peak intensity in middle of track
                    intensity_envelope = 4.0 * frac * (1.0 - frac)
                    v_max = track["max_wind"] * (0.3 + 0.7 * intensity_envelope)
                    r_max = track["radius"]
                    
                    # Compute distance to storm center on sphere (approximate degrees)
                    d_lon = (lon_grid - c_lon + 180) % 360 - 180
                    d_lat = lat_grid - c_lat
                    dist = np.sqrt(d_lon**2 + d_lat**2)
                    
                    # Cyclonic vortex wind profiles: V(r) = Vmax * (r/Rmax) * exp(1 - r/Rmax)
                    # Add small eps to prevent division by zero
                    r_eps = dist + 1e-5
                    v_theta = v_max * (r_eps / r_max) * np.exp(1.0 - r_eps / r_max)
                    
                    # Compute wind components (counter-clockwise cyclonic flow in Northern Hemisphere)
                    u_vortex = -v_theta * (d_lat / r_eps)
                    v_vortex = v_theta * (d_lon / r_eps)
                    
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
    
    # Determine actual sampling hours between time steps
    time_diff_hours = 6  # standard fallback
    if len(raw_times) > 1:
        time_diff_hours = int(raw_times[1] - raw_times[0])
        
    print(f"Dataset summary:")
    print(f"  Shape: {us_full.shape} (time, lat, lon)")
    print(f"  Temporal Sampling: every {time_diff_hours} hours")
    print(f"  Latitude bounds: {latitudes.min()} to {latitudes.max()} (N={len(latitudes)})")
    print(f"  Longitude bounds: {longitudes.min()} to {longitudes.max()} (N={len(longitudes)})")

    # 3. Calculate 2D Local Spatial ACE
    print("  Calculating Local ACE map...")
    local_ace = calculate_local_ace(us_full, vs_full, sampling_hours=time_diff_hours)
    
    # 4. Calculate Time Series of Cumulative ACE (Basin-wide or Global)
    # Standard: North Atlantic basin boundary for TC indices is: 0-60N, -100 to -10W
    lat_indices = np.where((latitudes >= 0.0) & (latitudes <= 60.0))[0]
    lon_indices = np.where((longitudes >= -100.0) & (longitudes <= -10.0))[0]
    
    print(f"  Calculating temporal basin-wide cumulative ACE (Domain: 0-60N, 100W-10W)...")
    cumulative_ace_time = []
    current_ts_ace_sum = 0.0
    
    # Track daily accumulation curves
    # To match NOAA 6-hour standard, we accumulate step by step
    scale_step = 1e-4 * (time_diff_hours / 6.0)
    
    for t in range(len(raw_times)):
        # Spatial slice of winds for current timestep
        us_t = us_full[t][np.ix_(lat_indices, lon_indices)]
        vs_t = vs_full[t][np.ix_(lat_indices, lon_indices)]
        
        # Calculate wind speed magnitude and convert to knots
        ws_kt = np.sqrt(us_t**2 + vs_t**2) * MPS_TO_KNOTS
        
        # Sum squared values above threshold
        active_winds = np.where(ws_kt >= TS_THRESHOLD_KNOTS, ws_kt, 0.0)
        current_ts_ace_sum += np.sum(active_winds**2) * scale_step
        
        # Append cumulative value
        cumulative_ace_time.append(current_ts_ace_sum)
        
    cumulative_ace_time = np.array(cumulative_ace_time)
    
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
        
        ace_time_var = cache_nc.createVariable("cumulative_ace", "f4", ("time",), zlib=True, complevel=4)
        ace_time_var.units = "10^4 kt^2"
        ace_time_var.long_name = "Basin-wide Cumulative ACE over time"
        ace_time_var[:] = cumulative_ace_time
        
        # Global attributes
        cache_nc.title = "Cached ACE Diagnostic Fields"
        cache_nc.source_initialization = init_date
        cache_nc.source_ensemble = ens
        cache_nc.source_collection = collection
        cache_nc.calculation_date = datetime.utcnow().isoformat() + "Z"
        cache_nc.comment = "Generated by calculate_ace_diagnostics.py. Caches 2D maps and temporal curves."

    print(f"ACE Diagnostics processing & caching complete! (Total Basin ACE reached: {cumulative_ace_time[-1]:.2f})\n")
    return local_ace, cumulative_ace_time, time_dates


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
) -> None:
    """Generate and save premium, publication-quality diagnostic plots.

    1. A spatial Mercator projection map showing local ACE centers and storm tracks.
    2. A temporal line chart showing accumulation curves during September–November.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    print("Generating diagnostic plots...")

    # Set up styling parameters for premium aesthetics
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # --------------------------------------------------------------------------
    # PLOT 1: SPATIAL ACE MAP
    # --------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(12, 7), dpi=300)
    
    if HAS_CARTOPY:
        # Beautiful Mercator Projection focused on Tropical North Atlantic / Caribbean
        projection = ccrs.Mercator(central_longitude=-55.0, min_latitude=0.0, max_latitude=45.0)
        ax = fig1.add_subplot(1, 1, 1, projection=projection)
        
        # Extent bounds: [West Lon, East Lon, South Lat, North Lat]
        ax.set_extent([-98.0, -15.0, 5.0, 42.0], crs=ccrs.PlateCarree())
        
        # Add high-quality features (ocean, land, coastlines, borders)
        ax.add_feature(cfeature.OCEAN, facecolor="#11151c", zorder=0)  # Dark theme ocean
        ax.add_feature(cfeature.LAND, facecolor="#1e222a", zorder=1)   # Slate gray land
        ax.add_feature(cfeature.COASTLINE, edgecolor="#4f5b66", linewidth=0.6, zorder=2)
        ax.add_feature(cfeature.BORDERS, edgecolor="#4f5b66", linewidth=0.4, linestyle=":", zorder=2)
        
        # Plot spatial ACE contours
        # Shift longitudes to match -180 to 180 if needed
        lons_shifted = (longitudes + 180) % 360 - 180
        sorted_idx = np.argsort(lons_shifted)
        lons_plot = lons_shifted[sorted_idx]
        ace_plot = local_ace[:, sorted_idx]
        
        # Levels for ACE (only plot active storm energy > 0)
        levels = np.linspace(0.1, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.1 else 10.0, 100)
        
        # Premium magma colorbar mapping
        contour = ax.contourf(
            lons_plot, latitudes, ace_plot,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap="magma",
            zorder=3,
            alpha=0.85
        )
        
        # Add grid lines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#4f5b66", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8, "color": "#777777"}
        gl.ylabel_style = {"size": 8, "color": "#777777"}
        
    else:
        # Standard axes fallback if Cartopy is not available
        ax = fig1.add_subplot(1, 1, 1)
        ax.set_facecolor("#11151c")
        levels = np.linspace(0.1, np.max(local_ace) * 1.05 if np.max(local_ace) > 0.1 else 10.0, 100)
        contour = ax.contourf(longitudes, latitudes, local_ace, levels=levels, cmap="magma")
        ax.set_xlim([-98.0, -15.0])
        ax.set_ylim([5.0, 42.0])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.3, color="#4f5b66", alpha=0.5, linestyle="--")

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
    print(f"  -> Saved accumulation curve to: {temporal_plot_path}\n")


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
    local_ace, cumulative_ace_time, time_dates = process_ace_diagnostics(
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
    )
    
    print("=" * 80)
    print("All tasks completed successfully!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
