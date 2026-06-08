#!/usr/bin/env python3
"""Fuse spatial and temporal Accumulated Cyclone Energy (ACE) diagnostics from multiple initialization dates.

This script implements a 'Lagged Ensemble Fusion' post-processing utility (Option A).
It reads NetCDF4 cache files across multiple initialization dates (e.g., 0824 and 0829)
for a given year, averages their 2D spatial ACE maps, aligns their time-series by calendar 
date to average cumulative curves, writes a new combined lagged-ensemble mean cache, 
and generates premium, publication-quality multi-panel diagnostics.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure we can import calculate_tc_conditioned_ace from the scripts folder
scripts_dir = Path(__file__).parent.resolve()
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

try:
    from calculate_tc_conditioned_ace import (
        BASINS,
        plot_ace_diagnostics,
        read_cache,
        thresholds_from_diagnostics,
        write_cache,
        write_member_summary_csv,
    )
except ImportError as e:
    print(f"ERROR: Could not import calculate_tc_conditioned_ace.py: {e}")
    sys.exit(1)


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def parse_years(value: str | None) -> list[str]:
    if not value:
        return []
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
    return [str(year) for year in sorted(years)]


def fuse_year_lagged_ensemble(
    year: str,
    init_dates: list[str],
    months: list[str],
    cache_dir: Path,
    plot_dir: Path,
) -> None:
    print("=" * 80)
    print(f"FUSING LAGGED ENSEMBLE FOR YEAR {year}")
    print(f"  Target Init Dates: {init_dates}")
    print(f"  Target Months    : {months}")
    print("-" * 80)

    # 1. Discover all matching ensemble member cache files (excluding ensmean and lagged caches)
    cache_files: list[Path] = []
    cache_files_by_init: dict[str, list[Path]] = {}
    for init_date_md in init_dates:
        init_paths: list[Path] = []
        pattern = f"tc_conditioned_ace_{year}{init_date_md}_ens*.nc4"
        for p in cache_dir.glob(pattern):
            if not p.name.endswith("_ensmean.nc4") and "lagged" not in p.name:
                cache_files.append(p)
                init_paths.append(p)
        cache_files_by_init[init_date_md] = sorted(init_paths)
    cache_files = sorted(cache_files)

    if not cache_files:
        print(f"WARNING: No member cache files found for year {year} and init dates {init_dates} in {cache_dir}")
        print("Please run the calculate_tc_conditioned_ace.py pipeline for these dates first.")
        return

    print(f"Found {len(cache_files)} lagged ensemble member cache files.")
    for init_date_md in init_dates:
        print(f"  {year}{init_date_md}: {len(cache_files_by_init.get(init_date_md, []))} member(s)")

    # 2. Load all cache files
    member_lats: list[np.ndarray] = []
    member_lons: list[np.ndarray] = []
    member_times: list[list[datetime]] = []
    member_local_ace: list[np.ndarray] = []
    member_diagnostics: list[dict[str, dict[str, np.ndarray]]] = []
    member_uses_vorticity: list[bool] = []
    member_local_ace_monthly: list[dict[str, np.ndarray]] = []
    member_results: list[tuple] = []

    for f in cache_files:
        try:
            member_init_date, member_ens, lats, lons, times, local_ace, diag, uses_vort, monthly = read_cache(f)
            member_label = f"{member_init_date}_{member_ens}"
            member_lats.append(lats)
            member_lons.append(lons)
            member_times.append(times)
            member_local_ace.append(local_ace)
            member_diagnostics.append(diag)
            member_uses_vorticity.append(uses_vort)
            member_local_ace_monthly.append(monthly)
            member_results.append((member_label, lats, lons, times, local_ace, diag, uses_vort, monthly))
        except Exception as e:
            print(f"ERROR reading cache file {f.name}: {e}")
            return

    # Verify coordinate compatibility
    ref_lats = member_lats[0]
    ref_lons = member_lons[0]
    for i, (lats, lons) in enumerate(zip(member_lats, member_lons)):
        if not np.allclose(lats, ref_lats) or not np.allclose(lons, ref_lons):
            print(f"ERROR: Coordinate mismatch in cache file {cache_files[i].name} compared to {cache_files[0].name}")
            return

    # 3. Average the integrated 2D spatial local ACE maps
    print("\nAveraging 2D spatial ACE maps...")
    fused_local_ace = np.mean(member_local_ace, axis=0)

    # 4. Average monthly spatial ACE maps
    print("Averaging monthly spatial ACE maps...")
    fused_local_ace_monthly: dict[str, np.ndarray] = {}
    for m in months:
        monthly_grids = []
        for i, monthly_dict in enumerate(member_local_ace_monthly):
            if m in monthly_dict:
                monthly_grids.append(monthly_dict[m])
            else:
                # If a month is missing from a member, default to zeros of 2D grid shape
                monthly_grids.append(np.zeros((len(ref_lats), len(ref_lons))))
        
        if monthly_grids:
            fused_local_ace_monthly[m] = np.mean(monthly_grids, axis=0)

    # 5. Align and average the temporal curves by calendar datetime
    print("Aligning and averaging temporal diagnostics...")
    all_times_set = set()
    for times in member_times:
        all_times_set.update(times)
    sorted_times = sorted(list(all_times_set))
    print(f"  Unified timeline length: {len(sorted_times)} time steps (from {sorted_times[0]} to {sorted_times[-1]})")

    # Map each member's datetimes to their indices for fast retrieval
    member_time_to_idx = [{t: idx for idx, t in enumerate(times)} for times in member_times]

    # Initialize empty lists for fused diagnostics
    fused_diagnostics: dict[str, dict[str, list[float]]] = {}
    for basin_name in BASINS:
        fused_diagnostics[basin_name] = {
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

    # Populate fused diagnostics by averaging at each timestamp
    for basin_name in BASINS:
        basin_fields = fused_diagnostics[basin_name]
        for t in sorted_times:
            # Find all members that cover this timestamp
            matching_values = {field: [] for field in basin_fields}
            for i, time_map in enumerate(member_time_to_idx):
                if t in time_map:
                    idx = time_map[t]
                    for field in basin_fields:
                        val = member_diagnostics[i][basin_name][field][idx]
                        matching_values[field].append(val)
            
            # Average available values for each field
            for field in basin_fields:
                vals = matching_values[field]
                if len(vals) > 0:
                    basin_fields[field].append(float(np.mean(vals)))
                else:
                    basin_fields[field].append(0.0)

    # Convert lists to final numpy arrays for plotting
    fused_diagnostics_np: dict[str, dict[str, np.ndarray]] = {}
    for basin_name in BASINS:
        fused_diagnostics_np[basin_name] = {}
        for field, values in fused_diagnostics[basin_name].items():
            fused_diagnostics_np[basin_name][field] = np.array(values, dtype="float32")
    basin_thresholds = thresholds_from_diagnostics(fused_diagnostics_np)

    # Metadata parameters
    uses_vorticity = any(member_uses_vorticity)
    lagged_init_date = f"{year}08_lagged"
    lagged_ens = "ensmean"

    # 6. Save fused ensemble mean cache file
    output_cache_path = cache_dir / f"tc_conditioned_ace_{year}_lagged_ensmean.nc4"
    print(f"\nWriting fused lagged-ensemble mean cache: {output_cache_path}")
    write_cache(
        output_path=output_cache_path,
        init_date=lagged_init_date,
        ens=lagged_ens,
        latitudes=ref_lats,
        longitudes=ref_lons,
        valid_times=sorted_times,
        local_ace=fused_local_ace,
        diagnostics=fused_diagnostics, # write_cache expects lists
        uses_vorticity=uses_vorticity,
        local_ace_monthly=fused_local_ace_monthly,
        basin_thresholds=basin_thresholds or None,
        threshold_mode="lagged_fused",
        threshold_source="input member caches",
    )
    member_summary_path = cache_dir / f"tc_conditioned_ace_{year}_lagged_member_summary.csv"
    write_member_summary_csv(
        output_path=member_summary_path,
        init_date=f"{year}_lagged",
        member_results=member_results,
        basin_thresholds=basin_thresholds or {},
        threshold_mode="lagged_members",
        threshold_source="input member caches",
    )

    # 7. Generate publication-quality fused maps and plots
    print("Generating unified publication-quality plots...")
    plot_ace_diagnostics(
        local_ace=fused_local_ace,
        cumulative_ace_time=fused_diagnostics_np["North Atlantic"]["cumulative_ace"],
        time_dates=sorted_times,
        latitudes=ref_lats,
        longitudes=ref_lons,
        init_date=lagged_init_date,
        ens=lagged_ens,
        plot_dir=plot_dir,
        basin_cumulative_ace={basin: data["cumulative_ace"] for basin, data in fused_diagnostics_np.items()},
        local_ace_monthly=fused_local_ace_monthly,
        basin_thresholds=basin_thresholds or None,
    )
    print("Lagged Ensemble Fusion completed successfully!")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/cache", help="Directory where NetCDF cache files are stored")
    parser.add_argument("--plot-dir", default="plots", help="Directory where output plots will be saved")
    parser.add_argument("--years", default="1991:2024", help="Years to process, e.g. 1991:2024, 1991,1992, or all.")
    parser.add_argument("--init-dates", default="0824,0829", help="Comma-separated month/day init dates to combine")
    parser.add_argument("--months", default="09,10", help="Forecast months to use for monthly tracking, separated by commas")
    
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    plot_dir = Path(args.plot_dir)
    init_dates = parse_list(args.init_dates)
    months = parse_list(args.months)

    # Determine which years to process
    if args.years.lower() == "all":
        # Scan cache directory for years
        years_set = set()
        for p in cache_dir.glob("tc_conditioned_ace_*_ens*.nc4"):
            name = p.name
            # extract 4 digit year right after tc_conditioned_ace_
            match = re.search(r"tc_conditioned_ace_(\d{4})\d{4}_ens", name)
            if match:
                years_set.add(match.group(1))
        years = sorted(list(years_set))
    else:
        years = parse_years(args.years)

    print(f"Selected years for lagged fusion: {', '.join(years)}")

    for year in years:
        fuse_year_lagged_ensemble(
            year=year,
            init_dates=init_dates,
            months=months,
            cache_dir=cache_dir,
            plot_dir=plot_dir,
        )


if __name__ == "__main__":
    main()
