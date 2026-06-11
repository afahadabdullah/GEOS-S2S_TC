#!/usr/bin/env python3
"""Create a full diagnostics suite from percentile-threshold ACE caches."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
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

try:
    import netCDF4
except ImportError:
    print("ERROR: netCDF4 package is required. Load the earth environment.", file=sys.stderr)
    sys.exit(2)

from calculate_ibtracs_observed_percentiles import basin_from_boxes, basin_from_code, decode_chars, read_storm_time_text
from calculate_tc_conditioned_ace import BASINS, HAS_CARTOPY, ccrs, cfeature, read_cache
from plot_ace_lead_anomaly_skill import (
    SKILL_FIELDS,
    YEARLY_FIELDS,
    add_anomalies,
    add_ibtracs_to_yearly_rows,
    aggregate_yearly_rows,
    cache_member_info,
    discover_member_files,
    final_member_rows,
    parse_lead_months,
    print_skill_summary,
    skill_rows,
)
from plot_ace_yearly_timeseries import BASIN_ORDER, DEFAULT_IBTRACS, parse_list, parse_years, setup_style
from ocean_mask_utils import add_ocean_only_args


DEFAULT_CACHE = "data/cache_ace_geos_pctl1991_2022_slp_qv_sep5_1991_2024"
DEFAULT_TABLE_DIR = "data/analysis/ace_percentile_diagnostics"
DEFAULT_PLOT_DIR = "plots/ace_percentile_diagnostics"
DEFAULT_PREFIX = "geos_pctl1991_2022_slp_qv_sep5"
PLOT_BLACK = "#1e222a"
GEOS_BLUE = "#334f8d"
OBS_ORANGE = "#d95f02"
SPREAD_BLUE = "#9fb0d2"
GRID_COLOR = "#d1d5db"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"  -> Wrote {path}")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def setup_axis(ax, ylabel: str | None = None) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def lead_sort_key(lead: str) -> int:
    text = lead.replace("lead", "")
    return int(text) if text.isdigit() else 99


def build_yearly_skill_tables(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    years = parse_years(args.years)
    init_dates = parse_list(args.init_dates)
    lead_months = parse_lead_months(args.lead_months)
    member_rows = final_member_rows(Path(args.cache_dir), years, init_dates, lead_months)
    yearly = aggregate_yearly_rows(member_rows)
    add_ibtracs_to_yearly_rows(yearly, Path(args.ibtracs), lead_months, args)
    add_anomalies(yearly)
    skill = skill_rows(yearly)
    return member_rows, yearly, skill


def filter_yearly_by_member_count(rows: list[dict[str, object]], min_members: int) -> list[dict[str, object]]:
    if min_members <= 0:
        return rows
    return [row for row in rows if int(float(row.get("n_members", 0))) >= min_members]


def read_spatial_member_grids(
    cache_dir: Path,
    years: set[int],
    init_dates: list[str],
    lead_months: list[tuple[str, str]],
    min_lat: float,
    max_lat: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[np.ndarray]], dict[tuple[str, int], list[np.ndarray]]]:
    files = discover_member_files(cache_dir, years, init_dates)
    if not files:
        raise RuntimeError(f"No member cache files found in {cache_dir}")
    grids_by_lead: dict[str, list[np.ndarray]] = defaultdict(list)
    grids_by_lead_year: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    ref_lats: np.ndarray | None = None
    ref_lons: np.ndarray | None = None

    for path in files:
        info = cache_member_info(path)
        if info is None:
            continue
        init_date, _ens = info
        year = int(init_date[:4])
        _, _, lats, lons, _times, local_ace, _diag, _uses_vort, monthly = read_cache(path)
        if ref_lats is None:
            ref_lats = lats
            ref_lons = lons
        month_lookup = {month: lead for lead, month in lead_months}
        for month, grid in monthly.items():
            lead = month_lookup.get(month)
            if lead is None:
                continue
            grid = np.asarray(grid, dtype="float64").copy()
            lat_mask = (lats < min_lat) | (lats > max_lat)
            grid[lat_mask, :] = np.nan
            grids_by_lead[lead].append(grid)
            grids_by_lead_year[(lead, year)].append(grid)
        if not monthly:
            # Backward-compatible fallback: use full cache when monthly grids are absent.
            grid = np.asarray(local_ace, dtype="float64").copy()
            lat_mask = (lats < min_lat) | (lats > max_lat)
            grid[lat_mask, :] = np.nan
            for lead, _month in lead_months:
                grids_by_lead[lead].append(grid)
                grids_by_lead_year[(lead, year)].append(grid)
    if ref_lats is None or ref_lons is None:
        raise RuntimeError(f"No readable member caches found in {cache_dir}")
    return ref_lats, ref_lons, grids_by_lead, grids_by_lead_year


def shifted_grid(lons: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lons_shifted = (np.asarray(lons, dtype="float64") + 180.0) % 360.0 - 180.0
    order = np.argsort(lons_shifted)
    return lons_shifted[order], grid[:, order]


def plot_spatial_map(
    lats: np.ndarray,
    lons: np.ndarray,
    grid: np.ndarray,
    path: Path,
    title: str,
    extent: tuple[float, float, float, float] | None,
    dpi: int,
) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    lons_plot, grid_plot = shifted_grid(lons, grid)
    finite = grid_plot[np.isfinite(grid_plot) & (grid_plot > 0.0)]
    vmax = float(np.nanpercentile(finite, 99.3)) if finite.size else 1.0
    vmax = max(vmax, 1.0)
    levels = np.linspace(0.001, vmax, 64)
    cmap = LinearSegmentedColormap.from_list(
        "ace_map",
        ["#ffffcc", "#ffeda0", "#feb24c", "#f03b20", "#bd0026", "#5c0026"],
        N=256,
    )

    fig = plt.figure(figsize=(13.5, 6.4), dpi=dpi)
    if HAS_CARTOPY:
        projection = ccrs.PlateCarree(central_longitude=180.0 if extent is None else 0.0)
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        if extent is None:
            ax.set_global()
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#dff1fb", zorder=0)
        contour = ax.contourf(
            lons_plot,
            lats,
            grid_plot,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            extend="max",
            zorder=1,
            alpha=0.9,
        )
        ax.add_feature(cfeature.LAND, facecolor="#ece7df", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.55, edgecolor="#555555", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#aaaaaa", linestyle=":", zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#888888", alpha=0.35, linestyle="--", zorder=4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
    else:
        ax = fig.add_subplot(1, 1, 1)
        contour = ax.contourf(lons_plot, lats, grid_plot, levels=levels, cmap=cmap, extend="max")
        if extent is None:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-25, 50)
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.35)
    cbar = fig.colorbar(contour, ax=ax, orientation="horizontal", pad=0.08, aspect=42, shrink=0.82)
    cbar.set_label("Mean spatial ACE (10^4 kt^2)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_all_spatial_maps(
    lats: np.ndarray,
    lons: np.ndarray,
    grids_by_lead: dict[str, list[np.ndarray]],
    grids_by_lead_year: dict[tuple[str, int], list[np.ndarray]],
    args: argparse.Namespace,
    plot_dir: Path,
) -> None:
    for lead in sorted(grids_by_lead, key=lead_sort_key):
        mean_grid = np.nanmean(np.stack(grids_by_lead[lead], axis=0), axis=0)
        plot_spatial_map(
            lats,
            lons,
            mean_grid,
            plot_dir / f"{args.prefix}_{lead}_global_spatial_ace_mean.png",
            f"GEOS Percentile ACE Spatial Mean ({lead}, all cached years)",
            None,
            args.dpi,
        )
        plot_spatial_map(
            lats,
            lons,
            mean_grid,
            plot_dir / f"{args.prefix}_{lead}_north_atlantic_spatial_ace_mean.png",
            f"North Atlantic GEOS Percentile ACE Spatial Mean ({lead})",
            (-100.0, -10.0, 0.0, 45.0),
            args.dpi,
        )
        year_grids = grids_by_lead_year.get((lead, args.focus_year), [])
        if year_grids:
            year_mean = np.nanmean(np.stack(year_grids, axis=0), axis=0)
            plot_spatial_map(
                lats,
                lons,
                year_mean,
                plot_dir / f"{args.prefix}_{args.focus_year}_{lead}_north_atlantic_spatial_ace.png",
                f"North Atlantic GEOS Percentile ACE ({args.focus_year}, {lead})",
                (-100.0, -10.0, 0.0, 45.0),
                args.dpi,
            )


def plot_climatology_bars(yearly: list[dict[str, object]], path: Path, dpi: int) -> None:
    leads = sorted({str(row["lead"]) for row in yearly}, key=lead_sort_key)
    basins = ["All Basins"] + BASIN_ORDER
    fig, axes = plt.subplots(len(leads), 1, figsize=(13.5, max(4.0, 3.8 * len(leads))), sharex=True, dpi=dpi)
    axes = np.atleast_1d(axes)
    x = np.arange(len(basins))
    width = 0.36
    for ax, lead in zip(axes, leads):
        geos = []
        geos_yerr = []
        obs = []
        for basin in basins:
            rows = [row for row in yearly if str(row["lead"]) == lead and str(row["basin_name"]) == basin]
            geos_values = np.asarray([safe_float(row["geos_mean_ace"]) for row in rows], dtype="float64")
            obs_values = np.asarray([safe_float(row["ibtracs_ace"]) for row in rows], dtype="float64")
            geos.append(float(np.nanmean(geos_values)) if geos_values.size else np.nan)
            geos_yerr.append(float(np.nanstd(geos_values)) if geos_values.size > 1 else 0.0)
            obs.append(float(np.nanmean(obs_values)) if obs_values.size else np.nan)
        ax.bar(x - width / 2, geos, width, color=GEOS_BLUE, yerr=geos_yerr, capsize=2.5, label="GEOS mean")
        ax.bar(x + width / 2, obs, width, color=PLOT_BLACK, alpha=0.85, label="IBTrACS")
        setup_axis(ax, "ACE")
        ax.set_title(f"{lead} ACE Climatology", fontweight="bold")
        ax.legend(frameon=False, ncol=2, fontsize=9)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(basins, rotation=28, ha="right")
    fig.suptitle("GEOS Percentile ACE Climatology vs IBTrACS", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_metric_summary(skill: list[dict[str, object]], path: Path, dpi: int) -> None:
    leads = sorted({str(row["lead"]) for row in skill}, key=lead_sort_key)
    basins = ["All Basins"] + BASIN_ORDER
    metrics = [
        ("anom_corr", "Anomaly correlation", 0.0),
        ("anom_rmse", "Anomaly RMSE", None),
        ("raw_bias", "Raw bias", 0.0),
        ("clim_ratio", "GEOS / IBTrACS climatology", 1.0),
    ]
    for row in skill:
        geos = safe_float(row.get("geos_clim_ace"))
        obs = safe_float(row.get("ibtracs_clim_ace"))
        row["clim_ratio"] = geos / obs if np.isfinite(geos) and np.isfinite(obs) and obs != 0.0 else np.nan

    fig, axes = plt.subplots(len(metrics), len(leads), figsize=(5.7 * len(leads), 12.5), sharex=True, dpi=dpi)
    axes = np.asarray(axes)
    x = np.arange(len(basins))
    for col, lead in enumerate(leads):
        for row_idx, (field, label, refline) in enumerate(metrics):
            ax = axes[row_idx, col]
            values = []
            for basin in basins:
                item = next((r for r in skill if str(r["lead"]) == lead and str(r["basin_name"]) == basin), None)
                values.append(safe_float(item.get(field)) if item else np.nan)
            ax.bar(x, values, color=GEOS_BLUE)
            if refline is not None:
                ax.axhline(refline, color=PLOT_BLACK, linestyle="--", linewidth=0.9)
            setup_axis(ax, label)
            ax.set_title(f"{lead} {label}", fontsize=10.5, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(basins, rotation=30, ha="right")
    fig.suptitle("ACE Anomaly Skill and Amplitude Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def rows_for_basin_lead(yearly: list[dict[str, object]], basin: str, lead: str) -> list[dict[str, object]]:
    return sorted(
        [row for row in yearly if str(row["basin_name"]) == basin and str(row["lead"]) == lead],
        key=lambda row: int(row["year"]),
    )


def plot_north_atlantic_timeseries(yearly: list[dict[str, object]], path: Path, dpi: int) -> None:
    leads = sorted({str(row["lead"]) for row in yearly}, key=lead_sort_key)
    fig, axes = plt.subplots(len(leads), 2, figsize=(15.0, max(4.5, 3.9 * len(leads))), sharex=True, dpi=dpi)
    axes = np.atleast_2d(axes)
    for idx, lead in enumerate(leads):
        rows = rows_for_basin_lead(yearly, "North Atlantic", lead)
        years = np.asarray([int(row["year"]) for row in rows], dtype="int32")
        geos = np.asarray([safe_float(row["geos_mean_ace"]) for row in rows], dtype="float64")
        spread = np.asarray([safe_float(row["geos_std_ace"]) for row in rows], dtype="float64")
        obs = np.asarray([safe_float(row["ibtracs_ace"]) for row in rows], dtype="float64")
        geos_anom = np.asarray([safe_float(row["geos_anom_ace"]) for row in rows], dtype="float64")
        obs_anom = np.asarray([safe_float(row["ibtracs_anom_ace"]) for row in rows], dtype="float64")

        ax = axes[idx, 0]
        ax.plot(years, geos, color=GEOS_BLUE, marker="o", linewidth=1.8, label="GEOS")
        if np.any(spread > 0):
            ax.fill_between(years, geos - spread, geos + spread, color=SPREAD_BLUE, alpha=0.35, linewidth=0, label="GEOS member spread")
        ax.plot(years, obs, color=PLOT_BLACK, marker="s", linestyle="--", linewidth=1.6, label="IBTrACS")
        setup_axis(ax, "ACE")
        ax.set_title(f"North Atlantic Raw ACE ({lead})", fontweight="bold")

        ax = axes[idx, 1]
        ax.axhline(0.0, color="#94a3b8", linewidth=0.9)
        ax.plot(years, geos_anom, color=GEOS_BLUE, marker="o", linewidth=1.8, label="GEOS anomaly")
        if np.any(spread > 0):
            ax.fill_between(years, geos_anom - spread, geos_anom + spread, color=SPREAD_BLUE, alpha=0.35, linewidth=0, label="GEOS member spread")
        ax.plot(years, obs_anom, color=PLOT_BLACK, marker="s", linestyle="--", linewidth=1.6, label="IBTrACS anomaly")
        setup_axis(ax, "ACE anomaly")
        ax.set_title(f"North Atlantic ACE Anomaly ({lead})", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[-1, 0].set_xlabel("Year")
    axes[-1, 1].set_xlabel("Year")
    fig.suptitle("North Atlantic GEOS Percentile ACE vs IBTrACS", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_na_scatter(yearly: list[dict[str, object]], path: Path, dpi: int) -> None:
    leads = sorted({str(row["lead"]) for row in yearly}, key=lead_sort_key)
    fig, axes = plt.subplots(1, len(leads), figsize=(6.2 * len(leads), 5.2), dpi=dpi)
    axes = np.atleast_1d(axes)
    for ax, lead in zip(axes, leads):
        rows = rows_for_basin_lead(yearly, "North Atlantic", lead)
        x = np.asarray([safe_float(row["ibtracs_anom_ace"]) for row in rows], dtype="float64")
        y = np.asarray([safe_float(row["geos_anom_ace"]) for row in rows], dtype="float64")
        spread = np.asarray([safe_float(row["geos_std_ace"]) for row in rows], dtype="float64")
        mask = np.isfinite(x) & np.isfinite(y)
        ax.errorbar(x[mask], y[mask], yerr=spread[mask], fmt="o", color=GEOS_BLUE, ecolor=SPREAD_BLUE, alpha=0.9, capsize=2.0)
        if np.any(mask):
            lo = float(np.nanmin([np.nanmin(x[mask]), np.nanmin(y[mask])]))
            hi = float(np.nanmax([np.nanmax(x[mask]), np.nanmax(y[mask])]))
            pad = 0.1 * (hi - lo if hi > lo else 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PLOT_BLACK, linestyle="--", linewidth=1.0)
            corr = np.corrcoef(x[mask], y[mask])[0, 1] if int(np.sum(mask)) >= 3 else np.nan
            ax.text(0.04, 0.94, f"r={corr:.2f}\nn={int(np.sum(mask))}", transform=ax.transAxes, va="top", fontsize=10)
        setup_axis(ax, "GEOS anomaly")
        ax.set_xlabel("IBTrACS anomaly")
        ax.set_title(f"North Atlantic {lead}", fontweight="bold")
    fig.suptitle("North Atlantic ACE Anomaly Scatter", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def finite_value(value) -> float | None:
    if np.ma.is_masked(value):
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def nature_is_selected(nature: str, allowed: set[str]) -> bool:
    return not allowed or nature.strip().upper() in allowed


def read_ibtracs_track_points(
    path: Path,
    year: int,
    month: str,
    basin_name: str,
    wind_var: str,
    threshold_kt: float,
    nature_filter: str,
    basin_method: str,
    ocean_only: bool,
) -> list[dict[str, object]]:
    allowed_natures = {item.upper() for item in parse_list(nature_filter)}
    rows: list[dict[str, object]] = []
    with netCDF4.Dataset(path, "r") as ds:
        required = ["time", "lat", "lon", wind_var, "sid", "nature", "basin"]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"Missing IBTrACS variable(s): {', '.join(missing)}")
        time_var = ds.variables["time"]
        time_values = time_var[:]
        time_mask = np.ma.getmaskarray(time_values)
        dates = netCDF4.num2date(time_values, time_var.units, calendar=getattr(time_var, "calendar", "standard"))
        dist2land_values = ds.variables["dist2land"][:] if ocean_only and "dist2land" in ds.variables else None
        for storm_idx in range(time_values.shape[0]):
            sid = decode_chars(ds.variables["sid"][storm_idx, :])
            name = decode_chars(ds.variables["name"][storm_idx, :]) if "name" in ds.variables else ""
            for time_idx in range(time_values.shape[1]):
                if time_mask[storm_idx, time_idx]:
                    continue
                date = dates[storm_idx, time_idx]
                if int(date.year) != year or f"{int(date.month):02d}" != month:
                    continue
                if int(date.hour) not in (0, 6, 12, 18):
                    continue
                nature = read_storm_time_text(ds.variables["nature"], storm_idx, time_idx)
                if not nature_is_selected(nature, allowed_natures):
                    continue
                lat = finite_value(ds.variables["lat"][storm_idx, time_idx])
                lon = finite_value(ds.variables["lon"][storm_idx, time_idx])
                wind = finite_value(ds.variables[wind_var][storm_idx, time_idx])
                if lat is None or lon is None or wind is None or wind < threshold_kt:
                    continue
                if ocean_only and dist2land_values is not None:
                    dist2land = finite_value(dist2land_values[storm_idx, time_idx])
                    if dist2land is None or dist2land <= 0:
                        continue
                code = decode_chars(ds.variables["basin"][storm_idx, time_idx, :])
                basin = basin_from_boxes(lat, lon) if basin_method == "boxes" else basin_from_code(code)
                if basin != basin_name:
                    continue
                rows.append(
                    {
                        "sid": sid,
                        "name": name,
                        "time": f"{int(date.year):04d}-{int(date.month):02d}-{int(date.day):02d} {int(date.hour):02d}:00",
                        "lat": float(lat),
                        "lon": ((float(lon) + 180.0) % 360.0) - 180.0,
                        "wind_kt": float(wind),
                        "ace": float(wind**2 * 1.0e-4),
                    }
                )
    return rows


def cumulative_geos_na_2020(cache_dir: Path, init_dates: list[str], lead: str, month: str, year: int) -> tuple[list[object], np.ndarray, np.ndarray]:
    files = discover_member_files(cache_dir, {year}, init_dates)
    curves: list[np.ndarray] = []
    times_ref: list[object] | None = None
    for path in files:
        info = cache_member_info(path)
        if info is None:
            continue
        _, _, _lats, _lons, times, _local_ace, diagnostics, _uses_vort, _monthly = read_cache(path)
        mask = np.asarray([time.strftime("%m") == month for time in times], dtype=bool)
        if not np.any(mask):
            continue
        month_times = [time for time, keep in zip(times, mask) if keep]
        steps = np.asarray(diagnostics["North Atlantic"]["step_ace"], dtype="float64")[mask]
        curve = np.cumsum(np.nan_to_num(steps, nan=0.0))
        if times_ref is None:
            times_ref = month_times
        if len(month_times) == len(times_ref):
            curves.append(curve)
    if not curves or times_ref is None:
        return [], np.asarray([]), np.asarray([])
    stack = np.stack(curves, axis=0)
    return times_ref, np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def cumulative_ibtracs_from_points(points: list[dict[str, object]]) -> tuple[list[object], np.ndarray]:
    grouped: dict[str, float] = defaultdict(float)
    for row in points:
        grouped[str(row["time"])] += float(row["ace"])
    times = sorted(grouped)
    time_values = [datetime.strptime(time, "%Y-%m-%d %H:%M") for time in times]
    return time_values, np.cumsum([grouped[time] for time in times])


def plot_na_2020_diagnostics(
    yearly: list[dict[str, object]],
    member_rows: list[dict[str, object]],
    lats: np.ndarray,
    lons: np.ndarray,
    grids_by_lead_year: dict[tuple[str, int], list[np.ndarray]],
    args: argparse.Namespace,
    plot_dir: Path,
) -> list[dict[str, object]]:
    lead_months = parse_lead_months(args.lead_months)
    summary_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(len(lead_months), 2, figsize=(14.8, max(4.0, 4.1 * len(lead_months))), dpi=args.dpi)
    axes = np.atleast_2d(axes)
    for idx, (lead, month) in enumerate(lead_months):
        year_rows = [row for row in yearly if int(row["year"]) == args.focus_year and str(row["lead"]) == lead and str(row["basin_name"]) == "North Atlantic"]
        member_values = np.asarray(
            [
                safe_float(row["geos_member_ace"])
                for row in member_rows
                if int(row["year"]) == args.focus_year and str(row["lead"]) == lead and str(row["basin_name"]) == "North Atlantic"
            ],
            dtype="float64",
        )
        obs_value = safe_float(year_rows[0].get("ibtracs_ace")) if year_rows else np.nan
        geos_mean = float(np.nanmean(member_values)) if member_values.size else np.nan
        geos_std = float(np.nanstd(member_values)) if member_values.size > 1 else 0.0
        summary_rows.append(
            {
                "year": args.focus_year,
                "lead": lead,
                "month": month,
                "basin_name": "North Atlantic",
                "geos_mean_ace": geos_mean,
                "geos_std_ace": geos_std,
                "ibtracs_ace": obs_value,
                "bias": geos_mean - obs_value if np.isfinite(geos_mean) and np.isfinite(obs_value) else np.nan,
                "n_members": int(np.sum(np.isfinite(member_values))),
            }
        )

        ax = axes[idx, 0]
        if member_values.size:
            ax.hist(member_values[np.isfinite(member_values)], bins=14, color=GEOS_BLUE, alpha=0.75, label="GEOS members")
        ax.axvline(geos_mean, color=GEOS_BLUE, linewidth=2.0, label="GEOS mean")
        if np.isfinite(obs_value):
            ax.axvline(obs_value, color=PLOT_BLACK, linestyle="--", linewidth=2.0, label="IBTrACS")
        setup_axis(ax, "Member count")
        ax.set_xlabel("North Atlantic ACE")
        ax.set_title(f"{args.focus_year} {lead} North Atlantic Member Distribution", fontweight="bold")
        ax.legend(frameon=False, fontsize=9)

        ax = axes[idx, 1]
        times_geos, mean_curve, std_curve = cumulative_geos_na_2020(Path(args.cache_dir), parse_list(args.init_dates), lead, month, args.focus_year)
        if len(times_geos):
            ax.plot(times_geos, mean_curve, color=GEOS_BLUE, linewidth=1.8, label="GEOS mean")
            ax.fill_between(times_geos, mean_curve - std_curve, mean_curve + std_curve, color=SPREAD_BLUE, alpha=0.35, linewidth=0, label="GEOS spread")
        points = read_ibtracs_track_points(
            Path(args.ibtracs),
            args.focus_year,
            month,
            "North Atlantic",
            args.wind_var,
            args.threshold_kt,
            args.nature_filter,
            args.basin_method,
            args.ocean_only,
        )
        times_obs, obs_curve = cumulative_ibtracs_from_points(points)
        if len(times_obs):
            ax.plot(times_obs, obs_curve, color=PLOT_BLACK, linestyle="--", linewidth=1.8, marker="s", markersize=3.0, label="IBTrACS")
        setup_axis(ax, "Cumulative ACE")
        ax.set_title(f"{args.focus_year} {lead} North Atlantic Cumulative ACE", fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"North Atlantic {args.focus_year} GEOS Percentile ACE Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / f"{args.prefix}_{args.focus_year}_north_atlantic_lead_diagnostics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {out}")
    return summary_rows


def print_focus_summary(skill: list[dict[str, object]], focus_rows: list[dict[str, object]]) -> None:
    print("")
    print("North Atlantic focus metrics")
    print(f"{'lead':7s} {'n':>3s} {'r':>8s} {'rmse':>8s} {'mae':>8s} {'bias':>9s} {'GEOSclim':>9s} {'IBclim':>9s} {'ratio':>8s}")
    for row in sorted([r for r in skill if str(r["basin_name"]) == "North Atlantic"], key=lambda r: lead_sort_key(str(r["lead"]))):
        geos = safe_float(row.get("geos_clim_ace"))
        obs = safe_float(row.get("ibtracs_clim_ace"))
        ratio = geos / obs if np.isfinite(geos) and np.isfinite(obs) and obs else np.nan
        print(
            f"{str(row['lead']):7s} {int(float(row['n_years'])):3d} "
            f"{safe_float(row['anom_corr']):8.3f} {safe_float(row['anom_rmse']):8.3f} "
            f"{safe_float(row['anom_mae']):8.3f} {safe_float(row['raw_bias']):9.3f} "
            f"{geos:9.3f} {obs:9.3f} {ratio:8.3f}"
        )
    print("")
    print("Focus-year North Atlantic ACE")
    print(f"{'lead':7s} {'mon':>3s} {'members':>7s} {'GEOS':>9s} {'spread':>9s} {'IBTrACS':>9s} {'bias':>9s}")
    for row in focus_rows:
        print(
            f"{str(row['lead']):7s} {str(row['month']):>3s} {int(row['n_members']):7d} "
            f"{safe_float(row['geos_mean_ace']):9.3f} {safe_float(row['geos_std_ace']):9.3f} "
            f"{safe_float(row['ibtracs_ace']):9.3f} {safe_float(row['bias']):9.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--years", default="1991:2024")
    parser.add_argument("--init-dates", default="0824,0829")
    parser.add_argument("--lead-months", default="09,10")
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS)
    parser.add_argument("--wind-var", default="usa_wind")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument("--nature-filter", default="TS")
    parser.add_argument("--all-natures", dest="nature_filter", action="store_const", const="")
    parser.add_argument("--basin-method", choices=("boxes", "ibtracs_code"), default="boxes")
    parser.add_argument("--all-hours", dest="synoptic_only", action="store_false")
    parser.add_argument("--min-members-per-year", type=int, default=20)
    parser.add_argument("--min-lat", type=float, default=-25.0)
    parser.add_argument("--max-lat", type=float, default=50.0)
    parser.add_argument("--focus-year", type=int, default=2020)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--dpi", type=int, default=220)
    parser.set_defaults(synoptic_only=True)
    add_ocean_only_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_style()
    table_dir = Path(args.table_dir)
    plot_dir = Path(args.plot_dir)

    member_rows, yearly, skill = build_yearly_skill_tables(args)
    yearly = filter_yearly_by_member_count(yearly, args.min_members_per_year)
    add_anomalies(yearly)
    skill = skill_rows(yearly)

    years_used = sorted({int(row["year"]) for row in yearly if str(row["basin_name"]) == "All Basins"})
    if not years_used:
        print("ERROR: no complete yearly rows remain after member-count filtering.", file=sys.stderr)
        return 1
    print(f"Using complete/member-filtered years ({len(years_used)}): {', '.join(str(year) for year in years_used)}")

    write_csv(table_dir / f"{args.prefix}_member_values.csv", member_rows, [
        "year", "init_date", "ens", "lead", "month", "basin_name", "geos_member_ace", "cache_file",
    ])
    write_csv(table_dir / f"{args.prefix}_yearly.csv", yearly, YEARLY_FIELDS)
    skill_fields = SKILL_FIELDS + ["clim_ratio"]
    for row in skill:
        geos = safe_float(row.get("geos_clim_ace"))
        obs = safe_float(row.get("ibtracs_clim_ace"))
        row["clim_ratio"] = geos / obs if np.isfinite(geos) and np.isfinite(obs) and obs != 0.0 else np.nan
    write_csv(table_dir / f"{args.prefix}_skill.csv", skill, skill_fields)
    write_csv(table_dir / f"{args.prefix}_years.csv", [{"year": year} for year in years_used], ["year"])

    lats, lons, grids_by_lead, grids_by_lead_year = read_spatial_member_grids(
        Path(args.cache_dir),
        set(years_used),
        parse_list(args.init_dates),
        parse_lead_months(args.lead_months),
        args.min_lat,
        args.max_lat,
    )
    plot_all_spatial_maps(lats, lons, grids_by_lead, grids_by_lead_year, args, plot_dir)
    plot_climatology_bars(yearly, plot_dir / f"{args.prefix}_lead_climatology_by_basin.png", args.dpi)
    plot_metric_summary(skill, plot_dir / f"{args.prefix}_skill_metrics_by_basin.png", args.dpi)
    plot_north_atlantic_timeseries(yearly, plot_dir / f"{args.prefix}_north_atlantic_timeseries_spread.png", args.dpi)
    plot_na_scatter(yearly, plot_dir / f"{args.prefix}_north_atlantic_anomaly_scatter.png", args.dpi)
    focus_rows = plot_na_2020_diagnostics(yearly, member_rows, lats, lons, grids_by_lead_year, args, plot_dir)
    write_csv(table_dir / f"{args.prefix}_{args.focus_year}_north_atlantic_summary.csv", focus_rows, [
        "year", "lead", "month", "basin_name", "geos_mean_ace", "geos_std_ace", "ibtracs_ace", "bias", "n_members",
    ])

    print_skill_summary(skill)
    print_focus_summary(skill, focus_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
