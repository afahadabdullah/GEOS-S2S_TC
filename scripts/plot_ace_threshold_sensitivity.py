#!/usr/bin/env python3
"""Evaluate GEOS TC-conditioned ACE sensitivity to wind threshold choices."""

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

from calculate_tc_conditioned_ace import BASINS, read_cache
from ocean_mask_utils import add_ocean_only_args, build_ocean_checker
from plot_ace_yearly_timeseries import (
    BASIN_ORDER,
    DEFAULT_IBTRACS,
    parse_list,
    parse_months,
    parse_years,
    read_ibtracs_observed_ace,
    setup_style,
)


DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_OUTPUT_DIR = "data/analysis/ace_threshold_sensitivity"
DEFAULT_PLOT_DIR = "plots/ace_threshold_sensitivity"


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for item in parse_list(value):
        out.append(float(item))
    return sorted(set(out))


def cache_member_info(path: Path) -> tuple[str, str] | None:
    match = re.match(r"tc_conditioned_ace_(\d{8})_(ens[^.]+)\.nc4$", path.name)
    if not match:
        return None
    if path.name.endswith("_ensmean.nc4") or "lagged" in path.name:
        return None
    return match.group(1), match.group(2)


def discover_member_files(cache_dir: Path, years: set[int], init_dates: list[str]) -> list[Path]:
    paths: list[Path] = []
    for year in sorted(years):
        for init_date_md in init_dates:
            paths.extend(sorted(cache_dir.glob(f"tc_conditioned_ace_{year}{init_date_md}_ens*.nc4")))
    return [path for path in sorted(paths) if cache_member_info(path) is not None]


def build_geos_ocean_checker(args: argparse.Namespace):
    if not args.ocean_only:
        return None
    checker, warning = build_ocean_checker(
        args.ocean_mask_source,
        mask_file=args.ocean_mask_file,
        threshold=args.ocean_threshold,
        require_mask=True,
    )
    print(f"Ocean-only GEOS cache filter enabled: source={checker.source}")
    if warning:
        print(f"WARNING: GEOS ocean mask fallback: {warning}")
    return checker


def time_scale_factors(times: list[object]) -> np.ndarray:
    if not times:
        return np.asarray([], dtype="float64")
    if len(times) == 1:
        return np.asarray([1.0e-4], dtype="float64")

    deltas = []
    for index in range(len(times)):
        if index < len(times) - 1:
            delta_hours = abs((times[index + 1] - times[index]).total_seconds()) / 3600.0
        else:
            delta_hours = abs((times[index] - times[index - 1]).total_seconds()) / 3600.0
        if not np.isfinite(delta_hours) or delta_hours <= 0.0 or delta_hours > 24.0:
            delta_hours = 6.0
        deltas.append(delta_hours)
    return np.asarray(deltas, dtype="float64") / 6.0 * 1.0e-4


def read_geos_member_rows(
    cache_dir: Path,
    years: set[int],
    init_dates: list[str],
    months: set[str],
    thresholds: list[float],
    ocean_checker,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    member_files = discover_member_files(cache_dir, years, init_dates)
    if not member_files:
        raise RuntimeError(f"No GEOS member ACE caches found in {cache_dir}")

    print(f"Reading {len(member_files)} GEOS member ACE cache(s) for threshold sensitivity.")
    geos_land_filtered_steps = 0
    for path in member_files:
        info = cache_member_info(path)
        if info is None:
            continue
        init_date, ens = info
        year = int(init_date[:4])
        _, _, _, _, times, _, diagnostics, _, _ = read_cache(path)
        time_months = np.asarray([time_value.strftime("%m") for time_value in times])
        month_mask = np.asarray([month in months for month in time_months], dtype=bool)
        scale = time_scale_factors(times)
        basin_inputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for basin_name in BASIN_ORDER:
            tc_flag = np.asarray(diagnostics[basin_name]["tc_flag"], dtype="float64")
            vmax = np.asarray(diagnostics[basin_name]["vmax_kt"], dtype="float64")
            finite_mask = np.isfinite(vmax) & np.isfinite(tc_flag) & month_mask
            if ocean_checker is not None:
                center_lat = np.asarray(diagnostics[basin_name]["center_lat"], dtype="float64")
                center_lon = np.asarray(diagnostics[basin_name]["center_lon"], dtype="float64")
                center_finite = finite_mask & np.isfinite(center_lat) & np.isfinite(center_lon)
                ocean_mask = np.zeros_like(finite_mask, dtype=bool)
                for index in np.flatnonzero(center_finite):
                    ocean_mask[index] = ocean_checker.is_ocean(float(center_lat[index]), float(center_lon[index]))
                geos_land_filtered_steps += int(np.sum(center_finite & ~ocean_mask))
                finite_mask &= ocean_mask
            basin_inputs[basin_name] = (tc_flag, vmax, finite_mask)

        for threshold in thresholds:
            all_total = 0.0
            all_active_steps = 0
            any_finite = False
            for basin_name in BASIN_ORDER:
                tc_flag, vmax, finite_mask = basin_inputs[basin_name]
                active = finite_mask & (tc_flag >= 0.5) & (vmax >= threshold)
                active_steps = int(np.sum(active))
                total_ace = float(np.nansum((vmax[active] ** 2.0) * scale[active])) if np.any(active) else 0.0
                all_total += total_ace
                all_active_steps += active_steps
                any_finite = any_finite or np.any(finite_mask)
                rows.append(
                    {
                        "year": year,
                        "init_date": init_date,
                        "ens": ens,
                        "basin_name": basin_name,
                        "threshold_kt": threshold,
                        "geos_member_ace": total_ace,
                        "active_steps": active_steps,
                        "cache_file": path.name,
                    }
                )
            rows.append(
                {
                    "year": year,
                    "init_date": init_date,
                    "ens": ens,
                    "basin_name": "All Basins",
                    "threshold_kt": threshold,
                    "geos_member_ace": all_total if any_finite else float("nan"),
                    "active_steps": all_active_steps if any_finite else "",
                    "cache_file": path.name,
                }
            )
    if geos_land_filtered_steps:
        print(f"Skipped {geos_land_filtered_steps:,} GEOS basin-time samples over land across threshold passes.")
    return rows


def aggregate_yearly(member_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, float], list[float]] = defaultdict(list)
    active_grouped: dict[tuple[int, str, float], list[float]] = defaultdict(list)
    members: dict[tuple[int, str, float], set[tuple[str, str]]] = defaultdict(set)
    for row in member_rows:
        key = (int(row["year"]), str(row["basin_name"]), float(row["threshold_kt"]))
        value = float(row["geos_member_ace"])
        if np.isfinite(value):
            grouped[key].append(value)
            if row.get("active_steps") != "":
                active_grouped[key].append(float(row["active_steps"]))
            members[key].add((str(row["init_date"]), str(row["ens"])))

    rows: list[dict[str, object]] = []
    for (year, basin_name, threshold), values in sorted(grouped.items()):
        array = np.asarray(values, dtype="float64")
        active_array = np.asarray(active_grouped.get((year, basin_name, threshold), []), dtype="float64")
        rows.append(
            {
                "year": year,
                "basin_name": basin_name,
                "threshold_kt": threshold,
                "geos_mean_ace": float(np.nanmean(array)),
                "geos_std_ace": float(np.nanstd(array)) if array.size > 1 else 0.0,
                "geos_mean_active_steps": float(np.nanmean(active_array)) if active_array.size else float("nan"),
                "geos_std_active_steps": float(np.nanstd(active_array)) if active_array.size > 1 else 0.0,
                "n_members": int(array.size),
                "member_ids": ";".join(f"{init}:{ens}" for init, ens in sorted(members[(year, basin_name, threshold)])),
            }
        )
    return rows


def add_observed_ace(rows: list[dict[str, object]], args: argparse.Namespace, months: set[str]) -> None:
    years = sorted({int(row["year"]) for row in rows})
    obs_ace, obs_counts = read_ibtracs_observed_ace(
        Path(args.ibtracs),
        years=set(years),
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
    for row in rows:
        year = int(row["year"])
        basin_name = str(row["basin_name"])
        if basin_name == "All Basins":
            obs = sum(float(obs_ace.get((year, basin), 0.0)) for basin in BASIN_ORDER)
            fixes = sum(int(obs_counts.get((year, basin), 0)) for basin in BASIN_ORDER)
        else:
            obs = float(obs_ace.get((year, basin_name), 0.0))
            fixes = int(obs_counts.get((year, basin_name), 0))
        geos = float(row["geos_mean_ace"])
        row["ibtracs_ace"] = obs
        row["ibtracs_fix_count"] = fixes
        row["geos_to_ibtracs_ratio"] = geos / obs if obs > 0.0 and np.isfinite(geos) else float("nan")
        geos_count = float(row.get("geos_mean_active_steps", np.nan))
        row["geos_to_ibtracs_count_ratio"] = geos_count / fixes if fixes > 0 and np.isfinite(geos_count) else float("nan")
        row["raw_bias_ace"] = geos - obs if np.isfinite(obs) and np.isfinite(geos) else float("nan")


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if float(np.nanstd(x_valid)) == 0.0 or float(np.nanstd(y_valid)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def summarize_thresholds(yearly_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in yearly_rows:
        grouped[(str(row["basin_name"]), float(row["threshold_kt"]))].append(row)

    rows: list[dict[str, object]] = []
    for (basin_name, threshold), group_rows in sorted(grouped.items()):
        geos = np.asarray([float(row["geos_mean_ace"]) for row in group_rows], dtype="float64")
        obs = np.asarray([float(row.get("ibtracs_ace", np.nan)) for row in group_rows], dtype="float64")
        geos_count = np.asarray([float(row.get("geos_mean_active_steps", np.nan)) for row in group_rows], dtype="float64")
        obs_count = np.asarray([float(row.get("ibtracs_fix_count", np.nan)) for row in group_rows], dtype="float64")
        mask = np.isfinite(geos) & np.isfinite(obs)
        diff = geos[mask] - obs[mask]
        geos_mean = float(np.nanmean(geos[mask])) if int(np.sum(mask)) else float("nan")
        obs_mean = float(np.nanmean(obs[mask])) if int(np.sum(mask)) else float("nan")
        count_mask = np.isfinite(geos_count) & np.isfinite(obs_count)
        geos_count_mean = float(np.nanmean(geos_count[count_mask])) if int(np.sum(count_mask)) else float("nan")
        obs_count_mean = float(np.nanmean(obs_count[count_mask])) if int(np.sum(count_mask)) else float("nan")
        rows.append(
            {
                "basin_name": basin_name,
                "threshold_kt": threshold,
                "n_years": int(np.sum(mask)),
                "geos_mean_ace": geos_mean,
                "ibtracs_mean_ace": obs_mean,
                "geos_to_ibtracs_ratio": geos_mean / obs_mean if obs_mean > 0.0 and np.isfinite(geos_mean) else float("nan"),
                "geos_mean_active_steps": geos_count_mean,
                "ibtracs_mean_fix_count": obs_count_mean,
                "geos_to_ibtracs_count_ratio": (
                    geos_count_mean / obs_count_mean
                    if obs_count_mean > 0.0 and np.isfinite(geos_count_mean)
                    else float("nan")
                ),
                "raw_bias": float(np.nanmean(diff)) if diff.size else float("nan"),
                "rmse": float(np.sqrt(np.nanmean(diff**2))) if diff.size else float("nan"),
                "corr": correlation(geos, obs),
            }
        )
    return rows


def best_threshold_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["basin_name"])].append(row)

    rows: list[dict[str, object]] = []
    for basin_name, group_rows in sorted(grouped.items()):
        finite = [
            row
            for row in group_rows
            if np.isfinite(float(row.get("geos_to_ibtracs_ratio", np.nan))) and float(row.get("geos_to_ibtracs_ratio", np.nan)) > 0.0
        ]
        if not finite:
            continue
        best = min(finite, key=lambda row: abs(np.log(float(row["geos_to_ibtracs_ratio"]))))
        rows.append(dict(best))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"  -> Wrote {path}")


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    print(f"  -> Read {path}")
    return rows


MEMBER_FIELDS = ["year", "init_date", "ens", "basin_name", "threshold_kt", "geos_member_ace", "active_steps", "cache_file"]
YEARLY_FIELDS = [
    "year",
    "basin_name",
    "threshold_kt",
    "geos_mean_ace",
    "geos_std_ace",
    "geos_mean_active_steps",
    "geos_std_active_steps",
    "n_members",
    "ibtracs_ace",
    "ibtracs_fix_count",
    "geos_to_ibtracs_ratio",
    "geos_to_ibtracs_count_ratio",
    "raw_bias_ace",
    "member_ids",
]
SUMMARY_FIELDS = [
    "basin_name",
    "threshold_kt",
    "n_years",
    "geos_mean_ace",
    "ibtracs_mean_ace",
    "geos_to_ibtracs_ratio",
    "geos_mean_active_steps",
    "ibtracs_mean_fix_count",
    "geos_to_ibtracs_count_ratio",
    "raw_bias",
    "rmse",
    "corr",
]


def table_paths(output_dir: Path, prefix: str) -> tuple[Path, Path, Path, Path]:
    return (
        output_dir / f"{prefix}_member_values.csv",
        output_dir / f"{prefix}_yearly.csv",
        output_dir / f"{prefix}_summary.csv",
        output_dir / f"{prefix}_best_thresholds.csv",
    )


def plot_ratio_curves(summary_rows: list[dict[str, object]], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=dpi)
    for basin_name in BASIN_ORDER + ["All Basins"]:
        rows = sorted([row for row in summary_rows if row["basin_name"] == basin_name], key=lambda row: float(row["threshold_kt"]))
        if not rows:
            continue
        thresholds = np.asarray([float(row["threshold_kt"]) for row in rows], dtype="float64")
        ratios = np.asarray([float(row["geos_to_ibtracs_ratio"]) for row in rows], dtype="float64")
        if basin_name == "All Basins":
            color = "#1e222a"
            linewidth = 2.6
            alpha = 1.0
        else:
            color = BASINS[basin_name]["color"]
            linewidth = 1.7
            alpha = 0.9
        ax.plot(thresholds, ratios, marker="o", markersize=3.5, linewidth=linewidth, color=color, alpha=alpha, label=basin_name)
    ax.axhline(1.0, color="#2a2d34", linewidth=1.2, linestyle="--")
    ax.set_xlabel("GEOS ACE wind threshold (kt)")
    ax.set_ylabel("Mean GEOS / IBTrACS ACE")
    ax.set_title("ACE Amplitude Sensitivity to GEOS Wind Threshold", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_best_thresholds(best_rows: list[dict[str, object]], path: Path, dpi: int) -> None:
    rows = [row for row in best_rows if row["basin_name"] != "All Basins"]
    basins = [str(row["basin_name"]) for row in rows]
    thresholds = np.asarray([float(row["threshold_kt"]) for row in rows], dtype="float64")
    ratios = np.asarray([float(row["geos_to_ibtracs_ratio"]) for row in rows], dtype="float64")
    colors = [BASINS[basin]["color"] for basin in basins]

    fig, ax = plt.subplots(figsize=(11.5, 5.5), dpi=dpi)
    bars = ax.bar(np.arange(len(basins)), thresholds, color=colors)
    ax.set_xticks(np.arange(len(basins)))
    ax.set_xticklabels(basins, rotation=28, ha="right")
    ax.set_ylabel("Best amplitude-matched threshold (kt)")
    ax.set_title("Best GEOS Threshold by Basin", fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ratio:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", default="geos_ace_threshold_sensitivity")
    parser.add_argument("--years", default="1991:2024")
    parser.add_argument("--init-dates", default="0824,0829")
    parser.add_argument("--months", default="09:10")
    parser.add_argument("--geos-thresholds", default="5,8,10,12,15,17,20,22,25,28,30,35,40")
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS)
    parser.add_argument("--wind-var", default="usa_wind")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument("--nature-filter", default="TS")
    parser.add_argument("--all-natures", dest="nature_filter", action="store_const", const="")
    parser.add_argument("--basin-method", choices=("boxes", "ibtracs_code"), default="boxes")
    parser.add_argument("--all-hours", dest="synoptic_only", action="store_false")
    parser.add_argument("--use-cached-tables", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.set_defaults(synoptic_only=True)
    add_ocean_only_args(parser)
    args = parser.parse_args(argv)

    setup_style()
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    member_path, yearly_path, summary_path, best_path = table_paths(output_dir, args.prefix)

    if args.use_cached_tables:
        if not summary_path.exists() or not best_path.exists():
            print(f"ERROR: cached threshold tables are missing under {output_dir}", file=sys.stderr)
            return 1
        summary_rows = read_csv(summary_path)
        best_rows = read_csv(best_path)
    else:
        years = parse_years(args.years)
        months = parse_months(args.months)
        thresholds = parse_float_list(args.geos_thresholds)
        ocean_checker = build_geos_ocean_checker(args)
        member_rows = read_geos_member_rows(Path(args.cache_dir), years, parse_list(args.init_dates), months, thresholds, ocean_checker)
        yearly_rows = aggregate_yearly(member_rows)
        add_observed_ace(yearly_rows, args, months)
        summary_rows = summarize_thresholds(yearly_rows)
        best_rows = best_threshold_rows(summary_rows)
        write_csv(member_path, member_rows, MEMBER_FIELDS)
        write_csv(yearly_path, yearly_rows, YEARLY_FIELDS)
        write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
        write_csv(best_path, best_rows, SUMMARY_FIELDS)

    plot_ratio_curves(summary_rows, plot_dir / f"{args.prefix}_ratio_curves.png", args.dpi)
    plot_best_thresholds(best_rows, plot_dir / f"{args.prefix}_best_thresholds.png", args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
