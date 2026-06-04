#!/usr/bin/env python3
"""Plot yearly ACE time series from cached TC-conditioned ACE NetCDF files."""

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

from calculate_tc_conditioned_ace import BASINS, read_cache, thresholds_from_diagnostics


BASIN_ORDER = list(BASINS)
DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_PLOT_DIR = "plots/ace_yearly_timeseries"


def parse_years(value: str) -> set[int]:
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
    return years


def cache_year_and_kind(path: Path) -> tuple[int | None, str]:
    lagged = re.match(r"tc_conditioned_ace_(\d{4})_lagged_ensmean\.nc4$", path.name)
    if lagged:
        return int(lagged.group(1)), "lagged"
    init = re.match(r"tc_conditioned_ace_(\d{8})_ensmean\.nc4$", path.name)
    if init:
        return int(init.group(1)[:4]), "init"
    return None, "other"


def discover_cache_files(cache_dir: Path, years: set[int], cache_kind: str) -> dict[int, list[Path]]:
    files_by_year: dict[int, dict[str, list[Path]]] = defaultdict(lambda: {"lagged": [], "init": [], "other": []})
    for path in sorted(cache_dir.glob("tc_conditioned_ace_*_ensmean.nc4")):
        year, kind = cache_year_and_kind(path)
        if year is None:
            continue
        if years and year not in years:
            continue
        files_by_year[year][kind].append(path)

    selected: dict[int, list[Path]] = {}
    for year, groups in sorted(files_by_year.items()):
        if cache_kind == "auto":
            selected[year] = groups["lagged"] if groups["lagged"] else groups["init"]
        elif cache_kind == "lagged":
            selected[year] = groups["lagged"]
        elif cache_kind == "init":
            selected[year] = groups["init"]
        else:
            selected[year] = groups["lagged"] + groups["init"] + groups["other"]
        selected[year] = sorted(selected[year])
    return {year: paths for year, paths in selected.items() if paths}


def setup_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "#ffffff"


def summarize_caches(files_by_year: dict[int, list[Path]]) -> tuple[list[dict[str, object]], dict[int, list[Path]]]:
    rows: list[dict[str, object]] = []
    used_files: dict[int, list[Path]] = {}
    for year, paths in sorted(files_by_year.items()):
        basin_values: dict[str, list[float]] = defaultdict(list)
        basin_thresholds: dict[str, list[float]] = defaultdict(list)
        good_paths: list[Path] = []
        for path in paths:
            try:
                _, _, _, _, _, _, diagnostics, _, _ = read_cache(path)
            except Exception as exc:
                print(f"WARNING: skipping unreadable cache {path}: {exc}")
                continue
            good_paths.append(path)
            thresholds = thresholds_from_diagnostics(diagnostics)
            for basin_name in BASIN_ORDER:
                curve = np.asarray(diagnostics[basin_name]["cumulative_ace"], dtype="float64")
                basin_values[basin_name].append(float(curve[-1]) if curve.size else float("nan"))
                if basin_name in thresholds:
                    basin_thresholds[basin_name].append(float(thresholds[basin_name]))

        if not good_paths:
            continue
        used_files[year] = good_paths
        total_values = []
        for cache_index in range(len(good_paths)):
            total = 0.0
            for basin_name in BASIN_ORDER:
                values = basin_values[basin_name]
                if cache_index < len(values) and np.isfinite(values[cache_index]):
                    total += values[cache_index]
            total_values.append(total)

        for basin_name in BASIN_ORDER:
            values = np.asarray(basin_values[basin_name], dtype="float64")
            values = values[np.isfinite(values)]
            thresholds = np.asarray(basin_thresholds[basin_name], dtype="float64")
            thresholds = thresholds[np.isfinite(thresholds)]
            rows.append(
                {
                    "year": year,
                    "basin_name": basin_name,
                    "mean_ace": float(np.nanmean(values)) if values.size else float("nan"),
                    "std_ace": float(np.nanstd(values)) if values.size > 1 else 0.0,
                    "n_caches": int(values.size),
                    "threshold_kt": float(np.nanmean(thresholds)) if thresholds.size else float("nan"),
                    "cache_files": ",".join(path.name for path in good_paths),
                }
            )
        total_array = np.asarray(total_values, dtype="float64")
        rows.append(
            {
                "year": year,
                "basin_name": "All Basins",
                "mean_ace": float(np.nanmean(total_array)) if total_array.size else float("nan"),
                "std_ace": float(np.nanstd(total_array)) if total_array.size > 1 else 0.0,
                "n_caches": int(total_array.size),
                "threshold_kt": float("nan"),
                "cache_files": ",".join(path.name for path in good_paths),
            }
        )
    return rows, used_files


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["year", "basin_name", "mean_ace", "std_ace", "n_caches", "threshold_kt", "cache_files"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  -> Wrote {path}")


def rows_for_basin(rows: list[dict[str, object]], basin_name: str) -> list[dict[str, object]]:
    return sorted([row for row in rows if row["basin_name"] == basin_name], key=lambda row: int(row["year"]))


def plot_basin_panels(rows: list[dict[str, object]], path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=dpi, sharex=True)
    for ax, basin_name in zip(axes.ravel(), BASIN_ORDER):
        basin_rows = rows_for_basin(rows, basin_name)
        years = np.asarray([int(row["year"]) for row in basin_rows], dtype="int32")
        mean_values = np.asarray([float(row["mean_ace"]) for row in basin_rows], dtype="float64")
        std_values = np.asarray([float(row["std_ace"]) for row in basin_rows], dtype="float64")
        color = BASINS[basin_name]["color"]
        ax.plot(years, mean_values, marker="o", markersize=3.5, linewidth=1.7, color=color)
        if np.any(std_values > 0):
            ax.fill_between(years, mean_values - std_values, mean_values + std_values, color=color, alpha=0.15, linewidth=0)
        ax.set_title(basin_name, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Yearly TC-Conditioned ACE by Basin", fontsize=13, fontweight="bold", y=0.99)
    fig.supxlabel("Initialization Year")
    fig.supylabel("ACE (10^4 kt^2)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_total(rows: list[dict[str, object]], path: Path, dpi: int) -> None:
    total_rows = rows_for_basin(rows, "All Basins")
    years = np.asarray([int(row["year"]) for row in total_rows], dtype="int32")
    mean_values = np.asarray([float(row["mean_ace"]) for row in total_rows], dtype="float64")
    std_values = np.asarray([float(row["std_ace"]) for row in total_rows], dtype="float64")
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=dpi)
    ax.plot(years, mean_values, marker="o", markersize=4, linewidth=2.0, color="#344e86")
    if np.any(std_values > 0):
        ax.fill_between(years, mean_values - std_values, mean_values + std_values, color="#344e86", alpha=0.15, linewidth=0)
    ax.set_title("Yearly All-Basin TC-Conditioned ACE", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Initialization Year")
    ax.set_ylabel("ACE (10^4 kt^2)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--years", default="", help="Optional years, e.g. 1991:2024 or 1991,1992.")
    parser.add_argument("--cache-kind", choices=("auto", "lagged", "init", "all"), default="auto")
    parser.add_argument("--prefix", default="ace_yearly_timeseries")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    setup_style()
    years = parse_years(args.years) if args.years else set()
    cache_dir = Path(args.cache_dir)
    plot_dir = Path(args.plot_dir)
    files_by_year = discover_cache_files(cache_dir, years, args.cache_kind)
    if not files_by_year:
        print(f"ERROR: no ACE ensmean caches found in {cache_dir}", file=sys.stderr)
        return 1

    print("Using ACE cache files by year:")
    for year, paths in files_by_year.items():
        print(f"  {year}: {', '.join(path.name for path in paths)}")

    rows, used_files = summarize_caches(files_by_year)
    used_years = sorted(used_files)
    if years:
        missing = sorted(years - set(used_years))
        if missing:
            print(f"Missing requested years with usable caches: {', '.join(str(year) for year in missing)}")
    print(f"Used years: {', '.join(str(year) for year in used_years)}")

    write_csv(plot_dir / f"{args.prefix}.csv", rows)
    plot_basin_panels(rows, plot_dir / f"{args.prefix}_by_basin.png", args.dpi)
    plot_total(rows, plot_dir / f"{args.prefix}_all_basins.png", args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
