#!/usr/bin/env python3
"""Calculate GEOS basin wind thresholds from an existing candidate inventory.

This is a lightweight post-processing helper for cases where the expensive
candidate inventory is already cached. It filters the inventory by init-year
and forecast month, then maps observed IBTrACS percentiles onto the selected
GEOS candidate Vmax distributions.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from candidate_inventory_defaults import CANONICAL_ALL_CENTERS_CANDIDATES
from calculate_geos_candidate_thresholds import BASINS, finite_percentile, load_observed_percentiles, summarize_values


DEFAULT_OBS_PERCENTILES = "data/obs/ibtracs/ibtracs_observed_percentiles.csv"
DEFAULT_OUTPUT = "data/calibration/geos_candidate_thresholds_from_inventory.csv"


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def parse_years(value: str) -> set[int]:
    years: set[int] = set()
    for item in parse_list(value):
        if ":" in item:
            start_text, end_text = item.split(":", 1)
            years.update(range(int(start_text), int(end_text) + 1))
        elif "-" in item:
            start_text, end_text = item.split("-", 1)
            years.update(range(int(start_text), int(end_text) + 1))
        else:
            years.add(int(item))
    return years


def parse_months(value: str) -> set[str]:
    months: set[str] = set()
    for item in parse_list(value):
        if ":" in item:
            start_text, end_text = item.split(":", 1)
            months.update(f"{month:02d}" for month in range(int(start_text), int(end_text) + 1))
        elif "-" in item:
            start_text, end_text = item.split("-", 1)
            months.update(f"{month:02d}" for month in range(int(start_text), int(end_text) + 1))
        else:
            months.add(f"{int(item):02d}")
    return months


def parse_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def parse_init_year(row: dict[str, str]) -> int | None:
    text = row.get("init_date", "")
    if len(text) < 4:
        return None
    try:
        return int(text[:4])
    except ValueError:
        return None


def is_accepted(row: dict[str, str]) -> bool:
    value = row.get("accepted_candidate", "")
    return value in {"", "1", "1.0", "true", "True", "TRUE"}


def inventory_schema(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        return "unknown"
    if "accepted_candidate" in fieldnames and "rejection_reason" in fieldnames:
        return "all-centers"
    return "accepted-only"


def collect_geos_values(args: argparse.Namespace) -> tuple[dict[str, list[float]], dict[str, object]]:
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidate inventory does not exist: {candidates_path}")

    years = parse_years(args.years)
    months = parse_months(args.months)
    values_by_basin: dict[str, list[float]] = {basin_name: [] for basin_name in BASINS}
    counters: Counter[str] = Counter()
    used_init_dates: set[str] = set()
    used_ensembles: set[str] = set()
    used_months: set[str] = set()

    with candidates_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        schema = inventory_schema(reader.fieldnames)
        for row in reader:
            counters["rows_read"] += 1
            basin_name = row.get("basin_name", "")
            if basin_name not in BASINS:
                counters["skipped_basin"] += 1
                continue

            init_year = parse_init_year(row)
            if init_year not in years:
                counters["skipped_year"] += 1
                continue

            month = row.get("forecast_month", "")
            try:
                month = f"{int(float(month)):02d}"
            except ValueError:
                month = month.zfill(2)
            if months and month not in months:
                counters["skipped_month"] += 1
                continue

            if not args.include_rejected and not is_accepted(row):
                counters["skipped_rejected"] += 1
                continue

            center_lat = parse_float(row.get("center_lat"))
            if np.isfinite(center_lat):
                if center_lat < args.min_lat or center_lat > args.max_lat:
                    counters["skipped_lat"] += 1
                    continue

            vmax = parse_float(row.get("vmax_kt"))
            if not np.isfinite(vmax):
                counters["skipped_vmax"] += 1
                continue

            values_by_basin[basin_name].append(float(vmax))
            counters["rows_used"] += 1
            if is_accepted(row):
                counters["accepted_rows_used"] += 1
            else:
                counters["rejected_rows_used"] += 1
            used_init_dates.add(row.get("init_date", ""))
            used_ensembles.add(row.get("ens", ""))
            used_months.add(month)

    metadata: dict[str, object] = {
        "schema": schema,
        "counters": counters,
        "used_init_dates": sorted(item for item in used_init_dates if item),
        "used_ensembles": sorted(item for item in used_ensembles if item),
        "used_months": sorted(item for item in used_months if item),
    }
    return values_by_basin, metadata


def write_thresholds(
    output_path: Path,
    values_by_basin: dict[str, list[float]],
    observed_rows: dict[tuple[str, str], dict[str, float | int | str]],
    observed_wind_vars: list[str],
    args: argparse.Namespace,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        "candidate_source",
        "candidate_schema",
        "calibration_years",
        "months",
        "include_rejected",
        "min_lat",
        "max_lat",
        "init_dates",
        "ensembles",
        "rows_read",
        "rows_used",
        "accepted_rows_used",
        "rejected_rows_used",
        "skipped_year",
        "skipped_month",
        "skipped_rejected",
        "skipped_lat",
        "skipped_vmax",
    ]
    counters: Counter[str] = metadata["counters"]  # type: ignore[assignment]
    used_init_dates: list[str] = metadata["used_init_dates"]  # type: ignore[assignment]
    used_ensembles: list[str] = metadata["used_ensembles"]  # type: ignore[assignment]
    used_months: list[str] = metadata["used_months"]  # type: ignore[assignment]

    rows: list[dict[str, object]] = []
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for basin_name in BASINS:
            stats = summarize_values(values_by_basin[basin_name])
            for wind_var in observed_wind_vars:
                obs = observed_rows.get((basin_name, wind_var))
                if obs is None:
                    observed_percentile = float("nan")
                    geos_threshold = float("nan")
                    obs_values = {
                        "observed_basin_method": args.observed_basin_method,
                        "observed_threshold_kt": float("nan"),
                        "observed_p": float("nan"),
                        "observed_percentile": float("nan"),
                        "observed_n_samples": 0,
                    }
                else:
                    observed_percentile = float(obs["observed_percentile"])
                    geos_threshold = finite_percentile(values_by_basin[basin_name], observed_percentile)
                    obs_values = obs

                row = {
                    "basin_name": basin_name,
                    "observed_wind_var": wind_var,
                    "observed_basin_method": obs_values["observed_basin_method"],
                    "observed_threshold_kt": obs_values["observed_threshold_kt"],
                    "observed_p": obs_values["observed_p"],
                    "observed_percentile": obs_values["observed_percentile"],
                    "observed_n_samples": obs_values["observed_n_samples"],
                    "geos_threshold_kt": geos_threshold,
                    **stats,
                    "candidate_source": str(args.candidates),
                    "candidate_schema": metadata["schema"],
                    "calibration_years": args.years,
                    "months": ",".join(used_months),
                    "include_rejected": int(args.include_rejected),
                    "min_lat": args.min_lat,
                    "max_lat": args.max_lat,
                    "init_dates": ",".join(used_init_dates),
                    "ensembles": ",".join(used_ensembles),
                    "rows_read": counters["rows_read"],
                    "rows_used": counters["rows_used"],
                    "accepted_rows_used": counters["accepted_rows_used"],
                    "rejected_rows_used": counters["rejected_rows_used"],
                    "skipped_year": counters["skipped_year"],
                    "skipped_month": counters["skipped_month"],
                    "skipped_rejected": counters["skipped_rejected"],
                    "skipped_lat": counters["skipped_lat"],
                    "skipped_vmax": counters["skipped_vmax"],
                }
                rows.append(row)
                writer.writerow(row)
    return rows


def print_threshold_table(rows: list[dict[str, object]], output_path: Path) -> None:
    print(f"GEOS inventory-derived basin thresholds: {output_path}")
    print(f"{'basin':22s} {'obs_p':>8s} {'obs_pct':>8s} {'n_geos':>10s} {'T_geos':>8s} {'median':>8s} {'p90':>8s}")
    for row in rows:
        print(
            f"{str(row['basin_name']):22s} "
            f"{float(row['observed_p']):8.3f} "
            f"{float(row['observed_percentile']):8.2f} "
            f"{int(float(row['geos_n_candidates'])):10d} "
            f"{float(row['geos_threshold_kt']):8.2f} "
            f"{float(row['geos_median_vmax_kt']):8.2f} "
            f"{float(row['geos_p90_vmax_kt']):8.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates", default=CANONICAL_ALL_CENTERS_CANDIDATES)
    parser.add_argument("--years", default="1991:2022")
    parser.add_argument("--months", default="09:10")
    parser.add_argument("--observed-percentiles", default=DEFAULT_OBS_PERCENTILES)
    parser.add_argument("--observed-wind-vars", default="usa_wind")
    parser.add_argument("--observed-basin-method", default="boxes")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--include-rejected", action="store_true", help="Use all written candidates, not only accepted_candidate=1.")
    parser.add_argument("--min-lat", type=float, default=-25.0)
    parser.add_argument("--max-lat", type=float, default=50.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observed_wind_vars = parse_list(args.observed_wind_vars)
    observed_rows = load_observed_percentiles(
        Path(args.observed_percentiles),
        observed_wind_vars,
        args.observed_basin_method,
    )
    if not observed_rows:
        print("ERROR: no matching observed percentile rows were found", file=sys.stderr)
        return 2

    values_by_basin, metadata = collect_geos_values(args)
    output_path = Path(args.output)
    rows = write_thresholds(output_path, values_by_basin, observed_rows, observed_wind_vars, args, metadata)
    print_threshold_table(rows, output_path)

    counters: Counter[str] = metadata["counters"]  # type: ignore[assignment]
    used_init_dates: list[str] = metadata["used_init_dates"]  # type: ignore[assignment]
    print("")
    print(f"Read candidate rows: {counters['rows_read']:,}")
    print(f"Used candidate rows: {counters['rows_used']:,}")
    print(f"Used init dates: {len(used_init_dates)} ({used_init_dates[0] if used_init_dates else 'none'} to {used_init_dates[-1] if used_init_dates else 'none'})")
    print(f"Wrote threshold CSV: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
