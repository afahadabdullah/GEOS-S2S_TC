#!/usr/bin/env python3
"""Compare candidate-based GEOS ACE aggregation methods against IBTrACS.

This is a fast exploratory bake-off. It reads cached GEOS candidate CSV files,
tests several ways to aggregate one or more candidate centers into ACE, and
ranks method/threshold combinations against IBTrACS ACE climatology for a small
set of years.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
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

from calculate_tc_conditioned_ace import BASINS
from ocean_mask_utils import add_ocean_only_args, build_ocean_checker, row_over_ocean_value
from plot_ace_yearly_timeseries import (
    BASIN_ORDER,
    DEFAULT_IBTRACS,
    parse_list,
    parse_months,
    parse_years,
    read_ibtracs_observed_ace,
    setup_style,
)


DEFAULT_CANDIDATES = "data/calibration/*_candidates.csv"
DEFAULT_OUTPUT_DIR = "data/analysis/ace_method_test"
DEFAULT_PLOT_DIR = "plots/ace_method_test"
DEFAULT_METHODS = (
    "sep5_vmax,sum_all,"
    "structure+sep5_vmax,structure+sum_all,"
    "slp_only+sep5_vmax,slp_warm+sep5_vmax,slp_qv+sep5_vmax,"
    "slp_warm_qv+sep5_vmax,slp_warm_qv_vort+sep5_vmax,"
    "slp_only+sum_all,slp_warm_qv+sum_all"
)
DEFAULT_THRESHOLDS = "0,5,8,10,12,15,17,20,22,25,28,30,35,40"
ACE_SCALE_6H = 1.0e-4


@dataclass(frozen=True)
class Candidate:
    init_date: str
    ens: str
    year: int
    month: str
    valid_time: datetime
    time_key: str
    basin_name: str
    center_lat: float
    center_lon: float
    vmax_kt: float
    slp_hpa: float
    over_ocean: bool | None
    passes_slp_anom: int
    passes_warm_core: int
    passes_qv: int
    passes_vorticity: int
    passes_structure: int
    accepted_candidate: int


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def parse_flag(value: str | None, default: int = 1) -> int:
    if value is None or value == "":
        return default
    return int(str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y"})


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def normalize_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def parse_float_list(value: str) -> list[float]:
    return sorted({float(item) for item in parse_list(value)})


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for match in sorted(matches):
            path = Path(match)
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def init_is_selected(init_date: str, selected: list[str]) -> bool:
    if not selected:
        return True
    for item in selected:
        if len(item) == 4 and init_date.endswith(item):
            return True
        if item == init_date:
            return True
    return False


def ens_is_selected(ens: str, selected: list[str]) -> bool:
    if not selected or selected == ["all"]:
        return True
    return ens in selected


def build_candidate_ocean_checker(args: argparse.Namespace):
    if not args.ocean_only:
        return None
    checker, warning = build_ocean_checker(
        args.ocean_mask_source,
        mask_file=args.ocean_mask_file,
        threshold=args.ocean_threshold,
        require_mask=True,
    )
    print(f"Ocean-only GEOS candidate filter enabled: source={checker.source}")
    if warning:
        print(f"WARNING: GEOS ocean mask fallback: {warning}")
    return checker


def read_candidates(paths: list[Path], args: argparse.Namespace) -> list[Candidate]:
    years = parse_years(args.years)
    months = parse_months(args.months)
    init_dates = parse_list(args.init_dates)
    ensembles = parse_list(args.ens)
    ocean_checker = None
    rows: list[Candidate] = []
    skipped = 0
    skipped_land = 0

    print("Reading GEOS candidate CSV files:")
    for path in paths:
        print(f"  - {path}")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                basin_name = raw.get("basin_name", "")
                if basin_name not in BASINS:
                    skipped += 1
                    continue
                init_date = raw.get("init_date", "")
                ens = raw.get("ens", "")
                if not init_is_selected(init_date, init_dates) or not ens_is_selected(ens, ensembles):
                    continue
                valid_time = parse_datetime(raw.get("valid_time"))
                if valid_time is None:
                    skipped += 1
                    continue
                year = valid_time.year
                month = valid_time.strftime("%m")
                if years and year not in years:
                    continue
                if months and month not in months:
                    continue

                center_lat = parse_float(raw.get("center_lat"))
                center_lon = normalize_lon(parse_float(raw.get("center_lon")))
                vmax_kt = parse_float(raw.get("vmax_kt"))
                slp_hpa = parse_float(raw.get("slp_hpa"))
                passes_slp_anom = parse_flag(raw.get("passes_slp_anom"), default=1)
                passes_warm_core = parse_flag(raw.get("passes_warm_core"), default=1)
                passes_qv = parse_flag(raw.get("passes_qv"), default=1)
                passes_vorticity = parse_flag(raw.get("passes_vorticity"), default=1)
                passes_structure = parse_flag(raw.get("passes_structure"), default=1)
                accepted_candidate = parse_flag(raw.get("accepted_candidate"), default=1)
                if not (np.isfinite(center_lat) and np.isfinite(center_lon) and np.isfinite(vmax_kt)):
                    skipped += 1
                    continue
                if not (args.min_lat <= center_lat <= args.max_lat):
                    continue

                over_ocean = row_over_ocean_value(raw)
                if args.ocean_only:
                    if over_ocean is None:
                        if ocean_checker is None:
                            ocean_checker = build_candidate_ocean_checker(args)
                        is_ocean = ocean_checker.is_ocean(center_lat, center_lon)
                    else:
                        is_ocean = over_ocean
                    if not is_ocean:
                        skipped_land += 1
                        continue

                rows.append(
                    Candidate(
                        init_date=init_date,
                        ens=ens,
                        year=year,
                        month=month,
                        valid_time=valid_time,
                        time_key=valid_time.strftime("%Y-%m-%d %H:%M:%S"),
                        basin_name=basin_name,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        vmax_kt=vmax_kt,
                        slp_hpa=slp_hpa,
                        over_ocean=over_ocean,
                        passes_slp_anom=passes_slp_anom,
                        passes_warm_core=passes_warm_core,
                        passes_qv=passes_qv,
                        passes_vorticity=passes_vorticity,
                        passes_structure=passes_structure,
                        accepted_candidate=accepted_candidate,
                    )
                )

    if skipped:
        print(f"Skipped {skipped:,} incomplete/unrecognized GEOS candidate rows.")
    if skipped_land:
        print(f"Skipped {skipped_land:,} GEOS candidate rows over land.")
    return rows


def print_candidate_coverage(candidates: list[Candidate], requested_years: set[int]) -> list[int]:
    years = sorted({candidate.year for candidate in candidates})
    missing = sorted(requested_years - set(years))
    init_dates = sorted({candidate.init_date for candidate in candidates})
    members_by_year: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for candidate in candidates:
        members_by_year[candidate.year].add((candidate.init_date, candidate.ens))

    if years:
        print(f"GEOS candidate years used: {', '.join(str(year) for year in years)}")
    else:
        print("GEOS candidate years used: none")
    if missing:
        print(f"Missing requested years after GEOS filters: {', '.join(str(year) for year in missing)}")
    if init_dates:
        print(f"Initialization dates used: {', '.join(init_dates)}")
    for year in years:
        print(f"  {year}: {len(members_by_year[year])} active init/member pair(s)")
    return years


def angular_distance_deg(a: Candidate, b: Candidate) -> float:
    lat1 = np.deg2rad(a.center_lat)
    lat2 = np.deg2rad(b.center_lat)
    dlat = lat2 - lat1
    dlon = np.deg2rad(normalize_lon(b.center_lon - a.center_lon))
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(np.rad2deg(2.0 * np.arcsin(min(1.0, np.sqrt(h)))))


def split_method(method: str) -> tuple[str, str]:
    if "+" in method:
        gate, aggregation = method.split("+", 1)
        return gate.strip(), aggregation.strip()
    return "accepted", method


def gate_passes(candidate: Candidate, gate: str) -> bool:
    if gate in {"accepted", "current", "current_all"}:
        return bool(candidate.accepted_candidate)
    if gate in {"structure", "structure_all"}:
        return bool(candidate.passes_structure)
    if gate == "slp_only":
        return bool(candidate.passes_slp_anom)
    if gate == "slp_warm":
        return bool(candidate.passes_slp_anom and candidate.passes_warm_core)
    if gate == "slp_qv":
        return bool(candidate.passes_slp_anom and candidate.passes_qv)
    if gate in {"slp_warm_qv", "slp_warm_qv_no_vort", "no_vort"}:
        return bool(candidate.passes_slp_anom and candidate.passes_warm_core and candidate.passes_qv)
    if gate in {"slp_warm_qv_vort", "with_vort"}:
        return bool(
            candidate.passes_slp_anom
            and candidate.passes_warm_core
            and candidate.passes_qv
            and candidate.passes_vorticity
        )
    raise ValueError(f"Unknown gate method: {gate}")


def select_candidates(group: list[Candidate], method: str, threshold: float) -> list[Candidate]:
    if not group:
        return []
    gate, aggregation = split_method(method)
    group = [candidate for candidate in group if gate_passes(candidate, gate)]
    if not group:
        return []

    if aggregation == "single_slp":
        finite_slp = [candidate for candidate in group if np.isfinite(candidate.slp_hpa)]
        selected = min(finite_slp, key=lambda item: item.slp_hpa) if finite_slp else max(group, key=lambda item: item.vmax_kt)
        return [selected] if selected.vmax_kt >= threshold else []

    if aggregation == "single_vmax":
        selected = max(group, key=lambda item: item.vmax_kt)
        return [selected] if selected.vmax_kt >= threshold else []

    thresholded = sorted([candidate for candidate in group if candidate.vmax_kt >= threshold], key=lambda item: item.vmax_kt, reverse=True)
    if not thresholded:
        return []

    if aggregation == "sum_all":
        return thresholded

    top_match = re.match(r"^top(\d+)_vmax$", aggregation)
    if top_match:
        return thresholded[: int(top_match.group(1))]

    sep_match = re.match(r"^sep([0-9.]+)_vmax$", aggregation)
    if sep_match:
        separation = float(sep_match.group(1))
        kept: list[Candidate] = []
        for candidate in thresholded:
            if all(angular_distance_deg(candidate, existing) >= separation for existing in kept):
                kept.append(candidate)
        return kept

    raise ValueError(f"Unknown method: {method}")


def group_candidates(candidates: list[Candidate]) -> dict[tuple[int, str, str, str, str], list[Candidate]]:
    groups: dict[tuple[int, str, str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        key = (candidate.year, candidate.init_date, candidate.ens, candidate.basin_name, candidate.time_key)
        groups[key].append(candidate)
    return groups


def active_members_by_year(candidates: list[Candidate]) -> dict[int, set[tuple[str, str]]]:
    members: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for candidate in candidates:
        members[candidate.year].add((candidate.init_date, candidate.ens))
    return members


def evaluate_geos_methods(
    candidates: list[Candidate],
    methods: list[str],
    thresholds: list[float],
) -> list[dict[str, object]]:
    groups = group_candidates(candidates)
    members_by_year = active_members_by_year(candidates)
    ace_values: dict[tuple[str, float, int, str, str, str], float] = defaultdict(float)
    active_counts: dict[tuple[str, float, int, str, str, str], int] = defaultdict(int)

    print(f"Testing {len(methods)} method(s) x {len(thresholds)} threshold(s) over {len(groups):,} basin-time groups.")
    for method in methods:
        for threshold in thresholds:
            for (year, init_date, ens, basin_name, _time_key), group in groups.items():
                selected = select_candidates(group, method, threshold)
                if not selected:
                    continue
                key = (method, threshold, year, init_date, ens, basin_name)
                ace_values[key] += float(sum(candidate.vmax_kt**2 * ACE_SCALE_6H for candidate in selected))
                active_counts[key] += len(selected)

    rows: list[dict[str, object]] = []
    for method in methods:
        for threshold in thresholds:
            for year, members in sorted(members_by_year.items()):
                for init_date, ens in sorted(members):
                    all_ace = 0.0
                    all_count = 0
                    for basin_name in BASIN_ORDER:
                        key = (method, threshold, year, init_date, ens, basin_name)
                        ace = float(ace_values.get(key, 0.0))
                        count = int(active_counts.get(key, 0))
                        all_ace += ace
                        all_count += count
                        rows.append(
                            {
                                "method": method,
                                "threshold_kt": threshold,
                                "year": year,
                                "init_date": init_date,
                                "ens": ens,
                                "basin_name": basin_name,
                                "geos_member_ace": ace,
                                "geos_active_count": count,
                            }
                        )
                    rows.append(
                        {
                            "method": method,
                            "threshold_kt": threshold,
                            "year": year,
                            "init_date": init_date,
                            "ens": ens,
                            "basin_name": "All Basins",
                            "geos_member_ace": all_ace,
                            "geos_active_count": all_count,
                        }
                    )
    return rows


def aggregate_yearly(member_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in member_rows:
        grouped[(str(row["method"]), float(row["threshold_kt"]), int(row["year"]), str(row["basin_name"]))].append(row)

    rows: list[dict[str, object]] = []
    for (method, threshold, year, basin_name), group in sorted(grouped.items()):
        ace = np.asarray([float(row["geos_member_ace"]) for row in group], dtype="float64")
        counts = np.asarray([float(row["geos_active_count"]) for row in group], dtype="float64")
        rows.append(
            {
                "method": method,
                "threshold_kt": threshold,
                "year": year,
                "basin_name": basin_name,
                "geos_mean_ace": float(np.nanmean(ace)),
                "geos_std_ace": float(np.nanstd(ace)) if ace.size > 1 else 0.0,
                "geos_mean_active_count": float(np.nanmean(counts)),
                "n_members": int(ace.size),
            }
        )
    return rows


def add_observed_ace(yearly_rows: list[dict[str, object]], args: argparse.Namespace, years: list[int]) -> None:
    months = parse_months(args.months)
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
    for row in yearly_rows:
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


def summarize_methods(yearly_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in yearly_rows:
        grouped[(str(row["method"]), float(row["threshold_kt"]), str(row["basin_name"]))].append(row)

    rows: list[dict[str, object]] = []
    for (method, threshold, basin_name), group in sorted(grouped.items()):
        geos = np.asarray([float(row["geos_mean_ace"]) for row in group], dtype="float64")
        obs = np.asarray([float(row.get("ibtracs_ace", np.nan)) for row in group], dtype="float64")
        diff = geos - obs
        mask = np.isfinite(geos) & np.isfinite(obs)
        geos_mean = float(np.nanmean(geos[mask])) if int(np.sum(mask)) else float("nan")
        obs_mean = float(np.nanmean(obs[mask])) if int(np.sum(mask)) else float("nan")
        ratio = geos_mean / obs_mean if obs_mean > 0.0 and np.isfinite(geos_mean) else float("nan")
        rows.append(
            {
                "method": method,
                "threshold_kt": threshold,
                "basin_name": basin_name,
                "n_years": int(np.sum(mask)),
                "geos_mean_ace": geos_mean,
                "ibtracs_mean_ace": obs_mean,
                "geos_to_ibtracs_ratio": ratio,
                "abs_log_ratio": abs(np.log(ratio)) if ratio > 0.0 and np.isfinite(ratio) else float("nan"),
                "raw_bias": float(np.nanmean(diff[mask])) if int(np.sum(mask)) else float("nan"),
                "rmse": float(np.sqrt(np.nanmean(diff[mask] ** 2))) if int(np.sum(mask)) else float("nan"),
                "corr": correlation(geos, obs),
            }
        )
    return rows


def rank_methods(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["method"]), float(row["threshold_kt"]))].append(row)

    rows: list[dict[str, object]] = []
    for (method, threshold), group in sorted(grouped.items()):
        basin_rows = [row for row in group if row["basin_name"] in BASIN_ORDER]
        weights = np.asarray([float(row["ibtracs_mean_ace"]) for row in basin_rows], dtype="float64")
        abs_logs = np.asarray([float(row["abs_log_ratio"]) for row in basin_rows], dtype="float64")
        ratios = np.asarray([float(row["geos_to_ibtracs_ratio"]) for row in basin_rows], dtype="float64")
        valid = np.isfinite(weights) & (weights > 0.0) & np.isfinite(abs_logs)
        if int(np.sum(valid)):
            weighted_abs_log = float(np.sum(weights[valid] * abs_logs[valid]) / np.sum(weights[valid]))
            weighted_ratio = float(np.sum(weights[valid] * ratios[valid]) / np.sum(weights[valid]))
        else:
            weighted_abs_log = float("nan")
            weighted_ratio = float("nan")
        all_basin = next((row for row in group if row["basin_name"] == "All Basins"), None)
        all_ratio = float(all_basin["geos_to_ibtracs_ratio"]) if all_basin is not None else float("nan")
        all_abs_log = abs(np.log(all_ratio)) if all_ratio > 0.0 and np.isfinite(all_ratio) else float("nan")
        rows.append(
            {
                "method": method,
                "threshold_kt": threshold,
                "weighted_abs_log_ratio": weighted_abs_log,
                "weighted_ratio": weighted_ratio,
                "weighted_required_multiplier": 1.0 / weighted_ratio
                if weighted_ratio > 0.0 and np.isfinite(weighted_ratio)
                else float("nan"),
                "all_basin_ratio": all_ratio,
                "all_basin_required_multiplier": 1.0 / all_ratio
                if all_ratio > 0.0 and np.isfinite(all_ratio)
                else float("nan"),
                "all_basin_abs_log_ratio": all_abs_log,
                "score": weighted_abs_log,
            }
        )

    rows.sort(key=lambda row: (float(row["score"]) if np.isfinite(float(row["score"])) else float("inf")))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def best_by_basin(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for basin_name in BASIN_ORDER + ["All Basins"]:
        basin_rows = [
            row
            for row in summary_rows
            if row["basin_name"] == basin_name and np.isfinite(float(row.get("abs_log_ratio", np.nan)))
        ]
        if not basin_rows:
            continue
        best = min(basin_rows, key=lambda row: float(row["abs_log_ratio"]))
        rows.append(dict(best))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"WARNING: no rows to write for {path}")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"  -> Wrote {path}")


def table_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "member": output_dir / f"{prefix}_member_values.csv",
        "yearly": output_dir / f"{prefix}_yearly.csv",
        "summary": output_dir / f"{prefix}_summary.csv",
        "ranking": output_dir / f"{prefix}_ranking.csv",
        "best": output_dir / f"{prefix}_best_by_basin.csv",
    }


def save_figure(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_all_basin_ratio(summary_rows: list[dict[str, object]], plot_dir: Path, prefix: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=dpi)
    methods = sorted({str(row["method"]) for row in summary_rows})
    for method in methods:
        rows = sorted(
            [row for row in summary_rows if row["method"] == method and row["basin_name"] == "All Basins"],
            key=lambda row: float(row["threshold_kt"]),
        )
        if not rows:
            continue
        thresholds = np.asarray([float(row["threshold_kt"]) for row in rows], dtype="float64")
        ratios = np.asarray([float(row["geos_to_ibtracs_ratio"]) for row in rows], dtype="float64")
        ax.plot(thresholds, ratios, marker="o", linewidth=1.7, markersize=3.4, label=method)
    ax.axhline(1.0, color="#22252b", linewidth=1.2, linestyle="--")
    ax.set_xlabel("GEOS candidate wind threshold (kt)")
    ax.set_ylabel("All-basin GEOS / IBTrACS ACE")
    ax.set_title("All-Basin ACE Ratio by Candidate Aggregation Method", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, plot_dir / f"{prefix}_all_basin_ratio.png", dpi)


def plot_score_heatmap(ranking_rows: list[dict[str, object]], plot_dir: Path, prefix: str, dpi: int) -> None:
    methods = sorted({str(row["method"]) for row in ranking_rows})
    thresholds = sorted({float(row["threshold_kt"]) for row in ranking_rows})
    values = np.full((len(methods), len(thresholds)), np.nan, dtype="float64")
    lookup = {(str(row["method"]), float(row["threshold_kt"])): float(row["weighted_abs_log_ratio"]) for row in ranking_rows}
    for i, method in enumerate(methods):
        for j, threshold in enumerate(thresholds):
            values[i, j] = lookup.get((method, threshold), np.nan)

    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=dpi)
    image = ax.imshow(values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{threshold:g}" for threshold in thresholds])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("GEOS candidate wind threshold (kt)")
    ax.set_title("Method Score: Lower Weighted |log(GEOS / IBTrACS ACE)| Is Better", fontsize=12, fontweight="bold")
    for i in range(len(methods)):
        for j in range(len(thresholds)):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if value > 0.55 else "#1e222a")
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label("weighted abs log ratio")
    save_figure(fig, plot_dir / f"{prefix}_method_score_heatmap.png", dpi)


def plot_best_basin_ratios(best_rows: list[dict[str, object]], plot_dir: Path, prefix: str, dpi: int) -> None:
    rows = [row for row in best_rows if row["basin_name"] in BASIN_ORDER]
    basins = [str(row["basin_name"]) for row in rows]
    ratios = np.asarray([float(row["geos_to_ibtracs_ratio"]) for row in rows], dtype="float64")
    colors = [BASINS[basin]["color"] for basin in basins]
    labels = [f"{row['method']}\nT={float(row['threshold_kt']):g}" for row in rows]

    fig, ax = plt.subplots(figsize=(12.0, 5.8), dpi=dpi)
    x = np.arange(len(rows))
    ax.axhline(1.0, color="#22252b", linewidth=1.1, linestyle="--")
    bars = ax.bar(x, ratios, color=colors, edgecolor="#ffffff", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(basins, rotation=25, ha="right")
    ax.set_ylabel("Best GEOS / IBTrACS ACE ratio")
    ax.set_title("Best Candidate Method by Basin", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=7)
    save_figure(fig, plot_dir / f"{prefix}_best_basin_ratios.png", dpi)


def print_top_methods(ranking_rows: list[dict[str, object]], limit: int = 12) -> None:
    print("\nTop candidate ACE aggregation methods")
    print(f"{'rank':>4}  {'method':16s} {'T':>6s} {'score':>9s} {'all_ratio':>9s} {'need_x':>8s} {'w_ratio':>9s}")
    for row in ranking_rows[:limit]:
        print(
            f"{int(row['rank']):4d}  "
            f"{str(row['method']):16s} "
            f"{float(row['threshold_kt']):6.1f} "
            f"{float(row['weighted_abs_log_ratio']):9.3f} "
            f"{float(row['all_basin_ratio']):9.3f} "
            f"{float(row['all_basin_required_multiplier']):8.2f} "
            f"{float(row['weighted_ratio']):9.3f}"
        )


def print_best_by_basin(best_rows: list[dict[str, object]]) -> None:
    print("\nBest method by basin")
    print(f"{'basin':20s} {'method':16s} {'T':>6s} {'ratio':>9s} {'need_x':>8s}")
    for row in best_rows:
        ratio = float(row["geos_to_ibtracs_ratio"])
        need_x = 1.0 / ratio if ratio > 0.0 and np.isfinite(ratio) else float("nan")
        print(
            f"{str(row['basin_name']):20s} "
            f"{str(row['method']):16s} "
            f"{float(row['threshold_kt']):6.1f} "
            f"{ratio:9.3f} "
            f"{need_x:8.2f}"
        )


def print_interpretation(ranking_rows: list[dict[str, object]]) -> None:
    if not ranking_rows:
        return
    best_weighted = ranking_rows[0]
    finite_all = [
        row
        for row in ranking_rows
        if np.isfinite(float(row.get("all_basin_abs_log_ratio", np.nan)))
    ]
    best_all = min(finite_all, key=lambda row: float(row["all_basin_abs_log_ratio"])) if finite_all else best_weighted
    all_ratio = float(best_all["all_basin_ratio"])
    need_x = float(best_all["all_basin_required_multiplier"])
    print("\nInterpretation")
    if np.isfinite(float(best_weighted["weighted_ratio"])):
        print(
            "Best basin-weighted method is "
            f"{best_weighted['method']} at T={float(best_weighted['threshold_kt']):.1f} kt: "
            f"weighted GEOS/IBTrACS ACE={float(best_weighted['weighted_ratio']):.3f}, "
            f"all-basin ratio={float(best_weighted['all_basin_ratio']):.3f}."
        )
    if np.isfinite(all_ratio):
        print(
            "Best all-basin amplitude method is "
            f"{best_all['method']} at T={float(best_all['threshold_kt']):.1f} kt: "
            f"GEOS/IBTrACS ACE={all_ratio:.3f}, requiring about {need_x:.2f}x more ACE to match."
        )
        if all_ratio < 0.75:
            print(
                "This is still far below 1.0, so candidate aggregation and wind threshold alone "
                "do not explain the ACE deficit in this cached accepted-candidate inventory."
            )
            print(
                "The next experiment should loosen or replace the structural detector, or generate "
                "a rejected-candidate inventory so SLP/warm-core/QV/vorticity gates can be tested."
            )
        elif 0.9 <= all_ratio <= 1.1:
            print(
                "All-basin ACE amplitude is close to IBTrACS for this short test. "
                "Check basin-by-basin stability with more years before promoting the method."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=[DEFAULT_CANDIDATES])
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS)
    parser.add_argument("--years", default="1991:1993")
    parser.add_argument("--init-dates", default="0824")
    parser.add_argument("--ens", default="all")
    parser.add_argument("--months", default="09:10")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--wind-var", default="usa_wind")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument("--nature-filter", default="TS")
    parser.add_argument("--all-natures", dest="nature_filter", action="store_const", const="")
    parser.add_argument("--basin-method", choices=("boxes", "ibtracs_code"), default="boxes")
    parser.add_argument("--all-hours", dest="synoptic_only", action="store_false")
    parser.add_argument("--min-lat", type=float, default=-25.0)
    parser.add_argument("--max-lat", type=float, default=50.0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", default="geos_ace_method_test")
    parser.add_argument("--dpi", type=int, default=300)
    parser.set_defaults(synoptic_only=True)
    add_ocean_only_args(parser)
    args = parser.parse_args(argv)

    setup_style()
    candidate_paths = expand_paths(args.candidates)
    if not candidate_paths:
        print(f"ERROR: no candidate CSV files matched: {' '.join(args.candidates)}", file=sys.stderr)
        return 1

    requested_years = parse_years(args.years)
    methods = parse_list(args.methods)
    thresholds = parse_float_list(args.thresholds)
    candidates = read_candidates(candidate_paths, args)
    years = print_candidate_coverage(candidates, requested_years)
    if not years:
        print("ERROR: no GEOS candidates remain after filters.", file=sys.stderr)
        return 1

    if set(years) != requested_years:
        print(f"Slicing IBTrACS comparison to GEOS-available years: {', '.join(str(year) for year in years)}")

    member_rows = evaluate_geos_methods(candidates, methods, thresholds)
    yearly_rows = aggregate_yearly(member_rows)
    add_observed_ace(yearly_rows, args, years)
    summary_rows = summarize_methods(yearly_rows)
    ranking_rows = rank_methods(summary_rows)
    best_rows = best_by_basin(summary_rows)

    paths = table_paths(Path(args.output_dir), args.prefix)
    write_csv(paths["member"], member_rows)
    write_csv(paths["yearly"], yearly_rows)
    write_csv(paths["summary"], summary_rows)
    write_csv(paths["ranking"], ranking_rows)
    write_csv(paths["best"], best_rows)

    plot_dir = Path(args.plot_dir)
    plot_all_basin_ratio(summary_rows, plot_dir, args.prefix, args.dpi)
    plot_score_heatmap(ranking_rows, plot_dir, args.prefix, args.dpi)
    plot_best_basin_ratios(best_rows, plot_dir, args.prefix, args.dpi)
    print_top_methods(ranking_rows)
    print_best_by_basin(best_rows)
    print_interpretation(ranking_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
