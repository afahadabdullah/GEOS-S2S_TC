#!/usr/bin/env python3
"""Evaluate GEOS ACE anomaly skill against IBTrACS by lead month and basin."""

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
from ocean_mask_utils import add_ocean_only_args
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
DEFAULT_TABLE_DIR = "data/analysis/ace_lead_skill"
DEFAULT_PLOT_DIR = "plots/ace_lead_skill"


def parse_lead_months(value: str) -> list[tuple[str, str]]:
    months = sorted(parse_months(value))
    return [(f"lead{index + 1}", month) for index, month in enumerate(months)]


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
    out: list[Path] = []
    for path in paths:
        if cache_member_info(path) is not None:
            out.append(path)
    return sorted(out)


def final_member_rows(
    cache_dir: Path,
    years: set[int],
    init_dates: list[str],
    lead_months: list[tuple[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    member_files = discover_member_files(cache_dir, years, init_dates)
    if not member_files:
        raise RuntimeError(f"No member ACE caches found in {cache_dir}")

    print(f"Reading {len(member_files)} GEOS member ACE cache(s) for lead skill.")
    for path in member_files:
        info = cache_member_info(path)
        if info is None:
            continue
        init_date, ens = info
        year = int(init_date[:4])
        _, _, _, _, times, _, diagnostics, _, _ = read_cache(path)
        time_months = np.asarray([time_value.strftime("%m") for time_value in times])

        for lead_name, month in lead_months:
            all_total = 0.0
            any_finite = False
            for basin_name in BASIN_ORDER:
                steps = np.asarray(diagnostics[basin_name]["step_ace"], dtype="float64")
                month_mask = time_months == month
                total_ace = float(np.nansum(steps[month_mask])) if steps.size and np.any(month_mask) else float("nan")
                if np.isfinite(total_ace):
                    all_total += total_ace
                    any_finite = True
                rows.append(
                    {
                        "year": year,
                        "init_date": init_date,
                        "ens": ens,
                        "lead": lead_name,
                        "month": month,
                        "basin_name": basin_name,
                        "geos_member_ace": total_ace,
                        "cache_file": path.name,
                    }
                )
            rows.append(
                {
                    "year": year,
                    "init_date": init_date,
                    "ens": ens,
                    "lead": lead_name,
                    "month": month,
                    "basin_name": "All Basins",
                    "geos_member_ace": all_total if any_finite else float("nan"),
                    "cache_file": path.name,
                }
            )
    return rows


def aggregate_yearly_rows(member_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    init_members: dict[tuple[int, str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in member_rows:
        key = (
            int(row["year"]),
            str(row["lead"]),
            str(row["month"]),
            str(row["basin_name"]),
        )
        value = float(row["geos_member_ace"])
        if np.isfinite(value):
            grouped[key].append(value)
            init_members[key].add((str(row["init_date"]), str(row["ens"])))

    rows: list[dict[str, object]] = []
    for (year, lead, month, basin_name), values in sorted(grouped.items()):
        array = np.asarray(values, dtype="float64")
        rows.append(
            {
                "year": year,
                "lead": lead,
                "month": month,
                "basin_name": basin_name,
                "geos_mean_ace": float(np.nanmean(array)),
                "geos_std_ace": float(np.nanstd(array)) if array.size > 1 else 0.0,
                "n_members": int(array.size),
                "member_ids": ";".join(f"{init}:{ens}" for init, ens in sorted(init_members[(year, lead, month, basin_name)])),
            }
        )
    return rows


def add_ibtracs_to_yearly_rows(
    rows: list[dict[str, object]],
    ibtracs_path: Path,
    lead_months: list[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    years = sorted({int(row["year"]) for row in rows})
    observed_by_lead: dict[tuple[str, int, str], float] = {}
    counts_by_lead: dict[tuple[str, int, str], int] = {}

    for lead_name, month in lead_months:
        obs_ace, obs_counts = read_ibtracs_observed_ace(
            ibtracs_path,
            years=set(years),
            months={month},
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
        for year in years:
            for basin_name in BASIN_ORDER:
                observed_by_lead[(lead_name, year, basin_name)] = float(obs_ace.get((year, basin_name), 0.0))
                counts_by_lead[(lead_name, year, basin_name)] = int(obs_counts.get((year, basin_name), 0))
            observed_by_lead[(lead_name, year, "All Basins")] = sum(
                float(obs_ace.get((year, basin), 0.0)) for basin in BASIN_ORDER
            )
            counts_by_lead[(lead_name, year, "All Basins")] = sum(
                int(obs_counts.get((year, basin), 0)) for basin in BASIN_ORDER
            )

    for row in rows:
        key = (str(row["lead"]), int(row["year"]), str(row["basin_name"]))
        row["ibtracs_ace"] = observed_by_lead.get(key, float("nan"))
        row["ibtracs_fix_count"] = counts_by_lead.get(key, 0)


def add_anomalies(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["lead"]), str(row["basin_name"]))].append(row)

    for group_rows in grouped.values():
        geos_values = np.asarray([float(row["geos_mean_ace"]) for row in group_rows], dtype="float64")
        obs_values = np.asarray([float(row.get("ibtracs_ace", np.nan)) for row in group_rows], dtype="float64")
        geos_clim = float(np.nanmean(geos_values)) if np.isfinite(geos_values).any() else float("nan")
        obs_clim = float(np.nanmean(obs_values)) if np.isfinite(obs_values).any() else float("nan")
        for row in group_rows:
            row["geos_clim_ace"] = geos_clim
            row["ibtracs_clim_ace"] = obs_clim
            row["geos_anom_ace"] = float(row["geos_mean_ace"]) - geos_clim if np.isfinite(geos_clim) else float("nan")
            obs = float(row.get("ibtracs_ace", np.nan))
            row["ibtracs_anom_ace"] = obs - obs_clim if np.isfinite(obs) and np.isfinite(obs_clim) else float("nan")
            row["raw_bias_ace"] = float(row["geos_mean_ace"]) - obs if np.isfinite(obs) else float("nan")


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if float(np.nanstd(x_valid)) == 0.0 or float(np.nanstd(y_valid)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def skill_rows(yearly_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in yearly_rows:
        grouped[(str(row["lead"]), str(row["month"]), str(row["basin_name"]))].append(row)

    rows: list[dict[str, object]] = []
    for (lead, month, basin_name), group_rows in sorted(grouped.items()):
        geos_anom = np.asarray([float(row.get("geos_anom_ace", np.nan)) for row in group_rows], dtype="float64")
        obs_anom = np.asarray([float(row.get("ibtracs_anom_ace", np.nan)) for row in group_rows], dtype="float64")
        geos_raw = np.asarray([float(row.get("geos_mean_ace", np.nan)) for row in group_rows], dtype="float64")
        obs_raw = np.asarray([float(row.get("ibtracs_ace", np.nan)) for row in group_rows], dtype="float64")
        mask = np.isfinite(geos_anom) & np.isfinite(obs_anom)
        diff = geos_anom[mask] - obs_anom[mask]
        raw_diff = geos_raw[mask] - obs_raw[mask]
        rows.append(
            {
                "lead": lead,
                "month": month,
                "basin_name": basin_name,
                "n_years": int(np.sum(mask)),
                "anom_corr": correlation(geos_anom, obs_anom),
                "anom_rmse": float(np.sqrt(np.nanmean(diff**2))) if diff.size else float("nan"),
                "anom_mae": float(np.nanmean(np.abs(diff))) if diff.size else float("nan"),
                "raw_bias": float(np.nanmean(raw_diff)) if raw_diff.size else float("nan"),
                "geos_clim_ace": float(np.nanmean(geos_raw[mask])) if int(np.sum(mask)) else float("nan"),
                "ibtracs_clim_ace": float(np.nanmean(obs_raw[mask])) if int(np.sum(mask)) else float("nan"),
            }
        )
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


MEMBER_FIELDS = ["year", "init_date", "ens", "lead", "month", "basin_name", "geos_member_ace", "cache_file"]
YEARLY_FIELDS = [
    "year",
    "lead",
    "month",
    "basin_name",
    "geos_mean_ace",
    "geos_std_ace",
    "n_members",
    "ibtracs_ace",
    "ibtracs_fix_count",
    "geos_clim_ace",
    "ibtracs_clim_ace",
    "geos_anom_ace",
    "ibtracs_anom_ace",
    "raw_bias_ace",
    "member_ids",
]
SKILL_FIELDS = [
    "lead",
    "month",
    "basin_name",
    "n_years",
    "anom_corr",
    "anom_rmse",
    "anom_mae",
    "raw_bias",
    "geos_clim_ace",
    "ibtracs_clim_ace",
]


def table_paths(table_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    return (
        table_dir / f"{prefix}_member_values.csv",
        table_dir / f"{prefix}_yearly.csv",
        table_dir / f"{prefix}_skill.csv",
    )


def plot_anomaly_panels(rows: list[dict[str, object]], lead: str, path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=dpi, sharex=True)
    for ax, basin_name in zip(axes.ravel(), BASIN_ORDER):
        basin_rows = sorted(
            [row for row in rows if row["basin_name"] == basin_name and row["lead"] == lead],
            key=lambda row: int(row["year"]),
        )
        years = np.asarray([int(row["year"]) for row in basin_rows], dtype="int32")
        geos_anom = np.asarray([float(row["geos_anom_ace"]) for row in basin_rows], dtype="float64")
        obs_anom = np.asarray([float(row["ibtracs_anom_ace"]) for row in basin_rows], dtype="float64")
        geos_std = np.asarray([float(row["geos_std_ace"]) for row in basin_rows], dtype="float64")
        color = BASINS[basin_name]["color"]
        ax.axhline(0.0, color="#999999", linewidth=0.9)
        ax.plot(years, geos_anom, color=color, linewidth=1.7, marker="o", markersize=3.2, label="GEOS anomaly")
        if np.any(geos_std > 0.0):
            ax.fill_between(years, geos_anom - geos_std, geos_anom + geos_std, color=color, alpha=0.16, linewidth=0)
        ax.plot(years, obs_anom, color="#1e222a", linewidth=1.5, marker="s", markersize=3.0, linestyle="--", label="IBTrACS anomaly")
        ax.set_title(basin_name, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"GEOS vs IBTrACS ACE Anomalies ({lead})", fontsize=13, fontweight="bold", y=0.99)
    fig.supxlabel("Year")
    fig.supylabel("ACE anomaly (10^4 kt^2)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_skill(skill: list[dict[str, object]], path: Path, dpi: int) -> None:
    leads = sorted({str(row["lead"]) for row in skill}, key=lambda value: int(value.replace("lead", "")) if value.replace("lead", "").isdigit() else value)
    basins = BASIN_ORDER
    x = np.arange(len(basins))
    width = 0.36 if len(leads) <= 2 else 0.8 / max(len(leads), 1)
    offsets = np.linspace(-width * (len(leads) - 1) / 2, width * (len(leads) - 1) / 2, len(leads)) if leads else []

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=dpi, sharex=True)
    for offset, lead in zip(offsets, leads):
        corr_values = []
        rmse_values = []
        for basin in basins:
            row = next((item for item in skill if item["lead"] == lead and item["basin_name"] == basin), None)
            corr_values.append(float(row["anom_corr"]) if row is not None else float("nan"))
            rmse_values.append(float(row["anom_rmse"]) if row is not None else float("nan"))
        axes[0].bar(x + offset, corr_values, width, label=lead)
        axes[1].bar(x + offset, rmse_values, width, label=lead)

    axes[0].axhline(0.0, color="#777777", linewidth=0.9)
    axes[0].set_ylabel("Anomaly correlation")
    axes[0].set_title("Lead-Month ACE Anomaly Skill by Basin", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_ylabel("Anomaly RMSE (10^4 kt^2)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(basins, rotation=28, ha="right")
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> Saved {path}")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_float(value: object, width: int = 8, precision: int = 3) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return f"{'nan':>{width}s}"
    return f"{number:{width}.{precision}f}"


def print_skill_summary(skill: list[dict[str, object]]) -> None:
    if not skill:
        print("\nNo lead skill metrics available.")
        return

    lead_sort = lambda value: int(value.replace("lead", "")) if value.replace("lead", "").isdigit() else 99
    basin_order = ["All Basins"] + BASIN_ORDER
    rows = sorted(
        skill,
        key=lambda row: (
            lead_sort(str(row.get("lead", ""))),
            basin_order.index(str(row.get("basin_name", "")))
            if str(row.get("basin_name", "")) in basin_order
            else len(basin_order),
        ),
    )

    print("\nACE lead anomaly skill metrics")
    print(
        f"{'lead':7s} {'mon':>3s} {'basin':20s} {'n':>3s} "
        f"{'r':>8s} {'rmse':>8s} {'mae':>8s} {'bias':>9s} "
        f"{'GEOSclim':>9s} {'IBclim':>9s} {'ratio':>8s}"
    )
    for row in rows:
        basin_name = str(row.get("basin_name", ""))
        if basin_name not in basin_order:
            continue
        geos_clim = safe_float(row.get("geos_clim_ace"))
        ibtracs_clim = safe_float(row.get("ibtracs_clim_ace"))
        ratio = geos_clim / ibtracs_clim if ibtracs_clim > 0.0 and np.isfinite(geos_clim) else float("nan")
        n_years = safe_float(row.get("n_years"))
        print(
            f"{str(row.get('lead', '')):7s} "
            f"{str(row.get('month', '')):>3s} "
            f"{basin_name:20s} "
            f"{int(n_years) if np.isfinite(n_years) else 0:3d} "
            f"{format_float(row.get('anom_corr'))} "
            f"{format_float(row.get('anom_rmse'))} "
            f"{format_float(row.get('anom_mae'))} "
            f"{format_float(row.get('raw_bias'), width=9)} "
            f"{format_float(geos_clim, width=9)} "
            f"{format_float(ibtracs_clim, width=9)} "
            f"{format_float(ratio)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", default="geos_lagged_vs_ibtracs_ace_lead_skill")
    parser.add_argument("--years", default="1991:2024")
    parser.add_argument("--init-dates", default="0824,0829", help="Month/day init dates to pool, e.g. 0824,0829.")
    parser.add_argument("--lead-months", default="09,10", help="Lead months to evaluate. Default maps 09->lead1, 10->lead2.")
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS)
    parser.add_argument("--wind-var", default="usa_wind")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument("--nature-filter", default="TS")
    parser.add_argument("--all-natures", dest="nature_filter", action="store_const", const="")
    parser.add_argument("--basin-method", choices=("boxes", "ibtracs_code"), default="boxes")
    parser.add_argument("--all-hours", dest="synoptic_only", action="store_false")
    parser.add_argument("--use-cached-tables", action="store_true", help="Read cached CSV tables and redraw plots only.")
    parser.add_argument(
        "--no-print-skill-summary",
        dest="print_skill_summary",
        action="store_false",
        help="Do not print the compact skill metric table after plotting.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.set_defaults(synoptic_only=True)
    parser.set_defaults(print_skill_summary=True)
    add_ocean_only_args(parser)
    args = parser.parse_args(argv)

    setup_style()
    table_dir = Path(args.table_dir)
    plot_dir = Path(args.plot_dir)
    member_path, yearly_path, skill_path = table_paths(table_dir, args.prefix)

    if args.use_cached_tables:
        if not yearly_path.exists() or not skill_path.exists():
            print(f"ERROR: cached lead tables are missing under {table_dir}", file=sys.stderr)
            return 1
        yearly = read_csv(yearly_path)
        skill = read_csv(skill_path)
    else:
        years = parse_years(args.years)
        init_dates = parse_list(args.init_dates)
        lead_months = parse_lead_months(args.lead_months)
        member_rows = final_member_rows(Path(args.cache_dir), years, init_dates, lead_months)
        yearly = aggregate_yearly_rows(member_rows)
        add_ibtracs_to_yearly_rows(yearly, Path(args.ibtracs), lead_months, args)
        add_anomalies(yearly)
        skill = skill_rows(yearly)
        write_csv(member_path, member_rows, MEMBER_FIELDS)
        write_csv(yearly_path, yearly, YEARLY_FIELDS)
        write_csv(skill_path, skill, SKILL_FIELDS)

    leads = sorted({str(row["lead"]) for row in yearly}, key=lambda value: int(value.replace("lead", "")) if value.replace("lead", "").isdigit() else value)
    for lead in leads:
        plot_anomaly_panels(yearly, lead, plot_dir / f"{args.prefix}_{lead}_anomalies_by_basin.png", args.dpi)
    plot_skill(skill, plot_dir / f"{args.prefix}_skill_by_basin.png", args.dpi)
    if args.print_skill_summary:
        print_skill_summary(skill)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
