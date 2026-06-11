#!/usr/bin/env python3
"""Compare ACE anomaly skill from two or more cached GEOS ACE methods."""

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

from ocean_mask_utils import add_ocean_only_args
from plot_ace_lead_anomaly_skill import (
    MEMBER_FIELDS,
    SKILL_FIELDS,
    YEARLY_FIELDS,
    add_anomalies,
    add_ibtracs_to_yearly_rows,
    aggregate_yearly_rows,
    final_member_rows,
    parse_lead_months,
    print_skill_summary,
    skill_rows,
)
from plot_ace_yearly_timeseries import BASIN_ORDER, DEFAULT_IBTRACS, parse_list, parse_years, setup_style


DEFAULT_METHODS = (
    "percentile=data/cache_ace_geos_pctl1991_2022_slp_qv_sep5_1991_2024",
    "constant25=data/cache_ace_constant25_slp_qv_sep5_1991_2024",
)
DEFAULT_TABLE_DIR = "data/analysis/ace_method_skill_comparison"
DEFAULT_PLOT_DIR = "plots/ace_method_skill_comparison"
METHOD_COLORS = ("#334f8d", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e")


def parse_method_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Method spec must be label=cache_dir: {value}")
        label, cache_dir = value.split("=", 1)
        label = label.strip()
        cache_dir = cache_dir.strip()
        if not label or not cache_dir:
            raise ValueError(f"Method spec must be label=cache_dir: {value}")
        if label in seen:
            raise ValueError(f"Duplicate method label: {label}")
        seen.add(label)
        specs.append((label, Path(cache_dir)))
    return specs


def table_paths(table_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    return (
        table_dir / f"{prefix}_member_values.csv",
        table_dir / f"{prefix}_yearly.csv",
        table_dir / f"{prefix}_skill.csv",
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"  -> Wrote {path}")


def method_years(yearly_rows: list[dict[str, object]], min_members: int) -> set[int]:
    years: set[int] = set()
    for row in yearly_rows:
        if str(row.get("basin_name")) != "All Basins":
            continue
        if min_members > 0 and int(float(row.get("n_members", 0))) < min_members:
            continue
        years.add(int(row["year"]))
    return years


def filter_yearly_rows(rows: list[dict[str, object]], years: set[int], min_members: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        if int(row["year"]) not in years:
            continue
        if min_members > 0 and int(float(row.get("n_members", 0))) < min_members:
            continue
        out.append(row)
    return out


def build_tables(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[str], list[str]]:
    years = parse_years(args.years)
    init_dates = parse_list(args.init_dates)
    lead_months = parse_lead_months(args.lead_months)
    method_specs = parse_method_specs(args.method)

    all_member_rows: list[dict[str, object]] = []
    yearly_by_method: dict[str, list[dict[str, object]]] = {}
    method_cache_dirs = {label: str(cache_dir) for label, cache_dir in method_specs}

    for label, cache_dir in method_specs:
        member_rows = final_member_rows(cache_dir, years, init_dates, lead_months)
        for row in member_rows:
            row["method"] = label
            row["cache_dir"] = str(cache_dir)
        all_member_rows.extend(member_rows)

        yearly_rows = aggregate_yearly_rows(member_rows)
        add_ibtracs_to_yearly_rows(yearly_rows, Path(args.ibtracs), lead_months, args)
        for row in yearly_rows:
            row["method"] = label
            row["cache_dir"] = str(cache_dir)
        yearly_by_method[label] = yearly_rows

    if args.common_years_only:
        common_years: set[int] | None = None
        for rows in yearly_by_method.values():
            years_for_method = method_years(rows, args.min_members_per_year)
            common_years = years_for_method if common_years is None else common_years.intersection(years_for_method)
        selected_years = set(sorted(common_years or set()))
    else:
        selected_years = years

    all_yearly_rows: list[dict[str, object]] = []
    all_skill_rows: list[dict[str, object]] = []
    for label, rows in yearly_by_method.items():
        filtered = filter_yearly_rows(rows, selected_years, args.min_members_per_year)
        if not filtered:
            print(f"WARNING: no yearly rows remain for method={label} after filters")
            continue
        add_anomalies(filtered)
        skill = skill_rows(filtered)
        for row in skill:
            row["method"] = label
            row["cache_dir"] = method_cache_dirs[label]
            ib_clim = float(row.get("ibtracs_clim_ace", np.nan))
            geos_clim = float(row.get("geos_clim_ace", np.nan))
            row["clim_ratio"] = geos_clim / ib_clim if np.isfinite(geos_clim) and np.isfinite(ib_clim) and ib_clim != 0.0 else float("nan")
        all_yearly_rows.extend(filtered)
        all_skill_rows.extend(skill)

    method_labels = [label for label, _ in method_specs]
    selected_year_list = [str(year) for year in sorted(selected_years)]
    print(f"Selected comparison years ({len(selected_year_list)}): {', '.join(selected_year_list)}")
    return all_member_rows, all_yearly_rows, all_skill_rows, method_labels, selected_year_list


def setup_axis(ax, ylabel: str) -> None:
    ax.grid(axis="y", color="#d1d5db", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel)


def skill_lookup(skill_rows: list[dict[str, object]], method: str, lead: str, basin: str) -> dict[str, object] | None:
    for row in skill_rows:
        if str(row.get("method")) == method and str(row.get("lead")) == lead and str(row.get("basin_name")) == basin:
            return row
    return None


def plot_metric_by_basin(
    skill_rows: list[dict[str, object]],
    methods: list[str],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    leads = sorted({str(row["lead"]) for row in skill_rows}, key=lambda text: int(text.replace("lead", "")) if text.replace("lead", "").isdigit() else text)
    basins = ["All Basins"] + BASIN_ORDER
    fig, axes = plt.subplots(len(leads), 1, figsize=(13.5, max(4.0, 3.7 * len(leads))), sharex=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    x = np.arange(len(basins))
    width = min(0.78 / max(len(methods), 1), 0.34)
    colors = {method: METHOD_COLORS[index % len(METHOD_COLORS)] for index, method in enumerate(methods)}

    for ax, lead in zip(axes_array, leads):
        for index, method in enumerate(methods):
            values = []
            for basin in basins:
                row = skill_lookup(skill_rows, method, lead, basin)
                values.append(float(row.get(metric, np.nan)) if row else np.nan)
            offset = (index - 0.5 * (len(methods) - 1)) * width
            ax.bar(x + offset, values, width=width, color=colors[method], label=method)
        if metric == "clim_ratio":
            ax.axhline(1.0, color="#1e222a", linestyle="--", linewidth=1.0)
        elif metric == "anom_corr":
            ax.axhline(0.0, color="#1e222a", linestyle="--", linewidth=1.0)
        setup_axis(ax, ylabel)
        ax.set_title(lead, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(basins, rotation=25, ha="right")
    axes_array[0].legend(frameon=False, ncol=min(len(methods), 4), fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def plot_all_basin_anomalies(
    yearly_rows: list[dict[str, object]],
    methods: list[str],
    path: Path,
    dpi: int,
) -> None:
    leads = sorted({str(row["lead"]) for row in yearly_rows}, key=lambda text: int(text.replace("lead", "")) if text.replace("lead", "").isdigit() else text)
    fig, axes = plt.subplots(len(leads), 1, figsize=(13.0, max(4.0, 3.7 * len(leads))), sharex=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    colors = {method: METHOD_COLORS[index % len(METHOD_COLORS)] for index, method in enumerate(methods)}

    for ax, lead in zip(axes_array, leads):
        lead_rows = [
            row for row in yearly_rows
            if str(row.get("lead")) == lead and str(row.get("basin_name")) == "All Basins"
        ]
        obs_by_year: dict[int, float] = {}
        for row in lead_rows:
            year = int(row["year"])
            obs_by_year[year] = float(row.get("ibtracs_anom_ace", np.nan))
        years = sorted(obs_by_year)
        if years:
            ax.plot(
                years,
                [obs_by_year[year] for year in years],
                color="#1e222a",
                linestyle="--",
                marker="s",
                markersize=3.0,
                linewidth=1.6,
                label="IBTrACS",
            )
        for method in methods:
            method_rows = sorted(
                [row for row in lead_rows if str(row.get("method")) == method],
                key=lambda row: int(row["year"]),
            )
            if not method_rows:
                continue
            ax.plot(
                [int(row["year"]) for row in method_rows],
                [float(row.get("geos_anom_ace", np.nan)) for row in method_rows],
                color=colors[method],
                marker="o",
                markersize=3.0,
                linewidth=1.6,
                label=method,
            )
        ax.axhline(0.0, color="#94a3b8", linewidth=0.9)
        setup_axis(ax, "ACE anomaly")
        ax.set_title(f"{lead} All Basins", fontsize=10.5, fontweight="bold")
    axes_array[0].legend(frameon=False, ncol=min(len(methods) + 1, 4), fontsize=9)
    axes_array[-1].set_xlabel("Year")
    fig.suptitle("All-Basin ACE Anomaly Comparison", fontsize=14, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path}")


def print_comparison_summary(skill_rows: list[dict[str, object]], methods: list[str]) -> None:
    print("")
    print("ACE method anomaly skill comparison")
    print(
        f"{'method':14s} {'lead':7s} {'mon':>3s} {'basin':20s} {'n':>3s} "
        f"{'r':>8s} {'rmse':>8s} {'mae':>8s} {'bias':>9s} {'GEOSclim':>9s} {'IBclim':>9s} {'ratio':>8s}"
    )
    basin_order = ["All Basins"] + BASIN_ORDER
    rows = sorted(
        skill_rows,
        key=lambda row: (
            methods.index(str(row.get("method"))) if str(row.get("method")) in methods else len(methods),
            str(row.get("lead")),
            basin_order.index(str(row.get("basin_name"))) if str(row.get("basin_name")) in basin_order else len(basin_order),
        ),
    )
    for row in rows:
        method = str(row.get("method", ""))
        basin = str(row.get("basin_name", ""))
        if basin not in basin_order:
            continue
        print(
            f"{method:14s} "
            f"{str(row.get('lead', '')):7s} "
            f"{str(row.get('month', '')):>3s} "
            f"{basin:20s} "
            f"{int(float(row.get('n_years', 0))):3d} "
            f"{float(row.get('anom_corr', np.nan)):8.3f} "
            f"{float(row.get('anom_rmse', np.nan)):8.3f} "
            f"{float(row.get('anom_mae', np.nan)):8.3f} "
            f"{float(row.get('raw_bias', np.nan)):9.3f} "
            f"{float(row.get('geos_clim_ace', np.nan)):9.3f} "
            f"{float(row.get('ibtracs_clim_ace', np.nan)):9.3f} "
            f"{float(row.get('clim_ratio', np.nan)):8.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method specification label=cache_dir. Repeat for multiple methods.",
    )
    parser.add_argument("--years", default="1991:2016")
    parser.add_argument("--init-dates", default="0824,0829")
    parser.add_argument("--lead-months", default="09,10")
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS)
    parser.add_argument("--wind-var", default="usa_wind")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument("--nature-filter", default="TS")
    parser.add_argument("--all-natures", dest="nature_filter", action="store_const", const="")
    parser.add_argument("--basin-method", choices=("boxes", "ibtracs_code"), default="boxes")
    parser.add_argument("--all-hours", dest="synoptic_only", action="store_false")
    parser.add_argument("--min-members-per-year", type=int, default=0)
    parser.add_argument("--all-years", dest="common_years_only", action="store_false", help="Keep each method's available years instead of intersecting common years.")
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--prefix", default="ace_method_skill_comparison")
    parser.add_argument("--dpi", type=int, default=300)
    parser.set_defaults(synoptic_only=True)
    parser.set_defaults(common_years_only=True)
    add_ocean_only_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.method:
        args.method = list(DEFAULT_METHODS)
    setup_style()
    table_dir = Path(args.table_dir)
    plot_dir = Path(args.plot_dir)

    member_rows, yearly_rows, skill_rows_out, methods, selected_years = build_tables(args)
    if not yearly_rows or not skill_rows_out:
        print("ERROR: no comparison rows were created", file=sys.stderr)
        return 1

    member_path, yearly_path, skill_path = table_paths(table_dir, args.prefix)
    member_fields = ["method", "cache_dir"] + MEMBER_FIELDS
    yearly_fields = ["method", "cache_dir"] + YEARLY_FIELDS
    skill_fields = ["method", "cache_dir"] + SKILL_FIELDS + ["clim_ratio"]
    write_csv(member_path, member_rows, member_fields)
    write_csv(yearly_path, yearly_rows, yearly_fields)
    write_csv(skill_path, skill_rows_out, skill_fields)
    write_csv(table_dir / f"{args.prefix}_years.csv", [{"year": year} for year in selected_years], ["year"])

    plot_all_basin_anomalies(yearly_rows, methods, plot_dir / f"{args.prefix}_all_basin_anomalies.png", args.dpi)
    plot_metric_by_basin(
        skill_rows_out,
        methods,
        "anom_corr",
        "Anomaly correlation",
        "ACE Anomaly Correlation by Method",
        plot_dir / f"{args.prefix}_anomaly_correlation.png",
        args.dpi,
    )
    plot_metric_by_basin(
        skill_rows_out,
        methods,
        "anom_rmse",
        "Anomaly RMSE",
        "ACE Anomaly RMSE by Method",
        plot_dir / f"{args.prefix}_anomaly_rmse.png",
        args.dpi,
    )
    plot_metric_by_basin(
        skill_rows_out,
        methods,
        "raw_bias",
        "Raw bias",
        "ACE Raw Bias by Method",
        plot_dir / f"{args.prefix}_raw_bias.png",
        args.dpi,
    )
    plot_metric_by_basin(
        skill_rows_out,
        methods,
        "clim_ratio",
        "GEOS / IBTrACS climatology",
        "ACE Climatology Ratio by Method",
        plot_dir / f"{args.prefix}_climatology_ratio.png",
        args.dpi,
    )
    print_comparison_summary(skill_rows_out, methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
