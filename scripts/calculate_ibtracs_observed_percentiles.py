#!/usr/bin/env python3
"""Calculate observed IBTrACS basin percentiles for TC threshold calibration.

This script estimates where the observed tropical-storm threshold, normally
34 kt, sits in the observed 6-hourly intensity distribution for each basin:

    p_b = fraction(Vobs_b <= 34 kt)

That percentile can then be mapped onto the GEOS candidate wind distribution:

    T_geos_b = percentile(Vgeos_b, 100 * p_b)

By default, samples are assigned to the same latitude/longitude basin boxes used
by the TC-conditioned ACE script, so the observed percentiles line up with the
model regions where thresholds will be applied. Samples are also filtered to
IBTrACS ``NATURE=TS`` by default, which keeps tropical cyclone fixes and removes
disturbance, subtropical, extratropical, mixed, and unreported-nature fixes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ocean_mask_utils import add_ocean_only_args, build_ocean_checker

try:
    import netCDF4
except ImportError:
    print("ERROR: netCDF4 package is required. Load the earth environment or install netCDF4.", file=sys.stderr)
    sys.exit(2)


DEFAULT_IBTRACS = "data/obs/ibtracs/IBTrACS.since1980.v04r01.nc"
DEFAULT_OUTPUT_CSV = "data/obs/ibtracs/ibtracs_observed_percentiles.csv"

BASINS = {
    "North Atlantic": {
        "codes": ("NA",),
        "lat_range": (0.0, 45.0),
        "lon_range": (-100.0, -10.0),
    },
    "Northeast Pacific": {
        "codes": ("EP", "CP"),
        "lat_range": (0.0, 40.0),
        "lon_range": (-180.0, -100.0),
    },
    "Northwest Pacific": {
        "codes": ("WP",),
        "lat_range": (0.0, 45.0),
        "lon_range": (100.0, 180.0),
    },
    "North Indian": {
        "codes": ("NI",),
        "lat_range": (0.0, 40.0),
        "lon_range": (40.0, 100.0),
    },
    "South Indian": {
        "codes": ("SI",),
        "lat_range": (-40.0, 0.0),
        "lon_range": (20.0, 135.0),
    },
    "South Pacific": {
        "codes": ("SP",),
        "lat_range": (-40.0, 0.0),
        "lon_ranges": [(135.0, 180.0), (-180.0, -120.0)],
    },
}


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def parse_int_set(value: str) -> set[int]:
    return {int(item) for item in parse_list(value)}


def decode_chars(values) -> str:
    array = np.ma.filled(np.asarray(values), b" ")
    chars: list[str] = []
    for item in array.reshape(-1):
        if isinstance(item, bytes):
            chars.append(item.decode("utf-8", errors="ignore"))
        elif isinstance(item, str):
            chars.append(item)
        else:
            value = int(item)
            if value > 0:
                chars.append(chr(value))
    return "".join(chars).strip()


def read_storm_time_text(variable, storm_index: int, time_index: int) -> str:
    if getattr(variable, "ndim", 0) <= 2:
        return decode_chars(variable[storm_index, time_index])
    return decode_chars(variable[storm_index, time_index, :])


def normalize_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def basin_from_boxes(lat: float, lon: float) -> str | None:
    lon = normalize_lon(lon)
    for basin_name, basin_def in BASINS.items():
        lat_min, lat_max = basin_def["lat_range"]
        if not (lat_min <= lat <= lat_max):
            continue

        if "lon_range" in basin_def:
            lon_min, lon_max = basin_def["lon_range"]
            if lon_min <= lon <= lon_max:
                return basin_name
        else:
            for lon_min, lon_max in basin_def["lon_ranges"]:
                if lon_min <= lon <= lon_max:
                    return basin_name
    return None


def basin_from_code(code: str) -> str | None:
    code = code.strip().upper()
    for basin_name, basin_def in BASINS.items():
        if code in basin_def["codes"]:
            return basin_name
    return None


def finite_position_value(value) -> float | None:
    if np.ma.is_masked(value):
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def finite_wind_value(value) -> float | None:
    out = finite_position_value(value)
    if out is None or out < 0.0:
        return None
    return out


def date_is_selected(date_value, start_year: int, end_year: int, months: set[int], synoptic_only: bool) -> bool:
    year = int(getattr(date_value, "year", 0))
    month = int(getattr(date_value, "month", 0))
    hour = int(getattr(date_value, "hour", 0))

    if year < start_year or year > end_year:
        return False
    if month not in months:
        return False
    if synoptic_only and hour not in (0, 6, 12, 18):
        return False
    return True


def nature_is_selected(nature_value: str, allowed_natures: set[str]) -> bool:
    return not allowed_natures or nature_value.strip().upper() in allowed_natures


def summarize(values: list[float], threshold_kt: float) -> dict[str, float | int]:
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        return {
            "n_samples": 0,
            "count_le_threshold": 0,
            "p_obs_threshold": np.nan,
            "percentile_obs_threshold": np.nan,
            "min_wind": np.nan,
            "p10_wind": np.nan,
            "median_wind": np.nan,
            "p90_wind": np.nan,
            "max_wind": np.nan,
        }

    count_le = int(np.sum(array <= threshold_kt))
    p_obs = count_le / float(array.size)
    return {
        "n_samples": int(array.size),
        "count_le_threshold": count_le,
        "p_obs_threshold": p_obs,
        "percentile_obs_threshold": 100.0 * p_obs,
        "min_wind": float(np.nanmin(array)),
        "p10_wind": float(np.nanpercentile(array, 10)),
        "median_wind": float(np.nanpercentile(array, 50)),
        "p90_wind": float(np.nanpercentile(array, 90)),
        "max_wind": float(np.nanmax(array)),
    }


def calculate_percentiles(args: argparse.Namespace) -> list[dict[str, object]]:
    months = parse_int_set(args.months)
    wind_vars = parse_list(args.wind_vars)
    allowed_natures = {item.upper() for item in parse_list(args.nature_filter)}
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    skipped_no_basin = 0
    skipped_nature = 0
    skipped_no_time = 0
    skipped_land = 0

    with netCDF4.Dataset(args.ibtracs, "r") as ds:
        missing_vars = [name for name in wind_vars if name not in ds.variables]
        if missing_vars:
            raise ValueError(f"Missing requested wind variable(s): {', '.join(missing_vars)}")

        required_vars = ["time", "lat", "lon", "basin"]
        if allowed_natures:
            required_vars.append("nature")
        missing_required = [name for name in required_vars if name not in ds.variables]
        if missing_required:
            raise ValueError(f"Missing required IBTrACS variable(s): {', '.join(missing_required)}")

        time_var = ds.variables["time"]
        time_values = time_var[:]
        time_mask = np.ma.getmaskarray(time_values)
        dates = netCDF4.num2date(
            time_values,
            time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
        )

        lat_values = ds.variables["lat"][:]
        lon_values = ds.variables["lon"][:]
        basin_values = ds.variables["basin"]
        nature_values = ds.variables["nature"] if allowed_natures else None
        wind_values = {name: ds.variables[name][:] for name in wind_vars}
        dist2land_values = ds.variables["dist2land"][:] if args.ocean_only and "dist2land" in ds.variables else None
        ocean_checker = None
        if args.ocean_only and dist2land_values is None:
            ocean_checker, ocean_warning = build_ocean_checker(
                args.ocean_mask_source,
                mask_file=args.ocean_mask_file,
                threshold=args.ocean_threshold,
                require_mask=True,
            )
            print(f"Ocean-only IBTrACS percentile filter enabled: source={ocean_checker.source}")
            if ocean_warning:
                print(f"WARNING: IBTrACS ocean mask fallback: {ocean_warning}")
        elif args.ocean_only:
            print("Ocean-only IBTrACS percentile filter enabled: source=dist2land")

        nstorm, ntime = time_values.shape
        for storm_index in range(nstorm):
            for time_index in range(ntime):
                if time_mask[storm_index, time_index]:
                    skipped_no_time += 1
                    continue

                date_value = dates[storm_index, time_index]
                if not date_is_selected(date_value, args.start_year, args.end_year, months, args.synoptic_only):
                    continue

                if nature_values is not None:
                    nature = read_storm_time_text(nature_values, storm_index, time_index)
                    if not nature_is_selected(nature, allowed_natures):
                        skipped_nature += 1
                        continue

                lat = finite_position_value(lat_values[storm_index, time_index])
                lon = finite_position_value(lon_values[storm_index, time_index])
                if lat is None or lon is None:
                    skipped_no_basin += 1
                    continue
                if args.ocean_only:
                    if dist2land_values is not None:
                        dist2land = finite_position_value(dist2land_values[storm_index, time_index])
                        is_ocean = dist2land is not None and dist2land > 0.0
                    else:
                        is_ocean = ocean_checker.is_ocean(lat, lon)
                    if not is_ocean:
                        skipped_land += 1
                        continue

                if args.basin_method == "boxes":
                    basin_name = basin_from_boxes(lat, lon)
                else:
                    basin_code = decode_chars(basin_values[storm_index, time_index, :])
                    basin_name = basin_from_code(basin_code)

                if basin_name is None:
                    skipped_no_basin += 1
                    continue

                for wind_var, wind_array in wind_values.items():
                    wind = finite_wind_value(wind_array[storm_index, time_index])
                    if wind is not None:
                        samples[(basin_name, wind_var)].append(wind)

    rows: list[dict[str, object]] = []
    for basin_name, basin_def in BASINS.items():
        for wind_var in wind_vars:
            stats = summarize(samples[(basin_name, wind_var)], args.threshold_kt)
            rows.append(
                {
                    "basin_name": basin_name,
                    "ibtracs_codes": ",".join(basin_def["codes"]),
                    "basin_method": args.basin_method,
                    "wind_var": wind_var,
                    "threshold_kt": args.threshold_kt,
                    "start_year": args.start_year,
                    "end_year": args.end_year,
                    "months": ",".join(str(month) for month in sorted(months)),
                    "synoptic_only": int(args.synoptic_only),
                    "nature_filter": ",".join(sorted(allowed_natures)) if allowed_natures else "ALL",
                    "ocean_only": int(args.ocean_only),
                    "ocean_mask_source": args.ocean_mask_source,
                    **stats,
                }
            )

    if args.verbose:
        print(f"Skipped missing time samples: {skipped_no_time}")
        print(f"Skipped samples outside nature filter: {skipped_nature}")
        print(f"Skipped land samples: {skipped_land}")
        print(f"Skipped samples outside supported basins/missing position: {skipped_no_basin}")

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "basin_name",
        "ibtracs_codes",
        "basin_method",
        "wind_var",
        "threshold_kt",
        "start_year",
        "end_year",
        "months",
        "synoptic_only",
        "nature_filter",
        "ocean_only",
        "ocean_mask_source",
        "n_samples",
        "count_le_threshold",
        "p_obs_threshold",
        "percentile_obs_threshold",
        "min_wind",
        "p10_wind",
        "median_wind",
        "p90_wind",
        "max_wind",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows: list[dict[str, object]]) -> None:
    print("Observed IBTrACS percentile of threshold wind")
    print("")
    print(
        f"{'basin':20s} {'wind':10s} {'n':>8s} {'p_obs':>8s} "
        f"{'pct':>8s} {'median':>8s} {'p90':>8s}"
    )
    for row in rows:
        n_samples = int(row["n_samples"])
        p_obs = float(row["p_obs_threshold"]) if n_samples else np.nan
        pct = float(row["percentile_obs_threshold"]) if n_samples else np.nan
        median = float(row["median_wind"]) if n_samples else np.nan
        p90 = float(row["p90_wind"]) if n_samples else np.nan
        print(
            f"{str(row['basin_name']):20s} {str(row['wind_var']):10s} {n_samples:8d} "
            f"{p_obs:8.4f} {pct:8.2f} {median:8.2f} {p90:8.2f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibtracs", default=DEFAULT_IBTRACS, help="Path to IBTrACS v04r01 NetCDF file.")
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--months", default="9,10", help="Months to include, separated by comma/colon/space.")
    parser.add_argument("--threshold-kt", type=float, default=34.0)
    parser.add_argument(
        "--wind-vars",
        default="wmo_wind,usa_wind",
        help="IBTrACS wind variables to evaluate, separated by comma/colon/space.",
    )
    parser.add_argument(
        "--basin-method",
        choices=("boxes", "ibtracs_code"),
        default="boxes",
        help="Assign samples using model basin boxes or the IBTrACS basin code variable.",
    )
    parser.add_argument(
        "--nature-filter",
        default="TS",
        help="IBTrACS NATURE codes to keep, separated by comma/colon/space. Default TS keeps tropical fixes only.",
    )
    parser.add_argument(
        "--all-natures",
        action="store_const",
        const="",
        dest="nature_filter",
        help="Disable IBTrACS NATURE filtering.",
    )
    parser.add_argument(
        "--all-hours",
        action="store_false",
        dest="synoptic_only",
        help="Include all IBTrACS times instead of only 00/06/12/18 UTC samples.",
    )
    add_ocean_only_args(parser)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--no-output-csv", action="store_true", help="Print only; do not write a CSV file.")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(synoptic_only=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = calculate_percentiles(args)
    print_table(rows)

    if not args.no_output_csv:
        output_path = Path(args.output_csv)
        write_csv(output_path, rows)
        print("")
        print(f"Wrote CSV: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
