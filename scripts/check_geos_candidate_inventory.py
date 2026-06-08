#!/usr/bin/env python3
"""Check GEOS candidate CSV schema and accepted/rejected row counts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from candidate_inventory_defaults import CANONICAL_ALL_CENTERS_CANDIDATES


REQUIRED_ALL_CENTER_COLUMNS = (
    "passes_slp_anom",
    "passes_warm_core",
    "passes_qv",
    "passes_vorticity",
    "passes_structure",
    "passes_separation",
    "accepted_candidate",
    "rejection_reason",
)


def parse_year(value: str | None) -> int | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt).year
        except ValueError:
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=CANONICAL_ALL_CENTERS_CANDIDATES)
    parser.add_argument("--show-columns", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.candidates)
    if not path.is_file():
        print(f"ERROR: candidate CSV not found: {path}")
        return 1

    accepted = Counter()
    rejection = Counter()
    basin = Counter()
    years: set[int] = set()
    rows = 0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [column for column in REQUIRED_ALL_CENTER_COLUMNS if column not in columns]
        if args.show_columns:
            print("columns:")
            for column in columns:
                print(f"  {column}")

        for row in reader:
            rows += 1
            accepted[row.get("accepted_candidate", "MISSING")] += 1
            rejection[row.get("rejection_reason", "MISSING")] += 1
            basin[row.get("basin_name", "MISSING")] += 1
            year = parse_year(row.get("valid_time"))
            if year is not None:
                years.add(year)

    print(f"candidate_csv: {path}")
    print(f"rows: {rows:,}")
    if years:
        print(f"years: {min(years)}-{max(years)} ({len(years)} unique)")
    if missing:
        print("schema: old/accepted-only or incomplete")
        print("missing all-center columns: " + ", ".join(missing))
    else:
        print("schema: all-centers")
    print("accepted_candidate:")
    for key, count in sorted(accepted.items()):
        print(f"  {key}: {count:,}")
    print("rejection_reason:")
    for key, count in sorted(rejection.items()):
        print(f"  {key}: {count:,}")
    print("basins:")
    for key, count in sorted(basin.items()):
        print(f"  {key}: {count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

