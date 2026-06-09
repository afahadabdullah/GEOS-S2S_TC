#!/usr/bin/env python3
"""Summarize TC-conditioned ACE cache progress across init dates."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SFC_ROOT = "/nobackupp27/afahad/project/GEOS-S2S_TC/data"
DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_INIT_DATES_FILE = "config/init_dates_late_aug_1991_2024.txt"


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def parse_years(value: str | None) -> set[int]:
    years: set[int] = set()
    if not value:
        return years
    # Keep ':' and '-' intact here so ranges like 1991:2024 are expanded
    # instead of being interpreted as the two isolated years 1991 and 2024.
    for item in [part for part in re.split(r"[\s,]+", value.strip()) if part]:
        if ":" in item:
            start, end = item.split(":", 1)
            years.update(range(int(start), int(end) + 1))
        elif "-" in item:
            start, end = item.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(item))
    return years


def read_init_dates(path: Path, years: set[int]) -> list[str]:
    init_dates: list[str] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        value = text.split()[0]
        if not re.fullmatch(r"\d{8}", value):
            continue
        if years and int(value[:4]) not in years:
            continue
        init_dates.append(value)
    return init_dates


def progress_bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[" + "-" * width + "] unknown"
    done = max(0, min(done, total))
    filled = int(round(width * done / total))
    pct = 100.0 * done / total
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({pct:5.1f}%)"


def discover_members(sfc_root: Path, init_date: str, ens: str) -> list[str]:
    if ens.lower() != "all":
        return parse_list(ens)
    init_dir = sfc_root / "GEOS_fcst" / init_date
    if not init_dir.is_dir():
        return []
    return sorted(path.name for path in init_dir.iterdir() if path.is_dir() and path.name.startswith("ens"))


def cached_members(cache_dir: Path, init_date: str) -> set[str]:
    out: set[str] = set()
    pattern = re.compile(rf"^tc_conditioned_ace_{re.escape(init_date)}_(ens[^.]+)\.nc4$")
    for path in cache_dir.glob(f"tc_conditioned_ace_{init_date}_ens*.nc4"):
        if path.name.endswith("_ensmean.nc4"):
            continue
        match = pattern.match(path.name)
        if match:
            out.add(match.group(1))
    return out


@dataclass
class InitProgress:
    init_date: str
    expected_members: int
    member_caches: int
    ensmean_done: bool
    summary_done: bool

    @property
    def year(self) -> int:
        return int(self.init_date[:4])

    @property
    def member_complete(self) -> bool:
        return self.expected_members > 0 and self.member_caches >= self.expected_members

    @property
    def fully_complete(self) -> bool:
        return self.member_complete and self.ensmean_done


def summarize(args: argparse.Namespace) -> list[InitProgress]:
    sfc_root = Path(args.sfc_root)
    cache_dir = Path(args.cache_dir)
    init_dates = read_init_dates(Path(args.init_dates_file), parse_years(args.years))
    rows: list[InitProgress] = []
    for init_date in init_dates:
        expected = discover_members(sfc_root, init_date, args.ens)
        cached = cached_members(cache_dir, init_date)
        if not expected and cached:
            expected = sorted(cached)
        if args.expected_members_per_init > 0:
            expected_count = args.expected_members_per_init
        else:
            expected_count = len(expected)
        member_done = len(cached if not expected else cached.intersection(expected))
        rows.append(
            InitProgress(
                init_date=init_date,
                expected_members=expected_count,
                member_caches=member_done,
                ensmean_done=(cache_dir / f"tc_conditioned_ace_{init_date}_ensmean.nc4").is_file(),
                summary_done=(cache_dir / f"tc_conditioned_ace_{init_date}_member_summary.csv").is_file(),
            )
        )
    return rows


def print_summary(rows: list[InitProgress], args: argparse.Namespace) -> None:
    total_inits = len(rows)
    complete_inits = sum(row.fully_complete for row in rows)
    ensmean_done = sum(row.ensmean_done for row in rows)
    summaries_done = sum(row.summary_done for row in rows)
    total_expected_members = sum(row.expected_members for row in rows)
    total_member_caches = sum(min(row.member_caches, row.expected_members) for row in rows if row.expected_members > 0)
    years = sorted({row.year for row in rows})
    full_years = 0
    by_year: dict[int, list[InitProgress]] = defaultdict(list)
    for row in rows:
        by_year[row.year].append(row)
    for year in years:
        year_rows = by_year[year]
        if year_rows and all(row.fully_complete for row in year_rows):
            full_years += 1

    print("ACE cache progress")
    print(f"  cache_dir={args.cache_dir}")
    print(f"  init_dates_file={args.init_dates_file}")
    print(f"  fully complete init dates : {progress_bar(complete_inits, total_inits)}")
    print(f"  ensmean caches           : {progress_bar(ensmean_done, total_inits)}")
    print(f"  member summaries         : {progress_bar(summaries_done, total_inits)}")
    print(f"  member caches            : {progress_bar(total_member_caches, total_expected_members)}")
    print(f"  complete years           : {progress_bar(full_years, len(years))}")

    if args.compact:
        incomplete = [row for row in rows if not row.fully_complete]
        if incomplete:
            preview = ", ".join(row.init_date for row in incomplete[:10])
            more = "" if len(incomplete) <= 10 else f", ... +{len(incomplete) - 10} more"
            print(f"  next incomplete init dates: {preview}{more}")
        return

    print("")
    print("year  init_done  ensmean  member_caches")
    for year in years:
        year_rows = by_year[year]
        init_done = sum(row.fully_complete for row in year_rows)
        ens_done = sum(row.ensmean_done for row in year_rows)
        member_done = sum(min(row.member_caches, row.expected_members) for row in year_rows if row.expected_members > 0)
        member_total = sum(row.expected_members for row in year_rows)
        print(
            f"{year}  "
            f"{init_done:2d}/{len(year_rows):<2d}      "
            f"{ens_done:2d}/{len(year_rows):<2d}     "
            f"{progress_bar(member_done, member_total, width=24)}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sfc-root", default=DEFAULT_SFC_ROOT)
    parser.add_argument("--init-dates-file", default=DEFAULT_INIT_DATES_FILE)
    parser.add_argument("--years", default="1991:2024")
    parser.add_argument("--ens", default="all")
    parser.add_argument(
        "--expected-members-per-init",
        type=int,
        default=0,
        help="Optional fixed expected member count. Default discovers ensemble dirs from SFC root.",
    )
    parser.add_argument("--compact", action="store_true", help="Print only the top-level progress bars.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = summarize(args)
    if not rows:
        print("ERROR: no init dates selected")
        return 1
    print_summary(rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
