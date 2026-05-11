#!/usr/bin/env python3
"""Summarize GEOS S2S3 collection download and untar coverage.

The script scans a local GEOS_fcst tree and reports, for each init date,
ensemble, and forecast month, whether the expected monthly file is missing,
downloaded as a tar, queued/submitted, or untarred/extracted. It can use a
reference tree, usually SFC, to define the expected init/ensemble universe for
another collection, usually ATM.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_ATM_ROOT = "/nobackupp17/afahad/GEOSS2S3_atm"
DEFAULT_ATM_COLLECTION = "atm_inst_6hr_glo_L720x361_p49"
DEFAULT_SFC_ROOT = "/nobackupp27/afahad/project/GEOS-S2S_TC/data"
DEFAULT_SFC_COLLECTION = "sfc_tavg_3hr_glo_L720x361_sfc"
DEFAULT_INIT_DATES_FILE = "/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt"


@dataclass(frozen=True)
class MonthStatus:
    status: str
    status_class: str
    local_dir: Path
    tar_path: Path
    untar_marker: Path
    combined_marker: Path
    shiftc_marker: Path
    extracted_files: int


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def read_init_dates(path: str | None) -> list[str]:
    if not path:
        return []
    text = Path(path).read_text()
    return [item for item in re.split(r"\s+", text) if re.fullmatch(r"\d{8}", item)]


def forecast_yyyymm(name: str, collection: str) -> str | None:
    collection_index = name.find(collection)
    search_text = name[collection_index + len(collection) :] if collection_index >= 0 else name
    match = re.search(r"(?:^|\.)(?:daily\.)?(\d{6})(?:\d{2})?(?:[_\.]|$)", search_text)
    return match.group(1) if match else None


def geos_root(root: Path) -> Path:
    return root / "GEOS_fcst"


def discover_init_dates(root: Path) -> list[str]:
    base = geos_root(root)
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir() and re.fullmatch(r"\d{8}", path.name))


def discover_global_ensembles(root: Path) -> list[str]:
    base = geos_root(root)
    ensembles: set[str] = set()
    if not base.exists():
        return []

    for path in base.glob("*/ens*"):
        if path.is_dir():
            ensembles.add(path.name)
    return sorted(ensembles, key=ensemble_sort_key)


def ensemble_sort_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"ens(\d+)", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def discover_ensembles_for_init(root: Path, init_date: str) -> list[str]:
    init_dir = geos_root(root) / init_date
    if not init_dir.exists():
        return []
    return sorted(
        (path.name for path in init_dir.iterdir() if path.is_dir() and path.name.startswith("ens")),
        key=ensemble_sort_key,
    )


def tar_name(init_date: str, collection: str, file_interval_tag: str, yyyymm: str) -> str:
    return f"{init_date}.{collection}.{file_interval_tag}.{yyyymm}.nc4.tar"


def extracted_nc4_files(local_dir: Path, init_date: str, collection: str, file_interval_tag: str, yyyymm: str) -> list[Path]:
    patterns = [
        f"{init_date}.{collection}.{file_interval_tag}.{yyyymm}*.nc4",
        f"{init_date}.{collection}.{yyyymm}*.nc4",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in local_dir.glob(pattern) if path.is_file())
    return sorted(set(files))


def month_status(
    root: Path,
    init_date: str,
    ens: str,
    collection: str,
    file_interval_tag: str,
    yyyymm: str,
) -> MonthStatus:
    local_dir = geos_root(root) / init_date / ens / collection
    tar_path = local_dir / tar_name(init_date, collection, file_interval_tag, yyyymm)
    untar_marker = Path(f"{tar_path}.untar_done")
    combined_marker = Path(f"{tar_path}.untar_slim_done")
    shiftc_marker = Path(f"{tar_path}.shiftc_submitted")
    nc4_files = extracted_nc4_files(local_dir, init_date, collection, file_interval_tag, yyyymm)

    if untar_marker.exists():
        status = "untar_done_marker"
        status_class = "untarred"
    elif combined_marker.exists():
        status = "untar_slim_done_marker"
        status_class = "untarred"
    elif nc4_files:
        status = "extracted_nc4"
        status_class = "untarred"
    elif tar_path.exists():
        status = "tar_present"
        status_class = "downloaded_not_untarred"
    elif shiftc_marker.exists():
        status = "shiftc_submitted"
        status_class = "submitted_not_downloaded"
    elif local_dir.exists():
        status = "collection_dir_only"
        status_class = "missing"
    else:
        status = "missing"
        status_class = "missing"

    return MonthStatus(
        status=status,
        status_class=status_class,
        local_dir=local_dir,
        tar_path=tar_path,
        untar_marker=untar_marker,
        combined_marker=combined_marker,
        shiftc_marker=shiftc_marker,
        extracted_files=len(nc4_files),
    )


def status_for_counts(counts: Counter[str], expected_total: int) -> str:
    if expected_total == 0:
        return "no_expected_items"
    if counts["untarred"] == expected_total:
        return "complete_untarred"
    if counts["missing"] == expected_total:
        return "missing_all"
    if counts["missing"] == 0 and counts["submitted_not_downloaded"] == 0:
        return "all_downloaded_some_not_untarred"
    return "partial"


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["atm", "sfc", "custom"],
        default=os.environ.get("DATASET", "atm"),
        help="Preset root/collection defaults. Explicit --data-root or --collection overrides this.",
    )
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--collection", default=os.environ.get("COLLECTION"))
    parser.add_argument("--reference-root", default=os.environ.get("REFERENCE_ROOT"))
    parser.add_argument("--reference-collection", default=os.environ.get("REFERENCE_COLLECTION"))
    parser.add_argument("--init-dates-file", default=os.environ.get("INIT_DATES_FILE", DEFAULT_INIT_DATES_FILE))
    parser.add_argument("--forecast-months", default=os.environ.get("FORECAST_MONTHS", "09 10"))
    parser.add_argument("--ensembles", default=os.environ.get("ENSEMBLES", ""))
    parser.add_argument("--file-interval-tag", default=os.environ.get("FILE_INTERVAL_TAG", "daily"))
    parser.add_argument("--report-dir", default=os.environ.get("REPORT_DIR"))
    parser.add_argument("--report-prefix", default=os.environ.get("REPORT_PREFIX", "collection_status"))
    return parser.parse_args()


def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.dataset == "sfc":
        args.data_root = args.data_root or DEFAULT_SFC_ROOT
        args.collection = args.collection or DEFAULT_SFC_COLLECTION
        args.report_prefix = "sfc_status" if args.report_prefix == "collection_status" else args.report_prefix
    elif args.dataset == "atm":
        args.data_root = args.data_root or DEFAULT_ATM_ROOT
        args.collection = args.collection or DEFAULT_ATM_COLLECTION
        args.reference_root = args.reference_root or DEFAULT_SFC_ROOT
        args.reference_collection = args.reference_collection or DEFAULT_SFC_COLLECTION
        args.report_prefix = "atm_status" if args.report_prefix == "collection_status" else args.report_prefix
    else:
        if not args.data_root or not args.collection:
            raise SystemExit("ERROR custom dataset requires --data-root and --collection")

    args.reference_collection = args.reference_collection or args.collection
    args.report_dir = args.report_dir or str(Path(args.data_root) / "reports")
    return args


def expected_ensembles_by_init(
    init_dates: list[str],
    data_root: Path,
    reference_root: Path | None,
    explicit_ensembles: list[str],
) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}
    reference_global = discover_global_ensembles(reference_root) if reference_root else []
    data_global = discover_global_ensembles(data_root)

    for init_date in init_dates:
        if explicit_ensembles:
            ensembles = explicit_ensembles
        elif reference_root:
            ensembles = discover_ensembles_for_init(reference_root, init_date) or reference_global
        else:
            ensembles = discover_ensembles_for_init(data_root, init_date) or data_global
        expected[init_date] = ensembles

    return expected


def main() -> int:
    args = apply_dataset_defaults(parse_args())
    data_root = Path(args.data_root)
    reference_root = Path(args.reference_root) if args.reference_root else None
    forecast_months = parse_list(args.forecast_months)
    explicit_ensembles = parse_list(args.ensembles)
    init_dates = read_init_dates(args.init_dates_file) or discover_init_dates(reference_root or data_root)

    if not init_dates:
        print("ERROR no init dates found", flush=True)
        return 2
    if not forecast_months:
        print("ERROR no forecast months configured", flush=True)
        return 3

    report_dir = Path(args.report_dir)
    summary_file = report_dir / f"{args.report_prefix}_summary.txt"
    detail_file = report_dir / f"{args.report_prefix}_detail.tsv"
    init_file = report_dir / f"{args.report_prefix}_init_summary.tsv"
    missing_file = report_dir / f"{args.report_prefix}_missing.tsv"
    downloaded_file = report_dir / f"{args.report_prefix}_downloaded_not_untarred.tsv"

    ensembles_by_init = expected_ensembles_by_init(init_dates, data_root, reference_root, explicit_ensembles)
    detail_lines = [
        "init_date\tens\tforecast_month\tyyyymm\tstatus\tstatus_class\tlocal_dir\ttar_path\t"
        "extracted_files\tuntar_marker\tuntar_slim_marker\tshiftc_marker"
    ]
    init_lines = [
        "init_date\texpected_items\tuntarred\tdownloaded_not_untarred\tsubmitted_not_downloaded\t"
        "missing\tstatus\texpected_ensembles"
    ]
    missing_lines = ["init_date\tens\tforecast_month\tyyyymm\tstatus\tlocal_dir\ttar_path"]
    downloaded_lines = ["init_date\tens\tforecast_month\tyyyymm\tstatus\tlocal_dir\ttar_path"]

    global_counts: Counter[str] = Counter()
    init_status_counts: Counter[str] = Counter()
    expected_total_all = 0
    expected_ensembles_missing = 0

    for init_date in init_dates:
        init_year = init_date[:4]
        ensembles = ensembles_by_init[init_date]
        if not ensembles:
            expected_ensembles_missing += 1

        init_counts: Counter[str] = Counter()
        expected_total = 0

        for ens in ensembles:
            for forecast_month in forecast_months:
                yyyymm = f"{init_year}{forecast_month}"
                status = month_status(
                    data_root,
                    init_date,
                    ens,
                    args.collection,
                    args.file_interval_tag,
                    yyyymm,
                )
                expected_total += 1
                expected_total_all += 1
                init_counts[status.status_class] += 1
                global_counts[status.status_class] += 1

                detail_lines.append(
                    "\t".join(
                        [
                            init_date,
                            ens,
                            forecast_month,
                            yyyymm,
                            status.status,
                            status.status_class,
                            str(status.local_dir),
                            str(status.tar_path),
                            str(status.extracted_files),
                            str(status.untar_marker),
                            str(status.combined_marker),
                            str(status.shiftc_marker),
                        ]
                    )
                )

                if status.status_class == "missing":
                    missing_lines.append(
                        "\t".join(
                            [
                                init_date,
                                ens,
                                forecast_month,
                                yyyymm,
                                status.status,
                                str(status.local_dir),
                                str(status.tar_path),
                            ]
                        )
                    )
                elif status.status_class == "downloaded_not_untarred":
                    downloaded_lines.append(
                        "\t".join(
                            [
                                init_date,
                                ens,
                                forecast_month,
                                yyyymm,
                                status.status,
                                str(status.local_dir),
                                str(status.tar_path),
                            ]
                        )
                    )

        init_status = status_for_counts(init_counts, expected_total)
        init_status_counts[init_status] += 1
        init_lines.append(
            "\t".join(
                [
                    init_date,
                    str(expected_total),
                    str(init_counts["untarred"]),
                    str(init_counts["downloaded_not_untarred"]),
                    str(init_counts["submitted_not_downloaded"]),
                    str(init_counts["missing"]),
                    init_status,
                    ",".join(ensembles),
                ]
            )
        )

    generated = datetime.now().strftime("%F %T")
    summary_lines = [
        f"Generated: {generated}",
        f"DATASET={args.dataset}",
        f"DATA_ROOT={data_root}",
        f"COLLECTION={args.collection}",
        f"REFERENCE_ROOT={reference_root or ''}",
        f"REFERENCE_COLLECTION={args.reference_collection or ''}",
        f"INIT_DATES_FILE={args.init_dates_file}",
        f"INIT_DATES={len(init_dates)}",
        f"FORECAST_MONTHS={' '.join(forecast_months)}",
        f"EXPECTED_ITEMS={expected_total_all}",
        f"UNTARRED={global_counts['untarred']}",
        f"DOWNLOADED_NOT_UNTARRED={global_counts['downloaded_not_untarred']}",
        f"SUBMITTED_NOT_DOWNLOADED={global_counts['submitted_not_downloaded']}",
        f"MISSING={global_counts['missing']}",
        f"INIT_COMPLETE_UNTARRED={init_status_counts['complete_untarred']}",
        f"INIT_ALL_DOWNLOADED_SOME_NOT_UNTARRED={init_status_counts['all_downloaded_some_not_untarred']}",
        f"INIT_PARTIAL={init_status_counts['partial']}",
        f"INIT_MISSING_ALL={init_status_counts['missing_all']}",
        f"INIT_NO_EXPECTED_ITEMS={init_status_counts['no_expected_items']}",
        f"INIT_WITH_NO_EXPECTED_ENSEMBLES={expected_ensembles_missing}",
        f"SUMMARY_FILE={summary_file}",
        f"DETAIL_FILE={detail_file}",
        f"INIT_SUMMARY_FILE={init_file}",
        f"MISSING_FILE={missing_file}",
        f"DOWNLOADED_NOT_UNTARRED_FILE={downloaded_file}",
    ]

    write_lines(summary_file, summary_lines)
    write_lines(detail_file, detail_lines)
    write_lines(init_file, init_lines)
    write_lines(missing_file, missing_lines)
    write_lines(downloaded_file, downloaded_lines)

    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
