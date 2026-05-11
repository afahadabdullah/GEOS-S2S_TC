#!/usr/bin/env python3
"""Untar GEOS S2S3 ATM monthly tar files and slim extracted files in one pass.

Each monthly tar is extracted into its existing collection directory, then each
extracted .nc4 file is slimmed in place to the requested pressure levels. The
script writes tar-level and file-level manifests plus sidecar markers so reruns
can resume safely after walltime limits or interruptions.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from slim_atm_vertical_levels import (  # noqa: E402
    DEFAULT_ATM_ROOT,
    DEFAULT_COLLECTION,
    DEFAULT_LEVELS_HPA,
    TIME_LIMIT_EXIT,
    append_manifest as append_slim_manifest,
    forecast_yyyymm,
    parse_float_list,
    parse_list,
    read_init_dates,
    slim_file,
    write_manifest_header as write_slim_manifest_header,
)

UNTAR_DONE_SUFFIX = ".untar_done"
COMBINED_DONE_SUFFIX = ".untar_slim_done"


@dataclass
class TarResult:
    path: str
    status: str
    init_date: str = ""
    ens: str = ""
    forecast_month: str = ""
    untar_status: str = ""
    members_found: int = 0
    members_extracted: int = 0
    slim_processed: int = 0
    slim_skipped: int = 0
    slim_errors: int = 0
    tar_deleted: int = 0
    message: str = ""


def discover_tars(
    atm_root: Path,
    collection: str,
    init_dates: set[str] | None,
    forecast_months: set[str] | None,
) -> list[Path]:
    geos_root = atm_root / "GEOS_fcst"
    tars: list[Path] = []

    for path in geos_root.glob(f"*/ens*/{collection}/*.tar"):
        if not path.name.endswith(".nc4.tar"):
            continue

        init_date = path.parent.parent.parent.name
        if init_dates is not None and init_date not in init_dates:
            continue

        if forecast_months:
            yyyymm = forecast_yyyymm(path.name, collection)
            if yyyymm is None or yyyymm[-2:] not in forecast_months:
                continue

        tars.append(path)

    return sorted(tars)


def sidecar(path: Path, suffix: str) -> Path:
    return Path(f"{path}{suffix}")


def write_text_marker(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines + [""]))


def combined_marker_is_current(tar_path: Path, levels_hpa: list[float]) -> bool:
    marker = sidecar(tar_path, COMBINED_DONE_SUFFIX)
    if not marker.exists() or marker.stat().st_mtime < tar_path.stat().st_mtime:
        return False

    level_text = ",".join(f"{level:g}" for level in levels_hpa)
    try:
        return f"levels_hpa={level_text}" in marker.read_text()
    except OSError:
        return False


def write_combined_marker(tar_path: Path, result: TarResult, levels_hpa: list[float]) -> None:
    level_text = ",".join(f"{level:g}" for level in levels_hpa)
    write_text_marker(
        sidecar(tar_path, COMBINED_DONE_SUFFIX),
        [
            f"time={datetime.now().isoformat(timespec='seconds')}",
            f"status={result.status}",
            f"levels_hpa={level_text}",
            f"untar_status={result.untar_status}",
            f"members_found={result.members_found}",
            f"members_extracted={result.members_extracted}",
            f"slim_processed={result.slim_processed}",
            f"slim_skipped={result.slim_skipped}",
            f"slim_errors={result.slim_errors}",
            f"tar_deleted={result.tar_deleted}",
            f"message={result.message}",
        ],
    )


def write_untar_marker(tar_path: Path, members: list[Path], status: str) -> None:
    write_text_marker(
        sidecar(tar_path, UNTAR_DONE_SUFFIX),
        [
            f"time={datetime.now().isoformat(timespec='seconds')}",
            f"status={status}",
            f"members={len(members)}",
            "member_paths=" + ",".join(str(path) for path in members),
        ],
    )


def clean_message(message: str) -> str:
    return message.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def write_tar_manifest_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "time\tstatus\ttar\tinit_date\tens\tforecast_month\tuntar_status\t"
        "members_found\tmembers_extracted\tslim_processed\tslim_skipped\t"
        "slim_errors\ttar_deleted\tmessage\n"
    )


def append_tar_manifest(path: Path, result: TarResult) -> None:
    with path.open("a") as handle:
        handle.write(
            "\t".join(
                [
                    datetime.now().strftime("%F %T"),
                    result.status,
                    result.path,
                    result.init_date,
                    result.ens,
                    result.forecast_month,
                    result.untar_status,
                    str(result.members_found),
                    str(result.members_extracted),
                    str(result.slim_processed),
                    str(result.slim_skipped),
                    str(result.slim_errors),
                    str(result.tar_deleted),
                    clean_message(result.message),
                ]
            )
            + "\n"
        )


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def tar_members(tar_path: Path) -> tuple[list[str], str]:
    completed = run_command(["tar", "-tf", str(tar_path)])
    if completed.returncode != 0:
        message = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
        raise RuntimeError(f"tar listing failed with status {completed.returncode}: {message}")

    members = [line.strip() for line in completed.stdout.splitlines() if line.strip().endswith(".nc4")]
    return members, completed.stdout


def member_to_path(base_dir: Path, member: str) -> Path:
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe tar member path: {member}")
    return base_dir.joinpath(*pure.parts)


def extract_tar(tar_path: Path) -> None:
    completed = run_command(["tar", "-xf", str(tar_path), "-C", str(tar_path.parent)])
    if completed.returncode != 0:
        message = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
        raise RuntimeError(f"tar extract failed with status {completed.returncode}: {message}")


def elapsed_seconds(args: argparse.Namespace, run_start_monotonic: float) -> int:
    if args.job_start_epoch > 0:
        return int(time.time() - args.job_start_epoch)
    return int(time.monotonic() - run_start_monotonic)


def time_limit_reached(args: argparse.Namespace, run_start_monotonic: float) -> bool:
    return args.stop_after_seconds > 0 and elapsed_seconds(args, run_start_monotonic) >= args.stop_after_seconds


def tar_metadata(tar_path: Path, collection: str) -> tuple[str, str, str]:
    init_date = tar_path.parent.parent.parent.name
    ens = tar_path.parent.parent.name
    yyyymm = forecast_yyyymm(tar_path.name, collection) or ""
    forecast_month = yyyymm[-2:] if yyyymm else ""
    return init_date, ens, forecast_month


def process_tar(
    tar_path: Path,
    args: argparse.Namespace,
    tar_manifest: Path,
    slim_manifest: Path,
    run_start_monotonic: float,
) -> tuple[TarResult, bool]:
    init_date, ens, forecast_month = tar_metadata(tar_path, args.collection)
    result = TarResult(str(tar_path), "started", init_date=init_date, ens=ens, forecast_month=forecast_month)

    if combined_marker_is_current(tar_path, args.levels_hpa) and not args.force:
        result.status = "skip_combined_marker_current"
        result.message = "tar already untarred and slimmed with requested levels"
        return result, False

    try:
        members_raw, _ = tar_members(tar_path)
        member_paths = [member_to_path(tar_path.parent, member) for member in members_raw]
        result.members_found = len(member_paths)

        if not member_paths:
            result.status = "error"
            result.message = "tar contains no .nc4 members"
            return result, False

        untar_marker = sidecar(tar_path, UNTAR_DONE_SUFFIX)
        all_members_present = all(path.exists() for path in member_paths)

        if args.dry_run:
            result.status = "dry_run"
            result.untar_status = "dry_run"
            result.message = f"would untar and slim {len(member_paths)} .nc4 files"
            return result, False

        if all_members_present and untar_marker.exists() and not args.force_untar:
            result.untar_status = "skip_untar_marker_present"
        elif all_members_present and not args.force_untar:
            result.untar_status = "skip_untar_members_present"
            write_untar_marker(tar_path, member_paths, result.untar_status)
        else:
            if time_limit_reached(args, run_start_monotonic):
                result.status = "partial_time_limit"
                result.untar_status = "not_started_time_limit"
                return result, True

            print(f"UNTAR {tar_path}", flush=True)
            extract_tar(tar_path)
            missing_after_extract = [path for path in member_paths if not path.exists()]
            if missing_after_extract:
                result.status = "error"
                result.untar_status = "untar_missing_members"
                result.message = "missing after untar: " + ",".join(str(path) for path in missing_after_extract[:10])
                return result, False
            result.untar_status = "untar_processed"
            result.members_extracted = len(member_paths)
            write_untar_marker(tar_path, member_paths, result.untar_status)

        slim_args = argparse.Namespace(
            levels_hpa=args.levels_hpa,
            tolerance_hpa=args.tolerance_hpa,
            complevel=args.complevel,
            force=args.force,
            dry_run=args.dry_run,
        )

        for index, member_path in enumerate(member_paths):
            if time_limit_reached(args, run_start_monotonic):
                result.status = "partial_time_limit"
                result.message = f"stopped before slimming member {index + 1} of {len(member_paths)}"
                return result, True

            slim_result = slim_file(member_path, slim_args)
            append_slim_manifest(slim_manifest, slim_result)

            if slim_result.status == "processed":
                result.slim_processed += 1
            elif slim_result.status == "error":
                result.slim_errors += 1
            else:
                result.slim_skipped += 1

            print(
                f"{slim_result.status} {slim_result.path} "
                f"levels_before={slim_result.levels_before or '-'} "
                f"levels_after={slim_result.levels_after or '-'} "
                f"vars_trimmed={slim_result.variables_trimmed} {slim_result.message}",
                flush=True,
            )

        if result.slim_errors:
            result.status = "error"
            result.message = "one or more extracted files failed during slimming"
            return result, False

        result.status = "processed"
        result.message = "untar and slimming complete"

        if args.delete_tar_after_success:
            tar_path.unlink()
            result.tar_deleted = 1
            result.message += "; tar deleted"

        if tar_path.exists():
            write_combined_marker(tar_path, result, args.levels_hpa)
        return result, False
    except Exception as exc:
        result.status = "error"
        result.message = f"{exc}\n{traceback.format_exc(limit=8)}"
        return result, False


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atm-root", default=os.environ.get("ATM_ROOT", DEFAULT_ATM_ROOT))
    parser.add_argument("--collection", default=os.environ.get("ATM_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--init-dates-file", default=os.environ.get("INIT_DATES_FILE"))
    parser.add_argument("--forecast-months", default=os.environ.get("FORECAST_MONTHS", ""))
    parser.add_argument("--levels-hpa", default=os.environ.get("LEVELS_HPA", DEFAULT_LEVELS_HPA))
    parser.add_argument("--tolerance-hpa", type=float, default=float(os.environ.get("LEVEL_TOLERANCE_HPA", "0.5")))
    parser.add_argument("--complevel", type=int, default=int(os.environ.get("COMPLEVEL", "4")))
    parser.add_argument("--tar-manifest", default=os.environ.get("TAR_MANIFEST_FILE"))
    parser.add_argument("--slim-manifest", default=os.environ.get("SLIM_MANIFEST_FILE"))
    parser.add_argument("--max-tars", type=int, default=int(os.environ.get("MAX_TARS", "0")))
    parser.add_argument(
        "--stop-after-seconds",
        type=int,
        default=int(os.environ.get("STOP_AFTER_SECONDS", os.environ.get("ELAPSED_LIMIT_SECONDS", "0"))),
        help="Stop before starting new work after this many elapsed seconds; exits 75 when work remains.",
    )
    parser.add_argument(
        "--job-start-epoch",
        type=int,
        default=int(os.environ.get("JOB_START_EPOCH", "0")),
        help="Unix epoch seconds for PBS job start; used with --stop-after-seconds.",
    )
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "0") == "1")
    parser.add_argument("--force-untar", action="store_true", default=os.environ.get("FORCE_UNTAR", "0") == "1")
    parser.add_argument(
        "--delete-tar-after-success",
        action="store_true",
        default=os.environ.get("DELETE_TAR_AFTER_SUCCESS", "0") == "1",
    )
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "0") == "1")

    args = parser.parse_args(argv)
    args.levels_hpa = parse_float_list(args.levels_hpa)
    if not args.levels_hpa:
        parser.error("--levels-hpa must contain at least one level")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        import netCDF4  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        print(f"ERROR required Python package is missing: {exc}", file=sys.stderr)
        return 2

    atm_root = Path(args.atm_root)
    init_dates = read_init_dates(args.init_dates_file)
    forecast_months = set(parse_list(args.forecast_months)) or None
    tar_manifest = Path(args.tar_manifest) if args.tar_manifest else atm_root / "job_state" / "atm_untar_slim_tar_manifest.tsv"
    slim_manifest = Path(args.slim_manifest) if args.slim_manifest else atm_root / "job_state" / "atm_untar_slim_file_manifest.tsv"

    tars = discover_tars(atm_root, args.collection, init_dates, forecast_months)
    if args.max_tars > 0:
        tars = tars[: args.max_tars]

    write_tar_manifest_header(tar_manifest)
    write_slim_manifest_header(slim_manifest)

    print(f"ATM_ROOT={atm_root}")
    print(f"COLLECTION={args.collection}")
    print(f"LEVELS_HPA={','.join(f'{level:g}' for level in args.levels_hpa)}")
    print(f"FORECAST_MONTHS={','.join(sorted(forecast_months)) if forecast_months else 'all'}")
    print(f"INIT_DATES_FILE={args.init_dates_file or 'all'}")
    print(f"COMPLEVEL={args.complevel}")
    print(f"MAX_TARS={args.max_tars}")
    print(f"STOP_AFTER_SECONDS={args.stop_after_seconds}")
    print(f"JOB_START_EPOCH={args.job_start_epoch}")
    print(f"FORCE={int(args.force)}")
    print(f"FORCE_UNTAR={int(args.force_untar)}")
    print(f"DELETE_TAR_AFTER_SUCCESS={int(args.delete_tar_after_success)}")
    print(f"DRY_RUN={int(args.dry_run)}")
    print(f"TAR_MANIFEST={tar_manifest}")
    print(f"SLIM_MANIFEST={slim_manifest}")
    print(f"TARS_FOUND={len(tars)}")

    if not tars:
        return 0

    run_start_monotonic = time.monotonic()
    stopped_for_time = False
    remaining_tars = 0
    counts: dict[str, int] = {}

    for index, tar_path in enumerate(tars):
        if time_limit_reached(args, run_start_monotonic):
            stopped_for_time = True
            remaining_tars = len(tars) - index
            print(
                f"TIME_LIMIT_REACHED elapsed_seconds={elapsed_seconds(args, run_start_monotonic)} "
                f"limit_seconds={args.stop_after_seconds} remaining_tars={remaining_tars}"
            )
            break

        result, stopped_inside_tar = process_tar(tar_path, args, tar_manifest, slim_manifest, run_start_monotonic)
        append_tar_manifest(tar_manifest, result)
        counts[result.status] = counts.get(result.status, 0) + 1
        print(
            f"{result.status} {result.path} untar_status={result.untar_status or '-'} "
            f"members={result.members_found} extracted={result.members_extracted} "
            f"slim_processed={result.slim_processed} slim_skipped={result.slim_skipped} "
            f"slim_errors={result.slim_errors} tar_deleted={result.tar_deleted} {result.message}",
            flush=True,
        )

        if stopped_inside_tar:
            stopped_for_time = True
            remaining_tars = len(tars) - index
            break

    print("Summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}={count}")
    if stopped_for_time:
        print("  stopped_for_time=1")
        print(f"  remaining_tars={remaining_tars}")

    if counts.get("error", 0):
        return 1
    if stopped_for_time:
        return TIME_LIMIT_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
