#!/usr/bin/env python3
"""Slim GEOS S2S3 SFC NetCDF files by keeping selected variables only.

The script rewrites each processed .nc4 file in place through a temporary file
in the same directory. Requested variables are kept when present, along with
coordinate/grid variables needed by those variables. A sidecar marker and a
manifest make reruns restart-safe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import stat
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_SFC_ROOT = "/nobackupp27/afahad/project/GEOS-S2S_TC/data"
DEFAULT_COLLECTION = "sfc_tavg_3hr_glo_L720x361_sfc"
DEFAULT_KEEP_VARIABLES = "QS,T2M,TS,US,VS"
DONE_SUFFIX = ".sfc_var_slim_done"
DONE_ATTR = "geos_s2s_tc_sfc_var_slim_done"
KEEP_ATTR = "geos_s2s_tc_sfc_var_slim_keep_variables"
TIME_LIMIT_EXIT = 75


@dataclass
class SlimResult:
    path: str
    status: str
    variables_before: int = 0
    variables_after: int = 0
    variables_kept: str = ""
    variables_missing: str = ""
    variables_removed: int = 0
    message: str = ""


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def normalize_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for name in names:
        key = name.upper()
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized


def keep_signature(names: list[str]) -> str:
    return ",".join(normalize_names(names))


def read_init_dates(path: str | None) -> set[str] | None:
    if not path:
        return None
    text = Path(path).read_text()
    dates = {item for item in re.split(r"\s+", text) if re.fullmatch(r"\d{8}", item)}
    return dates


def forecast_yyyymm(name: str, collection: str) -> str | None:
    collection_index = name.find(collection)
    search_text = name[collection_index + len(collection) :] if collection_index >= 0 else name
    match = re.search(r"(?:^|\.)(?:daily\.)?(\d{6})(?:\d{2})?(?:[_\.]|$)", search_text)
    if match:
        return match.group(1)
    return None


def discover_files(
    sfc_root: Path,
    collection: str,
    init_dates: set[str] | None,
    forecast_months: set[str] | None,
) -> list[Path]:
    geos_root = sfc_root / "GEOS_fcst"
    files: list[Path] = []

    for path in geos_root.glob(f"*/ens*/{collection}/*.nc4"):
        init_date = path.parent.parent.parent.name
        if init_dates is not None and init_date not in init_dates:
            continue

        if forecast_months:
            yyyymm = forecast_yyyymm(path.name, collection)
            if yyyymm is None or yyyymm[-2:] not in forecast_months:
                continue

        files.append(path)

    return sorted(files)


def marker_path(path: Path) -> Path:
    return Path(f"{path}{DONE_SUFFIX}")


def marker_is_current(path: Path, keep_vars: list[str]) -> bool:
    marker = marker_path(path)
    if not marker.exists() or marker.stat().st_mtime < path.stat().st_mtime:
        return False

    try:
        return f"keep_variables={keep_signature(keep_vars)}" in marker.read_text()
    except OSError:
        return False


def write_marker(path: Path, result: SlimResult, keep_vars: list[str]) -> None:
    marker_path(path).write_text(
        "\n".join(
            [
                f"time={datetime.now().isoformat(timespec='seconds')}",
                f"status={result.status}",
                f"keep_variables={keep_signature(keep_vars)}",
                f"variables_before={result.variables_before}",
                f"variables_after={result.variables_after}",
                f"variables_kept={result.variables_kept}",
                f"variables_missing={result.variables_missing}",
                f"variables_removed={result.variables_removed}",
                f"message={result.message}",
                "",
            ]
        )
    )


def safe_get_attr(obj, attr: str):
    try:
        return obj.getncattr(attr)
    except Exception:
        return None


def copy_attrs(src, dst, skip: set[str] | None = None) -> None:
    skip = skip or set()
    for attr in src.ncattrs():
        if attr in skip or attr == "_NCProperties":
            continue
        try:
            dst.setncattr(attr, src.getncattr(attr))
        except Exception as exc:
            print(f"WARN could not copy attribute {attr}: {exc}", file=sys.stderr)


def compression_kwargs(var, complevel: int) -> dict[str, object]:
    import numpy as np

    if not var.dimensions:
        return {}

    try:
        kind = np.dtype(var.dtype).kind
    except TypeError:
        return {}

    if kind == "O":
        return {}

    return {"zlib": True, "complevel": complevel, "shuffle": True}


def create_variable(dst, src_var, name: str, complevel: int):
    fill_value = safe_get_attr(src_var, "_FillValue")
    kwargs = compression_kwargs(src_var, complevel)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value

    out_var = dst.createVariable(name, src_var.datatype, src_var.dimensions, **kwargs)
    copy_attrs(src_var, out_var, skip={"_FillValue"})
    return out_var


def referenced_variables(src, var_name: str) -> set[str]:
    refs: set[str] = set()
    var = src.variables[var_name]

    for dim_name in var.dimensions:
        if dim_name in src.variables:
            refs.add(dim_name)

    for attr in ("coordinates", "bounds", "grid_mapping"):
        value = safe_get_attr(var, attr)
        if isinstance(value, str):
            for item in value.split():
                if item in src.variables:
                    refs.add(item)

    return refs


def expand_keep_set(src, keep_vars: list[str]) -> tuple[set[str], list[str], list[str]]:
    name_by_upper = {name.upper(): name for name in src.variables}
    requested_upper = normalize_names(keep_vars)
    requested_present = [name_by_upper[name] for name in requested_upper if name in name_by_upper]
    requested_missing = [name for name in requested_upper if name not in name_by_upper]

    keep_set: set[str] = set(requested_present)
    queue = list(requested_present)

    while queue:
        current = queue.pop()
        for ref_name in referenced_variables(src, current):
            if ref_name not in keep_set:
                keep_set.add(ref_name)
                queue.append(ref_name)

    return keep_set, requested_present, requested_missing


def file_has_done_attr(path: Path, keep_vars: list[str]) -> bool:
    from netCDF4 import Dataset

    signature = keep_signature(keep_vars)
    try:
        with Dataset(path, "r") as dataset:
            return (
                str(getattr(dataset, DONE_ATTR, "")).lower() == "true"
                and str(getattr(dataset, KEEP_ATTR, "")) == signature
            )
    except Exception:
        return False


def slim_file(path: Path, args) -> SlimResult:
    from netCDF4 import Dataset

    keep_vars = args.keep_variables

    if marker_is_current(path, keep_vars) and not args.force:
        return SlimResult(str(path), "skip_marker_current", message="sidecar marker is newer than file")

    if not args.force and file_has_done_attr(path, keep_vars):
        result = SlimResult(str(path), "skip_attr_done", message="file already has SFC slimming global attribute")
        write_marker(path, result, keep_vars)
        return result

    tmp_path: Path | None = None
    try:
        with Dataset(path, "r") as src:
            src.set_auto_maskandscale(False)
            keep_set, requested_present, requested_missing = expand_keep_set(src, keep_vars)
            variables_before = len(src.variables)

            if not requested_present:
                result = SlimResult(
                    str(path),
                    "skip_no_requested_vars",
                    variables_before=variables_before,
                    variables_missing=",".join(requested_missing),
                    message="none of the requested SFC variables were present",
                )
                write_marker(path, result, keep_vars)
                return result

            variables_after = len(keep_set)
            variables_removed = variables_before - variables_after
            variables_kept = ",".join(sorted(keep_set))
            variables_missing = ",".join(requested_missing)

            if args.dry_run:
                return SlimResult(
                    str(path),
                    "dry_run",
                    variables_before=variables_before,
                    variables_after=variables_after,
                    variables_kept=variables_kept,
                    variables_missing=variables_missing,
                    variables_removed=variables_removed,
                )

            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.sfc_slim.",
                suffix=".nc4",
                dir=str(path.parent),
            )
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)

            output_format = src.data_model if src.data_model in {"NETCDF4", "NETCDF4_CLASSIC"} else "NETCDF4"
            with Dataset(tmp_path, "w", format=output_format) as dst:
                copy_attrs(src, dst)
                dst.setncattr(DONE_ATTR, "true")
                dst.setncattr(KEEP_ATTR, keep_signature(keep_vars))
                dst.setncattr("geos_s2s_tc_sfc_var_slim_source", str(path))
                dst.setncattr("geos_s2s_tc_sfc_var_slim_time", datetime.utcnow().isoformat(timespec="seconds") + "Z")

                needed_dims = set()
                for name in keep_set:
                    needed_dims.update(src.variables[name].dimensions)

                for dim_name, dim in src.dimensions.items():
                    if dim_name not in needed_dims:
                        continue
                    dst.createDimension(dim_name, None if dim.isunlimited() else len(dim))

                for name in src.variables:
                    if name not in keep_set:
                        continue
                    out_var = create_variable(dst, src.variables[name], name, args.complevel)
                    src.variables[name].set_auto_maskandscale(False)
                    if src.variables[name].dimensions:
                        out_var[...] = src.variables[name][...]
                    else:
                        try:
                            out_var.assignValue(src.variables[name].getValue())
                        except Exception:
                            out_var[...] = src.variables[name][...]

            original_mode = stat.S_IMODE(path.stat().st_mode)
            os.replace(tmp_path, path)
            os.chmod(path, original_mode)
            tmp_path = None

            result = SlimResult(
                str(path),
                "processed",
                variables_before=variables_before,
                variables_after=variables_after,
                variables_kept=variables_kept,
                variables_missing=variables_missing,
                variables_removed=variables_removed,
            )
            write_marker(path, result, keep_vars)
            return result
    except Exception as exc:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        return SlimResult(str(path), "error", message=f"{exc}\n{traceback.format_exc(limit=8)}")


def write_manifest_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "time\tstatus\tfile\tvariables_before\tvariables_after\tvariables_removed\tvariables_kept\tvariables_missing\tmessage\n"
    )


def append_manifest(path: Path, result: SlimResult) -> None:
    clean_message = result.message.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    with path.open("a") as handle:
        handle.write(
            "\t".join(
                [
                    datetime.now().strftime("%F %T"),
                    result.status,
                    result.path,
                    str(result.variables_before),
                    str(result.variables_after),
                    str(result.variables_removed),
                    result.variables_kept,
                    result.variables_missing,
                    clean_message,
                ]
            )
            + "\n"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfc-root", default=os.environ.get("SFC_ROOT", DEFAULT_SFC_ROOT))
    parser.add_argument("--collection", default=os.environ.get("SFC_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--init-dates-file", default=os.environ.get("INIT_DATES_FILE"))
    parser.add_argument("--forecast-months", default=os.environ.get("FORECAST_MONTHS", ""))
    parser.add_argument("--keep-variables", default=os.environ.get("KEEP_VARIABLES", DEFAULT_KEEP_VARIABLES))
    parser.add_argument("--complevel", type=int, default=int(os.environ.get("COMPLEVEL", "4")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")))
    parser.add_argument("--manifest", default=os.environ.get("MANIFEST_FILE"))
    parser.add_argument("--max-files", type=int, default=int(os.environ.get("MAX_FILES", "0")))
    parser.add_argument(
        "--stop-after-seconds",
        type=int,
        default=int(os.environ.get("STOP_AFTER_SECONDS", os.environ.get("ELAPSED_LIMIT_SECONDS", "0"))),
        help="Stop before starting a new file after this many elapsed seconds; exits 75 when work remains.",
    )
    parser.add_argument(
        "--job-start-epoch",
        type=int,
        default=int(os.environ.get("JOB_START_EPOCH", "0")),
        help="Unix epoch seconds for PBS job start; used with --stop-after-seconds.",
    )
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "0") == "1")
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "0") == "1")

    args = parser.parse_args(argv)
    args.keep_variables = parse_list(args.keep_variables)
    if not args.keep_variables:
        parser.error("--keep-variables must contain at least one variable")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        import netCDF4  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        print(f"ERROR required Python package is missing: {exc}", file=sys.stderr)
        print("Load a Python environment with netCDF4 and numpy, or pass PYTHON_BIN to the PBS script.", file=sys.stderr)
        return 2

    sfc_root = Path(args.sfc_root)
    forecast_months = set(parse_list(args.forecast_months)) or None
    init_dates = read_init_dates(args.init_dates_file)
    manifest = Path(args.manifest) if args.manifest else sfc_root / "job_state" / "sfc_variable_slim_manifest.tsv"

    files = discover_files(sfc_root, args.collection, init_dates, forecast_months)
    if args.max_files > 0:
        files = files[: args.max_files]

    write_manifest_header(manifest)

    print(f"SFC_ROOT={sfc_root}")
    print(f"COLLECTION={args.collection}")
    print(f"KEEP_VARIABLES={keep_signature(args.keep_variables)}")
    print(f"FORECAST_MONTHS={','.join(sorted(forecast_months)) if forecast_months else 'all'}")
    print(f"INIT_DATES_FILE={args.init_dates_file or 'all'}")
    print(f"COMPLEVEL={args.complevel}")
    print(f"WORKERS={args.workers}")
    print(f"STOP_AFTER_SECONDS={args.stop_after_seconds}")
    print(f"JOB_START_EPOCH={args.job_start_epoch}")
    print(f"DRY_RUN={int(args.dry_run)}")
    print(f"MANIFEST={manifest}")
    print(f"FILES_FOUND={len(files)}")

    if not files:
        return 0

    if args.stop_after_seconds > 0 and args.workers > 1:
        print("WARN stop-after-seconds is enabled; forcing WORKERS=1 so the job can stop cleanly between files.")
        args.workers = 1

    run_start_monotonic = time.monotonic()
    stopped_for_time = False
    remaining_files = 0
    counts: dict[str, int] = {}

    if args.workers <= 1:
        for index, path in enumerate(files):
            if args.job_start_epoch > 0:
                elapsed_seconds = int(time.time() - args.job_start_epoch)
            else:
                elapsed_seconds = int(time.monotonic() - run_start_monotonic)

            if args.stop_after_seconds > 0 and elapsed_seconds >= args.stop_after_seconds:
                stopped_for_time = True
                remaining_files = len(files) - index
                print(
                    f"TIME_LIMIT_REACHED elapsed_seconds={elapsed_seconds} "
                    f"limit_seconds={args.stop_after_seconds} remaining_files={remaining_files}"
                )
                break

            result = slim_file(path, args)
            append_manifest(manifest, result)
            counts[result.status] = counts.get(result.status, 0) + 1
            print(
                f"{result.status} {result.path} "
                f"vars_before={result.variables_before} "
                f"vars_after={result.variables_after} "
                f"vars_removed={result.variables_removed} "
                f"kept={result.variables_kept or '-'} "
                f"missing={result.variables_missing or '-'}"
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_path = {executor.submit(slim_file, path, args): path for path in files}
            for future in concurrent.futures.as_completed(future_to_path):
                result = future.result()
                append_manifest(manifest, result)
                counts[result.status] = counts.get(result.status, 0) + 1
                print(
                    f"{result.status} {result.path} "
                    f"vars_before={result.variables_before} "
                    f"vars_after={result.variables_after} "
                    f"vars_removed={result.variables_removed} "
                    f"kept={result.variables_kept or '-'} "
                    f"missing={result.variables_missing or '-'}"
                )

    print("Summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}={count}")
    if stopped_for_time:
        print("  stopped_for_time=1")
        print(f"  remaining_files={remaining_files}")

    if counts.get("error", 0):
        return 1
    if stopped_for_time:
        return TIME_LIMIT_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
