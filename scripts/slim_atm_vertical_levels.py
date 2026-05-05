#!/usr/bin/env python3
"""Slim GEOS S2S3 ATM NetCDF files by keeping selected pressure levels.

The script rewrites each processed .nc4 file in place through a temporary file
in the same directory. Variables without a vertical-pressure dimension are
copied unchanged; variables with the vertical dimension are subset to the
requested levels. A sidecar marker and a manifest make reruns restart-safe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import stat
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_ATM_ROOT = "/nobackupp27/afahad/GEOSS2S3_atm"
DEFAULT_COLLECTION = "atm_inst_6hr_glo_L720x361_p49"
DEFAULT_LEVELS_HPA = "1000,950,850,500,200"
DONE_SUFFIX = ".vertical_slim_done"
DONE_ATTR = "geos_s2s_tc_vertical_slim_done"
LEVELS_ATTR = "geos_s2s_tc_vertical_slim_levels_hpa"


@dataclass
class SlimResult:
    path: str
    status: str
    vertical_dims: str = ""
    levels_before: str = ""
    levels_after: str = ""
    variables_trimmed: int = 0
    message: str = ""


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,:]+", value.strip()) if item]


def parse_float_list(value: str | None) -> list[float]:
    return [float(item) for item in parse_list(value)]


def read_init_dates(path: str | None) -> set[str] | None:
    if not path:
        return None
    text = Path(path).read_text()
    dates = {item for item in re.split(r"\s+", text) if re.fullmatch(r"\d{8}", item)}
    return dates


def forecast_yyyymm(name: str, collection: str) -> str | None:
    collection_index = name.find(collection)
    search_text = name[collection_index + len(collection) :] if collection_index >= 0 else name

    # Matches both monthly names like ".daily.199510.nc4" and extracted
    # 6-hourly names like ".19951001_0300z.nc4".
    match = re.search(r"(?:^|\.)(?:daily\.)?(\d{6})(?:\d{2})?(?:[_\.]|$)", search_text)
    if match:
        return match.group(1)
    return None


def discover_files(
    atm_root: Path,
    collection: str,
    init_dates: set[str] | None,
    forecast_months: set[str] | None,
) -> list[Path]:
    geos_root = atm_root / "GEOS_fcst"
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


def marker_is_current(path: Path, levels_hpa: Iterable[float]) -> bool:
    marker = marker_path(path)
    if not marker.exists() or marker.stat().st_mtime < path.stat().st_mtime:
        return False

    level_text = ",".join(f"{level:g}" for level in levels_hpa)
    try:
        return f"levels_hpa={level_text}" in marker.read_text()
    except OSError:
        return False


def write_marker(path: Path, result: SlimResult, levels_hpa: Iterable[float]) -> None:
    marker = marker_path(path)
    level_text = ",".join(f"{level:g}" for level in levels_hpa)
    marker.write_text(
        "\n".join(
            [
                f"time={datetime.now().isoformat(timespec='seconds')}",
                f"status={result.status}",
                f"levels_hpa={level_text}",
                f"vertical_dims={result.vertical_dims}",
                f"levels_before={result.levels_before}",
                f"levels_after={result.levels_after}",
                f"variables_trimmed={result.variables_trimmed}",
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


def coord_values_hpa(var) -> list[float] | None:
    import numpy as np

    try:
        values = np.asarray(var[:], dtype="float64").reshape(-1)
    except Exception:
        return None

    if values.size == 0:
        return None

    units = str(getattr(var, "units", "")).lower()
    if "pa" in units and "hpa" not in units and "mbar" not in units and "millibar" not in units:
        values = values / 100.0
    elif float(np.nanmax(np.abs(values))) > 2000.0:
        # Pressure coordinates without useful units are often in Pa.
        values = values / 100.0

    return values.tolist()


def match_level_indices(values_hpa: list[float], target_hpa: list[float], tolerance_hpa: float) -> list[int] | None:
    import numpy as np

    values = np.asarray(values_hpa, dtype="float64")
    indices: list[int] = []

    for target in target_hpa:
        differences = np.abs(values - target)
        index = int(np.nanargmin(differences))
        if float(differences[index]) > tolerance_hpa:
            return None
        indices.append(index)

    # Preserve the file's native vertical order and avoid duplicate indices.
    return sorted(set(indices))


def find_vertical_dims(dataset, target_hpa: list[float], tolerance_hpa: float) -> dict[str, dict[str, object]]:
    vertical_dims: dict[str, dict[str, object]] = {}

    for dim_name, dim in dataset.dimensions.items():
        if dim_name not in dataset.variables:
            continue

        coord = dataset.variables[dim_name]
        if len(coord.dimensions) != 1 or coord.dimensions[0] != dim_name:
            continue

        values = coord_values_hpa(coord)
        if values is None or len(values) != len(dim):
            continue

        indices = match_level_indices(values, target_hpa, tolerance_hpa)
        if indices is None or len(indices) != len(target_hpa):
            continue

        vertical_dims[dim_name] = {
            "indices": indices,
            "values_hpa": values,
            "selected_hpa": [values[index] for index in indices],
            "before": len(dim),
            "after": len(indices),
        }

    return vertical_dims


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


def source_slices(dimensions: tuple[str, ...], vertical_dims: dict[str, dict[str, object]]) -> tuple[object, ...]:
    slices: list[object] = []
    for dim_name in dimensions:
        if dim_name in vertical_dims:
            slices.append(vertical_dims[dim_name]["indices"])
        else:
            slices.append(slice(None))
    return tuple(slices)


def copy_variable_data(src_var, dst_var, vertical_dims: dict[str, dict[str, object]]) -> None:
    src_var.set_auto_maskandscale(False)

    if not src_var.dimensions:
        try:
            dst_var.assignValue(src_var.getValue())
        except Exception:
            dst_var[...] = src_var[...]
        return

    slices = source_slices(src_var.dimensions, vertical_dims)
    dst_var[...] = src_var[slices]


def file_has_done_attr(path: Path, target_hpa: list[float]) -> bool:
    from netCDF4 import Dataset

    level_text = ",".join(f"{level:g}" for level in target_hpa)
    try:
        with Dataset(path, "r") as dataset:
            return (
                str(getattr(dataset, DONE_ATTR, "")).lower() == "true"
                and str(getattr(dataset, LEVELS_ATTR, "")) == level_text
            )
    except Exception:
        return False


def slim_file(path: Path, args) -> SlimResult:
    from netCDF4 import Dataset

    target_hpa = args.levels_hpa
    level_text = ",".join(f"{level:g}" for level in target_hpa)

    if marker_is_current(path, target_hpa) and not args.force:
        return SlimResult(str(path), "skip_marker_current", message="sidecar marker is newer than file")

    if not args.force and file_has_done_attr(path, target_hpa):
        result = SlimResult(str(path), "skip_attr_done", message="file already has slimming global attribute")
        write_marker(path, result, target_hpa)
        return result

    tmp_path: Path | None = None
    try:
        with Dataset(path, "r") as src:
            src.set_auto_maskandscale(False)
            vertical_dims = find_vertical_dims(src, target_hpa, args.tolerance_hpa)

            if not vertical_dims:
                result = SlimResult(str(path), "skip_no_vertical_dim", message="no matching pressure-level coordinate")
                write_marker(path, result, target_hpa)
                return result

            variables_trimmed = sum(
                1
                for var in src.variables.values()
                if any(dim_name in vertical_dims for dim_name in var.dimensions)
            )
            levels_before = ",".join(f"{name}:{info['before']}" for name, info in vertical_dims.items())
            levels_after = ",".join(f"{name}:{info['after']}" for name, info in vertical_dims.items())
            selected = ";".join(
                f"{name}:{','.join(f'{value:g}' for value in info['selected_hpa'])}"
                for name, info in vertical_dims.items()
            )

            if args.dry_run:
                return SlimResult(
                    str(path),
                    "dry_run",
                    vertical_dims=",".join(vertical_dims),
                    levels_before=levels_before,
                    levels_after=levels_after,
                    variables_trimmed=variables_trimmed,
                    message=f"selected_hpa={selected}",
                )

            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.slim.",
                suffix=".nc4",
                dir=str(path.parent),
            )
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)

            output_format = src.data_model if src.data_model in {"NETCDF4", "NETCDF4_CLASSIC"} else "NETCDF4"
            with Dataset(tmp_path, "w", format=output_format) as dst:
                copy_attrs(src, dst)
                dst.setncattr(DONE_ATTR, "true")
                dst.setncattr(LEVELS_ATTR, level_text)
                dst.setncattr("geos_s2s_tc_vertical_slim_source", str(path))
                dst.setncattr("geos_s2s_tc_vertical_slim_time", datetime.utcnow().isoformat(timespec="seconds") + "Z")

                for dim_name, dim in src.dimensions.items():
                    if dim_name in vertical_dims:
                        dim_len = int(vertical_dims[dim_name]["after"])
                    elif dim.isunlimited():
                        dim_len = None
                    else:
                        dim_len = len(dim)
                    dst.createDimension(dim_name, dim_len)

                for name, var in src.variables.items():
                    out_var = create_variable(dst, var, name, args.complevel)
                    copy_variable_data(var, out_var, vertical_dims)

            original_mode = stat.S_IMODE(path.stat().st_mode)
            os.replace(tmp_path, path)
            os.chmod(path, original_mode)
            tmp_path = None

            result = SlimResult(
                str(path),
                "processed",
                vertical_dims=",".join(vertical_dims),
                levels_before=levels_before,
                levels_after=levels_after,
                variables_trimmed=variables_trimmed,
                message=f"selected_hpa={selected}",
            )
            write_marker(path, result, target_hpa)
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
        "time\tstatus\tfile\tvertical_dims\tlevels_before\tlevels_after\tvariables_trimmed\tmessage\n"
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
                    result.vertical_dims,
                    result.levels_before,
                    result.levels_after,
                    str(result.variables_trimmed),
                    clean_message,
                ]
            )
            + "\n"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atm-root", default=os.environ.get("ATM_ROOT", DEFAULT_ATM_ROOT))
    parser.add_argument("--collection", default=os.environ.get("ATM_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--init-dates-file", default=os.environ.get("INIT_DATES_FILE"))
    parser.add_argument("--forecast-months", default=os.environ.get("FORECAST_MONTHS", ""))
    parser.add_argument("--levels-hpa", default=os.environ.get("LEVELS_HPA", DEFAULT_LEVELS_HPA))
    parser.add_argument("--tolerance-hpa", type=float, default=float(os.environ.get("LEVEL_TOLERANCE_HPA", "0.5")))
    parser.add_argument("--complevel", type=int, default=int(os.environ.get("COMPLEVEL", "4")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")))
    parser.add_argument("--manifest", default=os.environ.get("MANIFEST_FILE"))
    parser.add_argument("--max-files", type=int, default=int(os.environ.get("MAX_FILES", "0")))
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "0") == "1")
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
        print("Load a Python environment with netCDF4 and numpy, or pass PYTHON_BIN to the PBS script.", file=sys.stderr)
        return 2

    atm_root = Path(args.atm_root)
    forecast_months = set(parse_list(args.forecast_months)) or None
    init_dates = read_init_dates(args.init_dates_file)
    manifest = Path(args.manifest) if args.manifest else atm_root / "job_state" / "atm_vertical_slim_manifest.tsv"

    files = discover_files(atm_root, args.collection, init_dates, forecast_months)
    if args.max_files > 0:
        files = files[: args.max_files]

    write_manifest_header(manifest)

    print(f"ATM_ROOT={atm_root}")
    print(f"COLLECTION={args.collection}")
    print(f"LEVELS_HPA={','.join(f'{level:g}' for level in args.levels_hpa)}")
    print(f"FORECAST_MONTHS={','.join(sorted(forecast_months)) if forecast_months else 'all'}")
    print(f"INIT_DATES_FILE={args.init_dates_file or 'all'}")
    print(f"COMPLEVEL={args.complevel}")
    print(f"WORKERS={args.workers}")
    print(f"DRY_RUN={int(args.dry_run)}")
    print(f"MANIFEST={manifest}")
    print(f"FILES_FOUND={len(files)}")

    if not files:
        return 0

    counts: dict[str, int] = {}
    if args.workers <= 1:
        for path in files:
            result = slim_file(path, args)
            append_manifest(manifest, result)
            counts[result.status] = counts.get(result.status, 0) + 1
            print(
                f"{result.status} {result.path} "
                f"levels_before={result.levels_before or '-'} "
                f"levels_after={result.levels_after or '-'} "
                f"vars_trimmed={result.variables_trimmed} {result.message}"
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
                    f"levels_before={result.levels_before or '-'} "
                    f"levels_after={result.levels_after or '-'} "
                    f"vars_trimmed={result.variables_trimmed} {result.message}"
                )

    print("Summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}={count}")

    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
