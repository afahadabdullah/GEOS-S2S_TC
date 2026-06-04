"""Small ocean/land masking helpers for GEOS TC candidate workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


SFC_OCEAN_FRACTION_CANDIDATES = ("FROCEAN", "FRSEAICE")
SFC_LAND_FRACTION_CANDIDATES = ("FRLAND", "FRLANDICE")


class OceanChecker(Protocol):
    source: str

    def is_ocean(self, lat: float, lon: float) -> bool:
        ...


def normalize_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def parse_optional_bool(value: str | None, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def add_ocean_only_args(parser) -> None:
    parser.add_argument(
        "--ocean-only",
        dest="ocean_only",
        action="store_true",
        default=True,
        help="Keep only ocean points/fixes. This is the default.",
    )
    parser.add_argument(
        "--no-ocean-only",
        dest="ocean_only",
        action="store_false",
        help="Disable ocean-only filtering.",
    )
    parser.add_argument(
        "--ocean-mask-source",
        choices=("auto", "sfc", "cartopy", "none"),
        default="auto",
        help="Ocean mask source. auto tries SFC fractions first, then Natural Earth through Cartopy.",
    )
    parser.add_argument(
        "--ocean-mask-file",
        default="",
        help="Optional SFC file containing FRLAND/FROCEAN-style land/ocean fractions.",
    )
    parser.add_argument(
        "--ocean-threshold",
        type=float,
        default=0.5,
        help="Minimum ocean fraction for SFC fraction masks. Default 0.5.",
    )


@dataclass
class NoopOceanChecker:
    source: str = "none"

    def is_ocean(self, lat: float, lon: float) -> bool:
        return True


@dataclass
class GridOceanChecker:
    latitudes: np.ndarray
    longitudes: np.ndarray
    ocean_mask: np.ndarray
    source: str

    def is_ocean(self, lat: float, lon: float) -> bool:
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return False
        lat_idx = int(np.nanargmin(np.abs(self.latitudes - lat)))
        lon_value = lon
        if np.nanmin(self.longitudes) >= 0.0 and lon_value < 0.0:
            lon_value = lon_value + 360.0
        elif np.nanmax(self.longitudes) > 180.0:
            lon_value = lon_value % 360.0
        else:
            lon_value = normalize_lon(lon_value)
        lon_idx = int(np.nanargmin(np.abs(self.longitudes - lon_value)))
        return bool(self.ocean_mask[lat_idx, lon_idx])


class NaturalEarthOceanChecker:
    source = "cartopy-natural-earth-land"

    def __init__(self) -> None:
        try:
            import cartopy.io.shapereader as shpreader
            from shapely.geometry import Point
            from shapely.prepared import prep
        except Exception as exc:  # pragma: no cover - depends on optional packages.
            raise RuntimeError(f"Cartopy/Shapely land mask is unavailable: {exc}") from exc

        land_path = shpreader.natural_earth(resolution="110m", category="physical", name="land")
        reader = shpreader.Reader(land_path)
        self._point_class = Point
        self._land_geometries = [prep(geometry) for geometry in reader.geometries()]

    def is_ocean(self, lat: float, lon: float) -> bool:
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return False
        point = self._point_class(normalize_lon(lon), lat)
        return not any(geometry.contains(point) for geometry in self._land_geometries)


def _first_present_variable(ds, candidates: tuple[str, ...]):
    names_by_upper = {name.upper(): name for name in ds.variables}
    for candidate in candidates:
        name = names_by_upper.get(candidate.upper())
        if name is not None:
            return ds.variables[name]
    return None


def _fraction_array(var) -> np.ndarray:
    array = np.ma.filled(np.asarray(var[:]), np.nan).astype("float64", copy=False)
    while array.ndim > 2:
        array = array[0]
    if np.nanmax(array) > 1.5:
        array = array / 100.0
    return array


def _coordinates_from_dataset(ds) -> tuple[np.ndarray, np.ndarray]:
    lat_name = "lat" if "lat" in ds.variables else "latitude"
    lon_name = "lon" if "lon" in ds.variables else "longitude"
    return (
        np.asarray(ds.variables[lat_name][:], dtype="float64"),
        np.asarray(ds.variables[lon_name][:], dtype="float64"),
    )


def ocean_checker_from_sfc_file(path: Path, threshold: float = 0.5) -> GridOceanChecker | None:
    try:
        import netCDF4
    except ImportError as exc:  # pragma: no cover - dependency checked by callers.
        raise RuntimeError(f"netCDF4 is required for SFC ocean mask: {exc}") from exc

    with netCDF4.Dataset(path, "r") as ds:
        latitudes, longitudes = _coordinates_from_dataset(ds)

        ocean_fraction = None
        for name in SFC_OCEAN_FRACTION_CANDIDATES:
            var = _first_present_variable(ds, (name,))
            if var is None:
                continue
            values = _fraction_array(var)
            ocean_fraction = values if ocean_fraction is None else ocean_fraction + values
        if ocean_fraction is not None:
            return GridOceanChecker(
                latitudes=latitudes,
                longitudes=longitudes,
                ocean_mask=np.asarray(ocean_fraction >= threshold, dtype=bool),
                source=f"sfc:{path.name}:ocean_fraction",
            )

        land_fraction = None
        for name in SFC_LAND_FRACTION_CANDIDATES:
            var = _first_present_variable(ds, (name,))
            if var is None:
                continue
            values = _fraction_array(var)
            land_fraction = values if land_fraction is None else land_fraction + values
        if land_fraction is not None:
            return GridOceanChecker(
                latitudes=latitudes,
                longitudes=longitudes,
                ocean_mask=np.asarray(land_fraction <= (1.0 - threshold), dtype=bool),
                source=f"sfc:{path.name}:land_fraction",
            )

    return None


def build_ocean_checker(
    source: str = "auto",
    sfc_path: Path | None = None,
    mask_file: str | Path | None = None,
    threshold: float = 0.5,
    require_mask: bool = False,
) -> tuple[OceanChecker, str | None]:
    if source == "none":
        if require_mask:
            raise RuntimeError("ocean-only filtering requested, but --ocean-mask-source none disables the mask")
        return NoopOceanChecker(), None

    warnings: list[str] = []
    if mask_file:
        mask_path = Path(mask_file)
        try:
            checker = ocean_checker_from_sfc_file(mask_path, threshold)
            if checker is not None:
                return checker, None
            warnings.append(f"no recognized SFC ocean/land fraction variables in {mask_path}")
        except Exception as exc:
            warnings.append(str(exc))

    if source in {"auto", "sfc"} and sfc_path is not None:
        try:
            checker = ocean_checker_from_sfc_file(sfc_path, threshold)
            if checker is not None:
                return checker, None
            warnings.append(f"no recognized SFC ocean/land fraction variables in {sfc_path}")
        except Exception as exc:
            warnings.append(str(exc))

    if source in {"auto", "cartopy"}:
        try:
            return NaturalEarthOceanChecker(), "; ".join(warnings) if warnings else None
        except Exception as exc:
            warnings.append(str(exc))

    warning_text = "; ".join(warnings) if warnings else "ocean mask disabled"
    if require_mask:
        raise RuntimeError(f"ocean-only filtering requested, but no usable ocean mask was built: {warning_text}")
    return NoopOceanChecker(), warning_text


def row_over_ocean_value(row: dict[str, str]) -> bool | None:
    value = row.get("over_ocean")
    if value is None or value == "":
        return None
    return parse_optional_bool(value, default=True)
