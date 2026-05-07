#!/bin/bash

# Move GEOS_fcst files from the old ATM staging root into the project data
# tree, preserving GEOS_fcst/<init>/<ens>/<collection>/ layout.

set -u
umask 022

SOURCE_ROOT="${SOURCE_ROOT:-/nobackupp17/afahad/GEOSS2S3_atm/GEOS_fcst}"
TARGET_ROOT="${TARGET_ROOT:-/nobackupp27/afahad/project/GEOS-S2S_TC/data/GEOS_fcst}"
DRY_RUN="${DRY_RUN:-1}"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "ERROR source root does not exist: ${SOURCE_ROOT}"
  exit 2
fi

mkdir -p "${TARGET_ROOT}"

echo "SOURCE_ROOT=${SOURCE_ROOT}"
echo "TARGET_ROOT=${TARGET_ROOT}"
echo "DRY_RUN=${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  rsync -avhn --ignore-existing --remove-source-files "${SOURCE_ROOT}/" "${TARGET_ROOT}/"
else
  rsync -avh --ignore-existing --remove-source-files "${SOURCE_ROOT}/" "${TARGET_ROOT}/"
fi

echo "Move step complete: $(date)"
echo "Files already present in TARGET_ROOT are left in SOURCE_ROOT."
echo "After checking the result, empty source directories can be removed with:"
echo "  find ${SOURCE_ROOT} -type d -empty -delete"
