#!/bin/bash

# Legacy helper for the previous direction of migration.
#
# Current workflows now use the project data tree as the unified root. Prefer:
#   scripts/move_geoss2s3_atm_to_project_data.sh

set -u
umask 022

SOURCE_ROOT="${SOURCE_ROOT:-/nobackupp27/afahad/project/GEOS-S2S_TC/data/GEOS_fcst}"
TARGET_ROOT="${TARGET_ROOT:-/nobackupp17/afahad/GEOSS2S3_atm/GEOS_fcst}"
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

echo "Move complete: $(date)"
echo "Empty source directories may remain. Remove them manually after checking:"
echo "  find ${SOURCE_ROOT} -type d -empty -delete"
