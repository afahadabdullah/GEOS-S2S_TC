#!/bin/bash

# Move misplaced GEOS S2S3 collection files back into separate SFC and ATM
# roots while preserving GEOS_fcst/<init>/<ens>/<collection>/ layout.
#
# Safe default:
#   env DRY_RUN=1 bash scripts/cleanup_split_sfc_atm_roots.sh
#
# Real move:
#   env DRY_RUN=0 bash scripts/cleanup_split_sfc_atm_roots.sh

set -u
umask 022

SFC_ROOT="${SFC_ROOT:-/nobackupp27/afahad/project/GEOS-S2S_TC/data}"
ATM_ROOT="${ATM_ROOT:-/nobackupp17/afahad/GEOSS2S3_atm}"
SFC_GEOS_ROOT="${SFC_GEOS_ROOT:-${SFC_ROOT%/}/GEOS_fcst}"
ATM_GEOS_ROOT="${ATM_GEOS_ROOT:-${ATM_ROOT%/}/GEOS_fcst}"

SFC_COLLECTIONS="${SFC_COLLECTIONS:-sfc_tavg_3hr_glo_L720x361_sfc}"
ATM_COLLECTIONS="${ATM_COLLECTIONS:-atm_inst_6hr_glo_L720x361_p49}"
DRY_RUN="${DRY_RUN:-1}"

move_count=0
missing_root_count=0

echo "SFC_GEOS_ROOT=${SFC_GEOS_ROOT}"
echo "ATM_GEOS_ROOT=${ATM_GEOS_ROOT}"
echo "SFC_COLLECTIONS=${SFC_COLLECTIONS}"
echo "ATM_COLLECTIONS=${ATM_COLLECTIONS}"
echo "DRY_RUN=${DRY_RUN}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR rsync is required but was not found."
  exit 2
fi

if [[ "${SFC_GEOS_ROOT}" == "${ATM_GEOS_ROOT}" ]]; then
  echo "ERROR SFC_GEOS_ROOT and ATM_GEOS_ROOT are identical; refusing to move."
  exit 3
fi

move_collection_dirs() {
  local source_root="$1"
  local target_root="$2"
  local collection="$3"
  local source_dir rel_path target_dir

  if [[ ! -d "${source_root}" ]]; then
    echo "WARN source root not found, skipping: ${source_root}"
    missing_root_count=$((missing_root_count + 1))
    return 0
  fi

  mkdir -p "${target_root}"

  while IFS= read -r source_dir; do
    rel_path="${source_dir#${source_root}/}"
    target_dir="${target_root}/${rel_path}"

    if [[ "${source_dir}" == "${target_dir}" ]]; then
      echo "SKIP identical source/target: ${source_dir}"
      continue
    fi

    echo "MOVE_COLLECTION ${collection}"
    echo "  from: ${source_dir}/"
    echo "  to:   ${target_dir}/"

    mkdir -p "${target_dir}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      rsync -avhn --ignore-existing --remove-source-files "${source_dir}/" "${target_dir}/"
    else
      rsync -avh --ignore-existing --remove-source-files "${source_dir}/" "${target_dir}/"
      find "${source_dir}" -type d -empty -delete
    fi

    move_count=$((move_count + 1))
  done < <(find "${source_root}" -type d -name "${collection}" | sort)
}

for collection in ${ATM_COLLECTIONS}; do
  move_collection_dirs "${SFC_GEOS_ROOT}" "${ATM_GEOS_ROOT}" "${collection}"
done

for collection in ${SFC_COLLECTIONS}; do
  move_collection_dirs "${ATM_GEOS_ROOT}" "${SFC_GEOS_ROOT}" "${collection}"
done

echo "Cleanup finished: $(date)"
echo "collection_dirs_considered=${move_count}"
echo "missing_roots=${missing_root_count}"
echo "Files that already existed at the target were left in the source root."
echo "If DRY_RUN=1, no files were moved."
