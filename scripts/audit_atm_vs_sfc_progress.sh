#!/bin/bash

# Compare local ATM progress against local SFC coverage.
#
# The SFC tree is treated as the expected universe. For each init/ens/month
# where SFC exists, this reports whether the matching ATM tar/extracted files
# are already present, queued, or still missing.

set -u
umask 022

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"

DEFAULT_SFC_ROOT="/nobackupp27/afahad/project/GEOS-S2S_TC/data"
DEFAULT_ATM_ROOT="/nobackupp17/afahad/GEOSS2S3_atm"

if [[ -n "${TARGET_ROOT:-}" ]]; then
  SFC_ROOT="${SFC_ROOT:-${TARGET_ROOT}}"
  ATM_ROOT="${ATM_ROOT:-${TARGET_ROOT}}"
else
  SFC_ROOT="${SFC_ROOT:-${DEFAULT_SFC_ROOT}}"
  ATM_ROOT="${ATM_ROOT:-${DEFAULT_ATM_ROOT}}"
fi

INIT_DATES_FILE="${INIT_DATES_FILE:-${REPO_ROOT}/config/init_dates_late_aug_1991_2024.txt}"
FORECAST_MONTHS="${FORECAST_MONTHS:-09 10 11}"
FILE_INTERVAL_TAG="${FILE_INTERVAL_TAG:-daily}"

SFC_COLLECTION="${SFC_COLLECTION:-sfc_tavg_3hr_glo_L720x361_sfc}"
ATM_COLLECTION="${ATM_COLLECTION:-atm_inst_6hr_glo_L720x361_p49}"
REMOTE_HOST="${REMOTE_HOST:-lfe6}"
REMOTE_ROOT="${REMOTE_ROOT:-/lou/la5/knakada/GEOSS2S3/GEOS_fcst}"

REPORT_DIR="${REPORT_DIR:-${ATM_ROOT}/reports}"
REPORT_PREFIX="${REPORT_PREFIX:-atm_vs_sfc}"
SUMMARY_FILE="${SUMMARY_FILE:-${REPORT_DIR}/${REPORT_PREFIX}_summary.txt}"
EXPECTED_FILE="${EXPECTED_FILE:-${REPORT_DIR}/${REPORT_PREFIX}_expected.tsv}"
MISSING_FILE="${MISSING_FILE:-${REPORT_DIR}/${REPORT_PREFIX}_missing_atm.tsv}"

mkdir -p "${REPORT_DIR}"

if [[ ! -f "${INIT_DATES_FILE}" ]]; then
  echo "ERROR init-date file not found: ${INIT_DATES_FILE}" >&2
  exit 2
fi

mapfile -t init_dates < <(tr -s '[:space:]' '\n' < "${INIT_DATES_FILE}" | grep -E '^[0-9]{8}$')
read -r -a forecast_month_array <<< "${FORECAST_MONTHS}"

if [[ "${#init_dates[@]}" -eq 0 ]]; then
  echo "ERROR no valid init dates found in ${INIT_DATES_FILE}" >&2
  exit 3
fi

if [[ "${#forecast_month_array[@]}" -eq 0 ]]; then
  echo "ERROR no forecast months configured" >&2
  exit 4
fi

has_month_nc4() {
  local dir="$1"
  local init_date="$2"
  local collection="$3"
  local yyyymm="$4"
  local files=()

  shopt -s nullglob
  files=(
    "${dir}/${init_date}.${collection}.${FILE_INTERVAL_TAG}.${yyyymm}"*.nc4
    "${dir}/${init_date}.${collection}.${yyyymm}"*.nc4
  )
  shopt -u nullglob

  [[ "${#files[@]}" -gt 0 ]]
}

month_status() {
  local dir="$1"
  local init_date="$2"
  local collection="$3"
  local yyyymm="$4"
  local tar_name="${init_date}.${collection}.${FILE_INTERVAL_TAG}.${yyyymm}.nc4.tar"

  if [[ -f "${dir}/${tar_name}.untar_done" ]]; then
    echo "extracted_marker"
  elif has_month_nc4 "${dir}" "${init_date}" "${collection}" "${yyyymm}"; then
    echo "extracted_nc4"
  elif [[ -f "${dir}/${tar_name}" ]]; then
    echo "tar_present"
  elif [[ -f "${dir}/${tar_name}.shiftc_submitted" ]]; then
    echo "shiftc_submitted"
  else
    echo "missing"
  fi
}

is_done_or_pending() {
  case "$1" in
    extracted_marker|extracted_nc4|tar_present|shiftc_submitted) return 0 ;;
    *) return 1 ;;
  esac
}

expected_count=0
atm_done_count=0
atm_missing_count=0
init_missing_count=0
ens_missing_count=0
sfc_month_missing_count=0

printf "init_date\tens\tforecast_month\tsfc_status\tatm_status\tatm_local_dir\tatm_tar\tremote_file\n" > "${EXPECTED_FILE}"
printf "init_date\tens\tforecast_month\tatm_local_dir\tatm_tar\tremote_file\n" > "${MISSING_FILE}"

for init_date in "${init_dates[@]}"; do
  init_year="${init_date:0:4}"
  init_dir="${SFC_ROOT}/GEOS_fcst/${init_date}"

  if [[ ! -d "${init_dir}" ]]; then
    init_missing_count=$((init_missing_count + 1))
    continue
  fi

  mapfile -t ens_dirs < <(find "${init_dir}" -mindepth 1 -maxdepth 1 -type d -name 'ens*' -exec basename {} \; | sort)

  if [[ "${#ens_dirs[@]}" -eq 0 ]]; then
    ens_missing_count=$((ens_missing_count + 1))
    continue
  fi

  for ens in "${ens_dirs[@]}"; do
    sfc_dir="${SFC_ROOT}/GEOS_fcst/${init_date}/${ens}/${SFC_COLLECTION}"
    atm_dir="${ATM_ROOT}/GEOS_fcst/${init_date}/${ens}/${ATM_COLLECTION}"

    for fcst_month in "${forecast_month_array[@]}"; do
      yyyymm="${init_year}${fcst_month}"
      atm_tar="${init_date}.${ATM_COLLECTION}.${FILE_INTERVAL_TAG}.${yyyymm}.nc4.tar"
      remote_file="${REMOTE_ROOT}/${init_date}/${ens}/${ATM_COLLECTION}/${atm_tar}"

      sfc_status="$(month_status "${sfc_dir}" "${init_date}" "${SFC_COLLECTION}" "${yyyymm}")"

      if ! is_done_or_pending "${sfc_status}"; then
        sfc_month_missing_count=$((sfc_month_missing_count + 1))
        continue
      fi

      atm_status="$(month_status "${atm_dir}" "${init_date}" "${ATM_COLLECTION}" "${yyyymm}")"
      expected_count=$((expected_count + 1))

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s:%s\n" \
        "${init_date}" "${ens}" "${fcst_month}" "${sfc_status}" "${atm_status}" \
        "${atm_dir}" "${atm_tar}" "${REMOTE_HOST}" "${remote_file}" >> "${EXPECTED_FILE}"

      if is_done_or_pending "${atm_status}"; then
        atm_done_count=$((atm_done_count + 1))
      else
        atm_missing_count=$((atm_missing_count + 1))
        printf "%s\t%s\t%s\t%s\t%s\t%s:%s\n" \
          "${init_date}" "${ens}" "${fcst_month}" "${atm_dir}" "${atm_tar}" \
          "${REMOTE_HOST}" "${remote_file}" >> "${MISSING_FILE}"
      fi
    done
  done
done

{
  echo "Generated: $(date)"
  echo "SFC_ROOT=${SFC_ROOT}"
  echo "ATM_ROOT=${ATM_ROOT}"
  echo "INIT_DATES_FILE=${INIT_DATES_FILE}"
  echo "FORECAST_MONTHS=${FORECAST_MONTHS}"
  echo "SFC_COLLECTION=${SFC_COLLECTION}"
  echo "ATM_COLLECTION=${ATM_COLLECTION}"
  echo "expected_from_sfc=${expected_count}"
  echo "atm_done_or_pending=${atm_done_count}"
  echo "atm_missing=${atm_missing_count}"
  echo "init_dirs_missing=${init_missing_count}"
  echo "ens_dirs_missing=${ens_missing_count}"
  echo "sfc_months_missing=${sfc_month_missing_count}"
  echo "expected_file=${EXPECTED_FILE}"
  echo "missing_file=${MISSING_FILE}"
} > "${SUMMARY_FILE}"

cat "${SUMMARY_FILE}"
