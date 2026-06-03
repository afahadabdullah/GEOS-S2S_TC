# GEOS-S2S_TC

Utilities for staging GEOS S2S3 forecast data for tropical cyclone evaluation.

Current focus:
- keep SFC and ATM GEOS S2S3 forecast collections in separate local roots
- untar locally available SFC monthly tar files for all available ensembles
- queue missing ATM monthly tar transfers with `shiftc` from a NAS PFE shell
- slim extracted ATM `.nc4` files to selected pressure levels to save space
- slim extracted SFC `.nc4` files to TC-relevant variables to save space
- audit ATM coverage against the local SFC tree before resuming transfers

This repo is being used to support later tropical cyclone diagnostics such as GPI and ACE for GEOS S2Sv2 vs S2Sv3 comparisons.

There is also an experimental ATM-informed ACE proxy at
[scripts/calculate_tc_conditioned_ace.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/calculate_tc_conditioned_ace.py).
It uses SFC winds for the ACE intensity term, but gates accumulation with ATM
structure from SLP, T, QV, and optional 850-hPa vorticity instead of relying on
a single model-specific wind threshold.

## Repository Layout

```text
GEOS-S2S_TC/
|-- config/
|   |-- init_dates_late_aug_1991_2024.txt
|   `-- init_dates_late_aug_1999_2024.txt
|-- scripts/
|   |-- copy_geos_s2s3_ens1_sepnov.pbs
|   |-- copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs
|   |-- audit_atm_vs_sfc_progress.sh
|   |-- summarize_collection_status.py
|   |-- submit_shiftc_atm_from_sfc_missing.pbs
|   |-- submit_shiftc_pull_son_allens.pbs
|   |-- untar_and_slim_atm.py
|   |-- submit_untar_slim_atm.pbs
|   |-- slim_atm_vertical_levels.py
|   |-- submit_slim_atm_vertical_levels.pbs
|   |-- slim_sfc_variables.py
|   |-- submit_slim_sfc_variables.pbs
|   |-- untar_local_sfc_late_aug_allens.pbs
|   |-- cleanup_split_sfc_atm_roots.sh
|   |-- move_geoss2s3_atm_to_project_data.sh
|   `-- move_project_data_to_geoss2s3_atm.sh
`-- README.md
```

## Data Assumptions

- Remote host: `lou`
- Remote root: `/lou/la5/knakada/GEOSS2S3/GEOS_fcst`
- Local SFC root: `/nobackupp27/afahad/project/GEOS-S2S_TC/data`
- Local SFC GEOS tree: `/nobackupp27/afahad/project/GEOS-S2S_TC/data/GEOS_fcst`
- Local ATM root: `/nobackupp17/afahad/GEOSS2S3_atm`
- Local ATM GEOS tree: `/nobackupp17/afahad/GEOSS2S3_atm/GEOS_fcst`
- Ensembles: all locally available `ens*` directories for the local SFC and shiftc-resume workflows
- Forecast months: September, October, and November unless a script explicitly overrides this
- Init dates: late August (`0824` and `0829`) from the supplied manifest files

SFC files are stored under:

```text
/nobackupp27/afahad/project/GEOS-S2S_TC/data/GEOS_fcst/<init_date>/<ens>/sfc_tavg_3hr_glo_L720x361_sfc/
```

ATM files are stored under:

```text
/nobackupp17/afahad/GEOSS2S3_atm/GEOS_fcst/<init_date>/<ens>/atm_inst_6hr_glo_L720x361_p49/
```

Use `scripts/cleanup_split_sfc_atm_roots.sh` if files were accidentally moved across roots.

## Main PBS Jobs

### 1. Local SFC untar for all available ensembles

Script:
- [scripts/untar_local_sfc_late_aug_allens.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/untar_local_sfc_late_aug_allens.pbs)

Defaults:
- source root: `/nobackupp28/knakada/GEOSS2S3/GEOS_fcst`
- target root: `/nobackupp27/afahad/project/GEOS-S2S_TC/data`
- collection: `sfc_tavg_3hr_glo_L720x361_sfc`
- init dates: `1991-2024`, late August
- forecast months: `09 10`
- ensembles: all available local `ens*` directories

### 2. ATM shiftc resume from local SFC coverage

Script:
- [scripts/submit_shiftc_atm_from_sfc_missing.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/submit_shiftc_atm_from_sfc_missing.pbs)

Defaults:
- SFC root: `/nobackupp27/afahad/project/GEOS-S2S_TC/data`
- ATM root: `/nobackupp17/afahad/GEOSS2S3_atm`
- collection: `atm_inst_6hr_glo_L720x361_p49`
- init dates: `1991-2024`, late August
- forecast months: `09 10 11`
- ensembles: all `ens*` directories already represented in the local SFC tree

### 3. Untar and slim ATM files in one compute job

Scripts:
- [scripts/untar_and_slim_atm.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/untar_and_slim_atm.py)
- [scripts/submit_untar_slim_atm.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/submit_untar_slim_atm.pbs)

Defaults:
- ATM root: `/nobackupp17/afahad/GEOSS2S3_atm`
- collection: `atm_inst_6hr_glo_L720x361_p49`
- forecast months: `09 10`
- kept pressure levels: `1000,950,850,500,200` hPa
- compression: NetCDF4 zlib compression level `4`
- conda environment: `earth`
- queue/walltime: `normal` queue with `8:00:00` walltime
- PBS resources: `select=1:ncpus=40:mpiprocs=40:model=sky_ele`
- continuation: stops after `27000` seconds, about 7.5 hours, and resubmits itself if tar files remain
- tar manifest: `/nobackupp17/afahad/GEOSS2S3_atm/job_state/atm_untar_slim_tar_manifest.tsv`
- file manifest: `/nobackupp17/afahad/GEOSS2S3_atm/job_state/atm_untar_slim_file_manifest.tsv`

The combined workflow extracts each monthly `.nc4.tar`, slims every extracted `.nc4` file to the requested pressure levels, and records both the tar-level progress and per-file slimming result. Existing `.untar_done` and `.vertical_slim_done` markers let reruns resume without repeating completed work.

### 4. Slim extracted ATM files to selected pressure levels

Scripts:
- [scripts/slim_atm_vertical_levels.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/slim_atm_vertical_levels.py)
- [scripts/submit_slim_atm_vertical_levels.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/submit_slim_atm_vertical_levels.pbs)

Defaults:
- ATM root: `/nobackupp17/afahad/GEOSS2S3_atm`
- collection: `atm_inst_6hr_glo_L720x361_p49`
- kept pressure levels: `1000,950,850,500,200` hPa
- compression: NetCDF4 zlib compression level `4`
- conda environment: `earth`
- queue/walltime: `normal` queue with `8:00:00` walltime
- PBS resources: `select=1:ncpus=40:mpiprocs=40:model=sky_ele`
- continuation: stops after `27000` seconds, about 7.5 hours, and resubmits itself if files remain
- tracking manifest: `/nobackupp17/afahad/GEOSS2S3_atm/job_state/atm_vertical_slim_manifest.tsv`

The PBS wrapper activates the `earth` conda environment before running Python. The script rewrites each extracted `.nc4` file in place through a temporary file in the same directory. Variables with the pressure-level dimension are reduced to the selected levels; variables without that dimension are copied unchanged. Completed files get a `.vertical_slim_done` marker, so new runs skip files that were already processed. Near the 7.5-hour mark, the processor stops before starting another file and the PBS wrapper submits a continuation job.

### 5. Slim extracted SFC files to TC variables

Scripts:
- [scripts/slim_sfc_variables.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/slim_sfc_variables.py)
- [scripts/submit_slim_sfc_variables.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/submit_slim_sfc_variables.pbs)

Defaults:
- SFC root: `/nobackupp27/afahad/project/GEOS-S2S_TC/data`
- collection: `sfc_tavg_3hr_glo_L720x361_sfc`
- kept data variables: `QS,T2M,TS,US,VS`; required coordinates such as `lat,lon,time` are kept automatically
- compression: NetCDF4 zlib compression level `4`
- conda environment: `earth`
- queue/walltime: `normal` queue with `8:00:00` walltime
- PBS resources: `select=1:ncpus=40:mpiprocs=40:model=sky_ele`
- continuation: stops after `27000` seconds, about 7.5 hours, and resubmits itself if files remain
- tracking manifest: `/nobackupp27/afahad/project/GEOS-S2S_TC/data/job_state/sfc_variable_slim_manifest.tsv`

The SFC slimmer removes every non-coordinate variable outside the keep list. It keeps coordinate/grid variables required by the retained variables, rewrites each `.nc4` in place through a temporary file, and writes `.sfc_var_slim_done` markers so reruns skip files already processed.

### 6. Legacy SFC scp workflow for ens1

Script:
- [scripts/copy_geos_s2s3_ens1_sepnov.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov.pbs)

Defaults:
- collection: `sfc_tavg_3hr_glo_L720x361_sfc`
- init dates: `1999-2024`, late August
- forecast months: `09 10 11`

### 7. Legacy ATM scp workflow for ens1

Script:
- [scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs)

Defaults:
- collection: `atm_inst_6hr_glo_L720x361_p49`
- init dates: `1991-2024`, late August
- forecast months: `09 10 11`

### 8. Experimental TC-conditioned ACE from SFC + ATM

Script:
- [scripts/calculate_tc_conditioned_ace.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/calculate_tc_conditioned_ace.py)
- [scripts/calculate_ibtracs_observed_percentiles.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/calculate_ibtracs_observed_percentiles.py)
- [scripts/calculate_geos_candidate_thresholds.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/calculate_geos_candidate_thresholds.py)
- [scripts/submit_geos_candidate_thresholds.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/submit_geos_candidate_thresholds.pbs)

Observed IBTrACS percentile calibration:

```tcsh
python scripts/calculate_ibtracs_observed_percentiles.py \
  --ibtracs data/obs/ibtracs/IBTrACS.since1980.v04r01.nc \
  --start-year 1991 \
  --end-year 2024 \
  --months 9,10 \
  --threshold-kt 34 \
  --wind-vars wmo_wind,usa_wind
```

This writes `data/obs/ibtracs/ibtracs_observed_percentiles.csv` by default.
The default basin assignment uses the same latitude/longitude basin boxes as
the GEOS TC-conditioned ACE script. Use `--basin-method ibtracs_code` to compare
against the IBTrACS basin-code grouping.

GEOS candidate wind threshold calibration for the currently available
`20200824/ens1` ATM subset:

```tcsh
python scripts/calculate_geos_candidate_thresholds.py \
  --sfc-root /nobackupp27/afahad/project/GEOS-S2S_TC/data \
  --atm-root /nobackupp17/afahad/GEOSS2S3_atm \
  --init-date 20200824 \
  --ens ens1 \
  --months 09,10 \
  --observed-percentiles data/obs/ibtracs/ibtracs_observed_percentiles.csv \
  --observed-wind-vars usa_wind
```

This writes two CSV files under `data/calibration/` by default:
- `geos_candidate_thresholds_<init>_<ens>.csv`
- `geos_candidate_thresholds_<init>_<ens>_candidates.csv`

The `_candidates.csv` file is the reusable TC-candidate inventory. It records
the source init date, ensemble, ATM file/time index, matched SFC file/time
index, valid time, basin, center location, Vmax, and structure-gate diagnostics
for every accepted GEOS candidate.

Submit the same calculation to PBS:

```tcsh
qsub -v INIT_DATE=20200824,ENS=ens1,FORECAST_MONTHS=09,10 scripts/submit_geos_candidate_thresholds.pbs
```

To calibrate over a date list later, pass `INIT_DATES_FILE` and leave
`INIT_DATE` empty:

```tcsh
qsub -v INIT_DATE=,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,ENS=ens1,FORECAST_MONTHS=09,10 scripts/submit_geos_candidate_thresholds.pbs
```

To write candidates for all available ensemble members, use `ENS=all`:

```tcsh
qsub -v INIT_DATE=,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,ENS=all,FORECAST_MONTHS=09,10 scripts/submit_geos_candidate_thresholds.pbs
```

Example:

```bash
python scripts/calculate_tc_conditioned_ace.py \
  --sfc-root /nobackupp27/afahad/project/GEOS-S2S_TC/data \
  --atm-root /nobackupp17/afahad/GEOSS2S3_atm \
  --init-date 20200824 \
  --ens ens1 \
  --months 09,10
```

Notes:
- it processes only ATM files that are actually present under the requested init/ens
- it matches each ATM valid time to the nearest SFC valid time within tolerance
- it accumulates ACE only when the basin candidate has a local SLP minimum, a
  positive warm-core anomaly, a positive low-level moisture anomaly, and, when
  U/V are available, hemisphere-consistent 850-hPa vorticity
- it writes a cache file under `data/cache/` by default and saves plots under
  `plots/` by default
- plots can be regenerated from an existing cache without re-reading SFC or ATM
  files:

```bash
python scripts/calculate_tc_conditioned_ace.py \
  --init-date 20200824 \
  --plot-only-cache data/cache/tc_conditioned_ace_20200824_ens1.nc4
```

## How the Jobs Work

The local SFC untar workflow is restart-safe:

- it discovers all available local `ens*` directories for each init date
- it untars one monthly tar file at a time
- it writes `.untar_done` markers next to completed tar files
- it tracks progress in a state directory under `/nobackupp27/afahad/project/GEOS-S2S_TC/data/job_state/`
- it can resubmit itself before walltime is exhausted

The ATM shiftc workflow is also restart-safe:

- it uses the local SFC tree as the expected init/ensemble/month universe
- it skips ATM files that are already extracted, already present as tar files, or already submitted to shiftc
- it writes queue, skip, error, and raw output logs under the ATM root
- it records `.shiftc_submitted` markers after successful shiftc submissions
- it should be run directly on PFE/login nodes, not submitted with `qsub`

State files include:
- `progress.tsv`
- `next_index.txt`
- `completed.ok`

## Coverage Summaries

Script:
- [scripts/summarize_collection_status.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/summarize_collection_status.py)

This script checks each expected init date, ensemble, and forecast month and classifies files as:
- `untarred`: `.untar_done`, `.untar_slim_done`, or extracted `.nc4` files are present
- `downloaded_not_untarred`: the monthly `.nc4.tar` file is present, but extracted files/markers are not
- `submitted_not_downloaded`: only a `.shiftc_submitted` marker is present
- `missing`: no tar, extracted files, or submitted marker are present

It writes summary, init-level, detail-level, missing, and downloaded-not-untarred TSV files under the data root `reports/` directory.

## Submission Examples

Update the repo on NAS:

```bash
cd /nobackupp27/afahad/project/GEOS-S2S_TC
git pull
```

Untar locally available SFC files:

```bash
qsub -v INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,RESUBMIT_SCRIPT=/nobackupp27/afahad/project/GEOS-S2S_TC/scripts/untar_local_sfc_late_aug_allens.pbs /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/untar_local_sfc_late_aug_allens.pbs
```

Clean up crossed SFC/ATM collection files. Start with the dry run:

```bash
env DRY_RUN=1 bash scripts/cleanup_split_sfc_atm_roots.sh
```

If the dry run looks correct, run the move:

```bash
env DRY_RUN=0 bash scripts/cleanup_split_sfc_atm_roots.sh
```

Audit ATM coverage against the local SFC tree:

```bash
bash scripts/audit_atm_vs_sfc_progress.sh
```

Summarize ATM download/untar coverage against the SFC reference tree:

```bash
env DATASET=atm FORECAST_MONTHS=09:10 python /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/summarize_collection_status.py
```

Summarize SFC download/untar coverage:

```bash
env DATASET=sfc FORECAST_MONTHS=09:10 python /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/summarize_collection_status.py
```

Resume missing ATM transfers with shiftc from PFE:

```bash
env SFC_ROOT=/nobackupp27/afahad/project/GEOS-S2S_TC/data ATM_ROOT=/nobackupp17/afahad/GEOSS2S3_atm INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt bash scripts/submit_shiftc_atm_from_sfc_missing.pbs
```

Untar downloaded ATM tar files and slim the extracted files in the same compute job:

```bash
qsub -v ATM_ROOT=/nobackupp17/afahad/GEOSS2S3_atm,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,FORECAST_MONTHS=09:10,CONDA_ENV=earth /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_untar_slim_atm.pbs
```

For a small tar-level test first:

```bash
qsub -v ATM_ROOT=/nobackupp17/afahad/GEOSS2S3_atm,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,FORECAST_MONTHS=09:10,MAX_TARS=1,CONDA_ENV=earth /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_untar_slim_atm.pbs
```

To remove monthly tar files after successful untar and slimming, add `DELETE_TAR_AFTER_SUCCESS=1` to the `qsub -v` list.

Slim already-extracted ATM `.nc4` files on compute nodes:

```bash
qsub -v ATM_ROOT=/nobackupp17/afahad/GEOSS2S3_atm,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,FORECAST_MONTHS=09:10,CONDA_ENV=earth /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_slim_atm_vertical_levels.pbs
```

For a small test first:

```bash
qsub -v ATM_ROOT=/nobackupp17/afahad/GEOSS2S3_atm,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,FORECAST_MONTHS=09:10,MAX_FILES=2,CONDA_ENV=earth /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_slim_atm_vertical_levels.pbs
```

If the compute node cannot auto-detect conda, add `CONDA_BASE=/full/path/to/miniconda3` to the `qsub -v` list.

To change the continuation threshold, add `ELAPSED_LIMIT_SECONDS=<seconds>` to the `qsub -v` list.

Slim already-extracted SFC `.nc4` files on compute nodes:

```bash
qsub -v SFC_ROOT=/nobackupp27/afahad/project/GEOS-S2S_TC/data,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,FORECAST_MONTHS=09:10,CONDA_ENV=earth /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_slim_sfc_variables.pbs
```

For a small SFC test first:

```bash
qsub -v SFC_ROOT=/nobackupp27/afahad/project/GEOS-S2S_TC/data,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,FORECAST_MONTHS=09:10,MAX_FILES=2,CONDA_ENV=earth /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_slim_sfc_variables.pbs
```

Legacy `ens1` scp submissions:

```bash
qsub -v INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1999_2024.txt,RESUBMIT_SCRIPT=/nobackupp27/afahad/project/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov.pbs /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov.pbs
```

```bash
qsub -v INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,RESUBMIT_SCRIPT=/nobackupp27/afahad/project/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs
```

## Notes

- The legacy `scp` scripts still support `ens1` workflows, but current SFC local untar and ATM shiftc-resume workflows use all available local `ens*` directories.
- The shiftc workflow does not untar ATM tar files; it only queues transfers and records shiftc submission IDs when available.
- The ATM vertical slimming workflow only processes extracted `.nc4` files. Monthly `.tar` files must be extracted before they can be slimmed.
- The SFC variable slimming workflow only processes extracted `.nc4` files. Monthly `.tar` files must be extracted before they can be slimmed.
- The cleanup script uses `rsync --ignore-existing --remove-source-files`, so files already present in the correct root are not overwritten.
- After checking a successful cleanup, empty directories can be removed manually from either GEOS tree with `find <GEOS_fcst_root> -type d -empty -delete`.
