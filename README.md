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
|   |-- submit_shiftc_atm_from_sfc_missing.pbs
|   |-- submit_shiftc_pull_son_allens.pbs
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
- Local ATM root: `/nobackupp27/afahad/GEOSS2S3_atm`
- Local ATM GEOS tree: `/nobackupp27/afahad/GEOSS2S3_atm/GEOS_fcst`
- Ensembles: all locally available `ens*` directories for the local SFC and shiftc-resume workflows
- Forecast months: September, October, and November unless a script explicitly overrides this
- Init dates: late August (`0824` and `0829`) from the supplied manifest files

SFC files are stored under:

```text
/nobackupp27/afahad/project/GEOS-S2S_TC/data/GEOS_fcst/<init_date>/<ens>/sfc_tavg_3hr_glo_L720x361_sfc/
```

ATM files are stored under:

```text
/nobackupp27/afahad/GEOSS2S3_atm/GEOS_fcst/<init_date>/<ens>/atm_inst_6hr_glo_L720x361_p49/
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
- ATM root: `/nobackupp27/afahad/GEOSS2S3_atm`
- collection: `atm_inst_6hr_glo_L720x361_p49`
- init dates: `1991-2024`, late August
- forecast months: `09 10 11`
- ensembles: all `ens*` directories already represented in the local SFC tree

### 3. Slim extracted ATM files to selected pressure levels

Scripts:
- [scripts/slim_atm_vertical_levels.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/slim_atm_vertical_levels.py)
- [scripts/submit_slim_atm_vertical_levels.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/submit_slim_atm_vertical_levels.pbs)

Defaults:
- ATM root: `/nobackupp27/afahad/GEOSS2S3_atm`
- collection: `atm_inst_6hr_glo_L720x361_p49`
- kept pressure levels: `1000,950,850,500,200` hPa
- compression: NetCDF4 zlib compression level `4`
- conda environment: `earth`
- queue/walltime: `normal` queue with `8:00:00` walltime
- PBS resources: `select=1:ncpus=40:mpiprocs=40:model=sky_ele`
- continuation: stops after `27000` seconds, about 7.5 hours, and resubmits itself if files remain
- tracking manifest: `/nobackupp27/afahad/GEOSS2S3_atm/job_state/atm_vertical_slim_manifest.tsv`

The PBS wrapper activates the `earth` conda environment before running Python. The script rewrites each extracted `.nc4` file in place through a temporary file in the same directory. Variables with the pressure-level dimension are reduced to the selected levels; variables without that dimension are copied unchanged. Completed files get a `.vertical_slim_done` marker, so new runs skip files that were already processed. Near the 7.5-hour mark, the processor stops before starting another file and the PBS wrapper submits a continuation job.

### 4. Legacy SFC scp workflow for ens1

### 4. Slim extracted SFC files to TC variables

Scripts:
- [scripts/slim_sfc_variables.py](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/slim_sfc_variables.py)
- [scripts/submit_slim_sfc_variables.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/submit_slim_sfc_variables.pbs)

Defaults:
- SFC root: `/nobackupp27/afahad/project/GEOS-S2S_TC/data`
- collection: `sfc_tavg_3hr_glo_L720x361_sfc`
- kept variables and aliases: `U10M,V10M,US,VS,TS,SST,T2M,QV2M,PRECTOT,PRECTOTCORR,PRECCON,PRECLSC,PRECSNO,PRECTOTLAND,LHF,LHFLX,EFLUX`
- compression: NetCDF4 zlib compression level `4`
- conda environment: `earth`
- queue/walltime: `normal` queue with `8:00:00` walltime
- PBS resources: `select=1:ncpus=40:mpiprocs=40:model=sky_ele`
- continuation: stops after `27000` seconds, about 7.5 hours, and resubmits itself if files remain
- tracking manifest: `/nobackupp27/afahad/project/GEOS-S2S_TC/data/job_state/sfc_variable_slim_manifest.tsv`

The SFC slimmer removes every non-coordinate variable outside the keep list. It keeps coordinate/grid variables required by the retained variables, rewrites each `.nc4` in place through a temporary file, and writes `.sfc_var_slim_done` markers so reruns skip files already processed.

### 5. Legacy SFC scp workflow for ens1

Script:
- [scripts/copy_geos_s2s3_ens1_sepnov.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov.pbs)

Defaults:
- collection: `sfc_tavg_3hr_glo_L720x361_sfc`
- init dates: `1999-2024`, late August
- forecast months: `09 10 11`

### 6. Legacy ATM scp workflow for ens1

Script:
- [scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs)

Defaults:
- collection: `atm_inst_6hr_glo_L720x361_p49`
- init dates: `1991-2024`, late August
- forecast months: `09 10 11`

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

Resume missing ATM transfers with shiftc from PFE:

```bash
env SFC_ROOT=/nobackupp27/afahad/project/GEOS-S2S_TC/data ATM_ROOT=/nobackupp27/afahad/GEOSS2S3_atm INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt bash scripts/submit_shiftc_atm_from_sfc_missing.pbs
```

Slim already-extracted ATM `.nc4` files on compute nodes:

```bash
qsub -v ATM_ROOT=/nobackupp27/afahad/GEOSS2S3_atm,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,FORECAST_MONTHS=09:10,CONDA_ENV=earth /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_slim_atm_vertical_levels.pbs
```

For a small test first:

```bash
qsub -v ATM_ROOT=/nobackupp27/afahad/GEOSS2S3_atm,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,FORECAST_MONTHS=09:10,MAX_FILES=2,CONDA_ENV=earth /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_slim_atm_vertical_levels.pbs
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
