# GEOS-S2S_TC

Utilities for staging GEOS S2S3 forecast data for tropical cyclone evaluation.

Current focus:
- preserve GEOS S2S3 forecast files under one local project data tree
- untar locally available SFC monthly tar files for all available ensembles
- queue missing ATM monthly tar transfers with `sup shiftc`
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
|   |-- untar_local_sfc_late_aug_allens.pbs
|   |-- move_geoss2s3_atm_to_project_data.sh
|   `-- move_project_data_to_geoss2s3_atm.sh
`-- README.md
```

## Data Assumptions

- Remote host: `lou`
- Remote root: `/lou/la5/knakada/GEOSS2S3/GEOS_fcst`
- Local project root: `/nobackupp27/afahad/project/GEOS-S2S_TC/data`
- Local GEOS tree: `/nobackupp27/afahad/project/GEOS-S2S_TC/data/GEOS_fcst`
- Ensembles: all locally available `ens*` directories for the local SFC and shiftc-resume workflows
- Forecast months: September, October, and November unless a script explicitly overrides this
- Init dates: late August (`0824` and `0829`) from the supplied manifest files

Files are stored under:

```text
/nobackupp27/afahad/project/GEOS-S2S_TC/data/GEOS_fcst/<init_date>/<ens>/<collection>/
```

The unified processing root is:

```text
/nobackupp27/afahad/project/GEOS-S2S_TC/data
```

The older ATM staging root was `/nobackupp27/afahad/GEOSS2S3_atm/GEOS_fcst`. Use `scripts/move_geoss2s3_atm_to_project_data.sh` to move any files from that older root back into the project data tree.

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
- target root: `/nobackupp27/afahad/project/GEOS-S2S_TC/data`
- collection: `atm_inst_6hr_glo_L720x361_p49`
- init dates: `1991-2024`, late August
- forecast months: `09 10 11`
- ensembles: all `ens*` directories already represented in the local SFC tree

### 3. Legacy SFC scp workflow for ens1

Script:
- [scripts/copy_geos_s2s3_ens1_sepnov.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov.pbs)

Defaults:
- collection: `sfc_tavg_3hr_glo_L720x361_sfc`
- init dates: `1999-2024`, late August
- forecast months: `09 10 11`

### 4. Legacy ATM scp workflow for ens1

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
- it writes queue, skip, error, and raw output logs under the project data root
- it records `.shiftc_submitted` markers after successful shiftc submissions

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

Move any files that were previously staged under the old ATM root back into
the project data tree. Start with the dry run:

```bash
DRY_RUN=1 bash scripts/move_geoss2s3_atm_to_project_data.sh
```

If the dry run looks correct, run the move:

```bash
DRY_RUN=0 bash scripts/move_geoss2s3_atm_to_project_data.sh
```

Audit ATM coverage against the local SFC tree:

```bash
bash scripts/audit_atm_vs_sfc_progress.sh
```

Resume missing ATM transfers with shiftc:

```bash
qsub -v TARGET_ROOT=/nobackupp27/afahad/project/GEOS-S2S_TC/data,INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/submit_shiftc_atm_from_sfc_missing.pbs
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
- The move script uses `rsync --ignore-existing --remove-source-files`, so files already present in the project data tree are not overwritten.
- After checking a successful move, empty old staging directories can be removed manually with `find /nobackupp27/afahad/GEOSS2S3_atm/GEOS_fcst -type d -empty -delete`.
