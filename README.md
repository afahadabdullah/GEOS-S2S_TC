# GEOS-S2S_TC

Utilities for staging GEOS S2S3 forecast data for tropical cyclone evaluation.

Current focus:
- copy late-August `ens1` forecast archives from the remote GEOS S2S3 store on `lou`
- preserve the original directory layout under local storage
- untar monthly forecast files in place
- resume automatically across multiple PBS jobs

This repo is being used to support later tropical cyclone diagnostics such as GPI and ACE for GEOS S2Sv2 vs S2Sv3 comparisons.

## Repository Layout

```text
GEOS-S2S_TC/
|-- config/
|   |-- init_dates_late_aug_1991_2024.txt
|   `-- init_dates_late_aug_1999_2024.txt
|-- scripts/
|   |-- copy_geos_s2s3_ens1_sepnov.pbs
|   `-- copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs
`-- README.md
```

## Data Assumptions

- Remote host: `lou`
- Remote root: `/lou/la5/knakada/GEOSS2S3/GEOS_fcst`
- Local destination root: `/nobackupp27/afahad/GEOSS2S3_atm`
- Ensemble: `ens1`
- Forecast months: September, October, November
- Init dates: late August (`0824` and `0829`) from the supplied manifest files

Downloaded files are stored under:

```text
/nobackupp27/afahad/GEOSS2S3_atm/GEOS_fcst/<init_date>/ens1/<collection>/
```

## Available PBS Jobs

### 1. Surface 3-hourly collection

Script:
- [scripts/copy_geos_s2s3_ens1_sepnov.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov.pbs)

Defaults:
- collection: `sfc_tavg_3hr_glo_L720x361_sfc`
- init dates: `1999-2024`, late August
- forecast months: `09 10 11`

### 2. Atmospheric 6-hourly collection

Script:
- [scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs](/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs)

Defaults:
- collection: `atm_inst_6hr_glo_L720x361_p49`
- init dates: `1991-2024`, late August
- forecast months: `09 10 11`

## How the Jobs Work

The pull workflow is restart-safe:

- it copies one monthly tar file at a time
- it untars in place after each successful copy
- it writes `.untar_done` markers next to completed tar files
- it tracks progress in a state directory under `/nobackupp27/afahad/GEOSS2S3_atm/job_state/`
- it can bootstrap progress from existing `.untar_done` files created by older runs
- it can resubmit itself before walltime is exhausted

State files include:
- `progress.tsv`
- `next_index.txt`
- `completed.ok`

## Submission Examples

From the repo `scripts/` directory:

```bash
qsub copy_geos_s2s3_ens1_sepnov.pbs
qsub copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs
```

From the repo root:

```bash
qsub scripts/copy_geos_s2s3_ens1_sepnov.pbs
qsub scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs
```

Explicit submit with resolved paths:

```bash
qsub -v INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1999_2024.txt,RESUBMIT_SCRIPT=/nobackupp27/afahad/project/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov.pbs /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov.pbs
```

```bash
qsub -v INIT_DATES_FILE=/nobackupp27/afahad/project/GEOS-S2S_TC/config/init_dates_late_aug_1991_2024.txt,RESUBMIT_SCRIPT=/nobackupp27/afahad/project/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs /nobackupp27/afahad/project/GEOS-S2S_TC/scripts/copy_geos_s2s3_ens1_sepnov_atm_inst6hr_p49.pbs
```

## Notes

- The jobs assume passwordless `ssh/scp` access from the compute node to `lou`.
- If a remote file is missing, the job records that and moves on.
- If `scp` or `tar` fails for a file, the job stops so the same file can be retried on the next run.
- The scripts are currently set up for `ens1` only.
