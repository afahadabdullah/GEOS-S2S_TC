"""Shared GEOS candidate-inventory filenames.

The old broad glob can silently mix accepted-only and all-center schemas.
Keep the all-center inventory as the canonical default for diagnostics that
need structure-gate flags.
"""

CANONICAL_ALL_CENTERS_PREFIX = "geos_candidate_thresholds_allcenters_19910824_20240829_all"
CANONICAL_ALL_CENTERS_CANDIDATES = f"data/calibration/{CANONICAL_ALL_CENTERS_PREFIX}_candidates.csv"
CANONICAL_ALL_CENTERS_THRESHOLDS = f"data/calibration/{CANONICAL_ALL_CENTERS_PREFIX}.csv"
CANONICAL_ALL_CENTERS_PROGRESS = f"data/calibration/{CANONICAL_ALL_CENTERS_PREFIX}_progress.csv"

