#!/usr/bin/env bash
# Run the full pipeline in order. Set RAW_DIR to keep the large downloads elsewhere.
set -euo pipefail
cd "$(dirname "$0")"
for s in 01_download_data.py 02_prepare_wells.py 03_spatial_autocorrelation.py \
         04_covariates_depth_aquifer.py 05_variogram_kriging.py 06_county_ranking_and_maps.py; do
  echo "== $s"
  python -I "$s"
done
