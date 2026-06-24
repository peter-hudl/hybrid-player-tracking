#!/usr/bin/env bash
# Upload processed parquet files to S3.
# Run from the root of a local clone that has the data/ directory populated.
# Requires AWS credentials with write access to hudlrd-datasets.
#
# Usage:
#   bash tools/upload_to_s3.sh [--dry-run]

set -euo pipefail

S3_PREFIX="s3://hudlrd-datasets/focus_nexus/halmstadt-u19-2025-08-16/parquet"
LOCAL_DIR="$(dirname "$0")/../data"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: data/ directory not found. Run from repo root or populate data/ first."
    exit 1
fi

DRYRUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRYRUN="--dryrun"
    echo "Dry run — no files will be uploaded."
fi

echo "Uploading to $S3_PREFIX ..."
aws s3 sync "$LOCAL_DIR" "$S3_PREFIX" \
    --exclude "*" \
    --include "*.parquet" \
    --recursive \
    $DRYRUN

echo "Done."
