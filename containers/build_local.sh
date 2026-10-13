#!/usr/bin/env bash
# Build the three images locally with the names the pipeline expects, e.g. before the
# GitHub Actions workflow has published them:   bash containers/build_local.sh
set -euo pipefail
cd "$(dirname "$0")"
version=$(grep -E "^\s*version\s*=" ../nextflow.config | head -1 | sed -E "s/.*'(.*)'.*/\1/")
for name in python uni2h r; do
    docker build -t "ghcr.io/miguelrosell/tileparity-${name}:${version}" "${name}"
done
