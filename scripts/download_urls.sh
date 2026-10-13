#!/usr/bin/env bash
# Resumable download of the files listed in a TSV (columns: sample, kind, url), e.g. the
# Visium HD microscope images and binned outputs of cohort A copied from the 10x
# Genomics dataset pages (assets/cohort_a_10x_urls.tsv is a template).
#
#     bash scripts/download_urls.sh assets/cohort_a_10x_urls.tsv /data/tileparity/raw
#
# Each file goes to <OUTDIR>/<sample>/; .tar.gz archives (binned outputs) are extracted.
set -euo pipefail
table=${1:?usage: download_urls.sh TABLE.tsv OUTDIR}
outdir=${2:?usage: download_urls.sh TABLE.tsv OUTDIR}
tail -n +2 "$table" | while IFS=$'\t' read -r sample kind url; do
    [[ -z "${url// }" || "$url" == TODO* ]] && { echo "skip ${sample} ${kind}: no URL" >&2; continue; }
    dest="${outdir}/${sample}"
    mkdir -p "$dest"
    file="${dest}/$(basename "${url%%\?*}")"
    echo "downloading ${sample} ${kind} -> ${file}"
    curl -L --fail --retry 5 -C - -o "$file" "$url"
    if [[ "$file" == *.tar.gz ]]; then
        tar -xzf "$file" -C "$dest" && rm "$file"
    fi
done
