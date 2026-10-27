#!/usr/bin/env bash
# Download the gated UNI2-h weights once; then run the pipeline with
#     --uni2h_weights <OUTDIR>/pytorch_model.bin
#
# Requirements: a Hugging Face account with access granted to MahmoodLab/UNI2-h and a
# token in HF_TOKEN. Either Python with huggingface_hub installed, or Docker.
#
#     export HF_TOKEN=hf_xxx
#     bash scripts/download_uni2h_weights.sh ~/models/uni2h           # local Python
#     USE_DOCKER=1 bash scripts/download_uni2h_weights.sh ~/models/uni2h
set -euo pipefail
outdir=${1:?usage: download_uni2h_weights.sh OUTDIR}
repo_bin="$(cd "$(dirname "$0")/../bin" && pwd)"
mkdir -p "$outdir"
: "${HF_TOKEN:?export HF_TOKEN=<your Hugging Face token> first}"
if [[ "${USE_DOCKER:-0}" == "1" ]]; then
    image=${UNI2H_IMAGE:-ghcr.io/miguelrosell/tileparity-uni2h:0.3.0}
    docker run --rm -u "$(id -u):$(id -g)" -e HF_TOKEN -e HF_HOME=/out/.hf_home \
        -v "$(cd "$outdir" && pwd)":/out -v "$repo_bin":/tp_bin:ro \
        "$image" python /tp_bin/tp_download_uni2h.py --outdir /out
else
    python3 "$repo_bin/tp_download_uni2h.py" --outdir "$outdir"
fi
