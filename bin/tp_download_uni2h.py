#!/usr/bin/env python3
"""Download the gated UNI2-h weights (pytorch_model.bin) from Hugging Face.

Needs a Hugging Face account with access to MahmoodLab/UNI2-h and a token in the
HF_TOKEN environment variable (or a prior `huggingface-cli login`). Run it once and
pass the file to the pipeline with --uni2h_weights, which also lets compute nodes
without internet access run UNI2-h.
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=".", help="where to write pytorch_model.bin")
    ap.add_argument("--repo", default="MahmoodLab/UNI2-h")
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".hf_download_cache"
    path = hf_hub_download(args.repo, filename="pytorch_model.bin", local_dir=str(cache),
                           token=os.environ.get("HF_TOKEN") or None)
    target = out / "pytorch_model.bin"
    shutil.move(path, target)
    shutil.rmtree(cache, ignore_errors=True)
    h = hashlib.sha256()
    with open(target, "rb") as fh:
        while block := fh.read(1 << 24):
            h.update(block)
    digest = h.hexdigest()
    (out / "pytorch_model.bin.sha256").write_text(f"{digest}  pytorch_model.bin\n")
    print(f"wrote {target} (sha256 {digest})")


if __name__ == "__main__":
    main()
