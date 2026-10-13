#!/usr/bin/env python3
"""Print an nf-core style versions.yml block: tp_versions.py PROCESS_NAME pkg1 pkg2 ..."""

import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def version(pkg: str) -> str:
    if pkg == "tileparity":
        import tileparity_lib

        return tileparity_lib.__version__
    if pkg == "python":
        return sys.version.split()[0]
    if pkg == "libvips":
        import pyvips

        return ".".join(str(pyvips.version(i)) for i in range(3))
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "NA"


if __name__ == "__main__":
    process, *packages = sys.argv[1:]
    print(f'"{process}":')
    for pkg in ["tileparity", "python", *packages]:
        print(f"    {pkg}: {version(pkg)}")
