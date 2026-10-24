"""Lineage: where every output comes from.

Every process writes ``<prefix>.<step>.lineage.json`` next to its outputs. The file
describes the step that made the outputs (command, parameters, software, input and
output files with checksums) and contains, once each, the records of all the steps
upstream of it, so the last file of a chain (e.g. the export) holds the complete route
from the original image to the final object::

    {
      "schema": "tileparity-lineage/1",
      "record": {
        "uid": "lbp:A_P1_mpp0.5000:3f9c2a1b7d0e",
        "step": "lbp", "process": "FASTLBP_EXTRACT", "prefix": "A_P1_mpp0.5000",
        "created_utc": "2026-10-24T10:00:00Z", "tileparity_version": "0.2.0",
        "command": ["tp_extract_lbp.py", "--channels-image", ...],
        "parameters": {...}, "software": {"python": "3.13.1", "fastlbp": "0.3.4", ...},
        "inputs": {"channels": {"name": ..., "bytes": ..., "sha256": ...}, ...},
        "outputs": {"features": {...}},
        "parents": ["lbpchannels:A_P1_mpp0.5000:..."]
      },
      "ancestors": {"lbpchannels:...": {...}, "prepare:...": {...}}
    }

``route(lineage)`` lists the steps from the first to the last, which is what the
"Where it comes from" line under every figure shows.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from importlib import metadata as _metadata
from pathlib import Path

from . import __version__

SCHEMA = "tileparity-lineage/1"
FULL_HASH_LIMIT = 2 * 1024**3  # files up to 2 GiB are hashed completely
PARTIAL_HASH_BYTES = 64 * 1024**2  # larger files: first and last 64 MiB plus the size


def file_info(path: str | Path) -> dict:
    """Name, size and checksum of a file (or a short description of a directory)."""
    p = Path(path)
    if not p.exists():
        return {"name": p.name, "exists": False}
    if p.is_dir():
        return {"name": p.name, "directory": True}
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        if size <= FULL_HASH_LIMIT:
            while block := fh.read(1 << 24):
                h.update(block)
            return {"name": p.name, "bytes": size, "sha256": h.hexdigest()}
        h.update(fh.read(PARTIAL_HASH_BYTES))
        fh.seek(-PARTIAL_HASH_BYTES, os.SEEK_END)
        h.update(fh.read(PARTIAL_HASH_BYTES))
        h.update(str(size).encode())
    return {"name": p.name, "bytes": size, "sha256_head_tail_size": h.hexdigest()}


def software(packages: list[str]) -> dict:
    out = {"tileparity": __version__, "python": sys.version.split()[0]}
    for pkg in packages:
        try:
            out[pkg] = _metadata.version(pkg)
        except _metadata.PackageNotFoundError:
            out[pkg] = "NA"
    return out


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a tileparity lineage file")
    return data


def build(step: str, process: str, prefix: str, parameters: dict, inputs: dict, outputs: dict,
          parents: list[str | Path] | None = None, packages: list[str] | None = None) -> dict:
    """Lineage of one step. ``inputs`` and ``outputs`` map a role to a path (or None);
    ``parents`` are the lineage files of the steps that produced the inputs."""
    ancestors: dict = {}
    parent_uids: list[str] = []
    for path in parents or []:
        if not path:
            continue
        parent = load(path)
        ancestors.update(parent.get("ancestors", {}))
        ancestors[parent["record"]["uid"]] = parent["record"]
        parent_uids.append(parent["record"]["uid"])
    record = {
        "step": step,
        "process": process,
        "prefix": prefix,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tileparity_version": __version__,
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "parameters": _jsonable(parameters),
        "software": software(packages or []),
        "inputs": {k: file_info(v) for k, v in inputs.items() if v},
        "outputs": {k: file_info(v) for k, v in outputs.items() if v},
        "parents": parent_uids,
    }
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:12]
    record = {"uid": f"{step}:{prefix}:{digest}", **record}
    return {"schema": SCHEMA, "record": record, "ancestors": ancestors}


def write(path: str | Path, lineage: dict) -> None:
    Path(path).write_text(json.dumps(lineage, indent=2, default=str))


def route(lineage: dict) -> list[dict]:
    """Records of this step and all its ancestors, upstream first."""
    records = {**lineage.get("ancestors", {}), lineage["record"]["uid"]: lineage["record"]}
    order: list[str] = []

    def visit(uid: str) -> None:
        if uid in order or uid not in records:
            return
        for parent in records[uid].get("parents", []):
            visit(parent)
        order.append(uid)

    visit(lineage["record"]["uid"])
    return [records[u] for u in order]


def route_text(lineage: dict | None, extra_steps: list[str] | None = None) -> str:
    """One line for a figure caption: the chain of processes and the original inputs."""
    if lineage is None:
        steps = list(extra_steps or [])
        return " -> ".join(steps)
    recs = route(lineage)
    chain = [short_process(r["process"]) for r in recs] + list(extra_steps or [])
    first = recs[0]
    sources = ", ".join(f"{role} '{info.get('name')}'" for role, info in first.get("inputs", {}).items())
    return f"{' -> '.join(chain)} (tileparity {first['tileparity_version']}); original inputs: {sources}"


def source_text(process: str, inputs: dict, parents: list[str | Path] | None = None, context: str = "") -> str:
    """The "where it comes from" line for a figure made by ``process`` before its own
    lineage file exists: upstream chain of processes, original input files, ``context``."""
    chain: list[str] = []
    roots: dict = {}
    for path in parents or []:
        if not path:
            continue
        for rec in route(load(path)):
            if short_process(rec["process"]) not in chain:
                chain.append(short_process(rec["process"]))
            if not rec.get("parents"):
                roots.update({role: info.get("name") for role, info in rec.get("inputs", {}).items()})
    chain.append(short_process(process))
    if not roots:
        roots = {role: Path(str(p)).name for role, p in inputs.items() if p}
    sources = ", ".join(f"{role} '{name}'" for role, name in roots.items())
    text = f"{' -> '.join(chain)} (tileparity {__version__}); original inputs: {sources}"
    return f"{text}; {context}" if context else text


def short_process(process: str) -> str:
    """'TILEPARITY_PIPELINE:TILEPARITY:FASTLBP_EXTRACT' -> 'FASTLBP_EXTRACT' (captions only;
    lineage records keep the full name)."""
    return str(process).rsplit(":", 1)[-1]


def add_cli(ap) -> None:
    """Command-line options shared by every tp_*.py script."""
    ap.add_argument("--process", default=None, help="Nextflow process name, recorded in the lineage file")
    ap.add_argument("--lineage-parent", action="append", default=[],
                    help="lineage file of a step whose outputs are inputs here (repeatable)")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, TypeError):
            return str(obj)
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    return obj
