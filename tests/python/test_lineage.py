"""Lineage files and figure captions."""

import json

import numpy as np
from PIL import Image

from tileparity_lib import figures, lineage


def _write(tmp_path, name, content=b"x"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_lineage_inherits_every_ancestor_once(tmp_path):
    img = _write(tmp_path, "slide.tif", b"pixels")
    tiles = _write(tmp_path, "s.tiles.tsv.gz", b"tiles")
    prep = lineage.build("prepare", "TILEPARITY_PREPARE", "s", {"mpp": 0.5}, {"image": img}, {"tiles": tiles})
    lineage.write(tmp_path / "s.prepare.lineage.json", prep)
    chans = lineage.build("channels", "TILEPARITY_CHANNELS", "s", {}, {}, {},
                          parents=[tmp_path / "s.prepare.lineage.json"])
    lineage.write(tmp_path / "s.channels.lineage.json", chans)
    lbp = lineage.build("lbp", "FASTLBP_EXTRACT", "s", {}, {}, {}, parents=[tmp_path / "s.channels.lineage.json"])
    lineage.write(tmp_path / "s.lbp.lineage.json", lbp)
    st = lineage.build("st", "TILEPARITY_AGGREGATEST", "s", {}, {}, {}, parents=[tmp_path / "s.prepare.lineage.json"])
    lineage.write(tmp_path / "s.st.lineage.json", st)
    export = lineage.build("anndata", "TILEPARITY_EXPORTANNDATA", "s", {}, {}, {},
                           parents=[tmp_path / "s.lbp.lineage.json", tmp_path / "s.st.lineage.json"])
    assert len(export["ancestors"]) == 4  # prepare appears once although both parents descend from it
    route = [r["process"] for r in lineage.route(export)]
    assert route[0] == "TILEPARITY_PREPARE" and route[-1] == "TILEPARITY_EXPORTANNDATA"
    assert route.index("TILEPARITY_CHANNELS") < route.index("FASTLBP_EXTRACT")
    assert prep["record"]["inputs"]["image"]["sha256"] == __import__("hashlib").sha256(b"pixels").hexdigest()
    assert prep["record"]["uid"].startswith("prepare:s:")
    text = lineage.source_text("FASTLBP_EXTRACT", {}, parents=[tmp_path / "s.channels.lineage.json"])
    assert text.startswith("TILEPARITY_PREPARE -> TILEPARITY_CHANNELS -> FASTLBP_EXTRACT")
    assert "image 'slide.tif'" in text
    json.dumps(export)  # serialisable
    # captions show the short process name; the record keeps the full one
    full = lineage.build("lbp", "TILEPARITY_PIPELINE:TILEPARITY:FASTLBP_EXTRACT", "s", {}, {}, {},
                         parents=[tmp_path / "s.channels.lineage.json"])
    assert full["record"]["process"] == "TILEPARITY_PIPELINE:TILEPARITY:FASTLBP_EXTRACT"
    assert lineage.route_text(full).startswith("TILEPARITY_PREPARE -> TILEPARITY_CHANNELS -> FASTLBP_EXTRACT (")


def test_partial_hash_for_large_files(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage, "FULL_HASH_LIMIT", 10)
    monkeypatch.setattr(lineage, "PARTIAL_HASH_BYTES", 4)
    info = lineage.file_info(_write(tmp_path, "big.bin", b"0123456789abcdef"))
    assert "sha256_head_tail_size" in info and info["bytes"] == 16


def test_caption_is_added_below_the_figure():
    img = Image.fromarray(np.zeros((50, 80, 3), np.uint8))
    out = figures.with_caption(img, "what " * 60, "how", "caveats", "source µm ≥ 3", title="t")
    assert out.width == 960 and out.height > 50 + 80
    assert figures.plain("0.5 µm/px, ≥ 50%") == "0.5 um/px, >= 50%"
