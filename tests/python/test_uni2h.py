import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from tileparity_lib import uni2h  # noqa: E402
from tileparity_lib.image import write_tiled_tiff  # noqa: E402


def test_adapter_keeps_the_tile_pixels_and_normalises(tmp_path):
    import pyvips

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(448, 448, 3), dtype=np.uint8)
    path = tmp_path / "img.tif"
    write_tiled_tiff(pyvips.Image.new_from_array(rgb).copy(interpretation="srgb"), path, compression="deflate")
    ds = uni2h.TileDataset(str(path), pd.DataFrame({"x0_px": [224], "y0_px": [0]}), 224)
    x = ds[0]
    assert tuple(x.shape) == (3, 224, 224)
    mean = np.array(uni2h.IMAGENET_MEAN)[:, None, None]
    std = np.array(uni2h.IMAGENET_STD)[:, None, None]
    expected = (rgb[0:224, 224:448].transpose(2, 0, 1) / 255.0 - mean) / std
    np.testing.assert_allclose(x.numpy(), expected, atol=1e-5)


def test_test_mode_model_outputs_1536d(tmp_path):
    import pyvips

    rgb = np.full((224, 448, 3), 200, np.uint8)
    path = tmp_path / "img.tif"
    write_tiled_tiff(pyvips.Image.new_from_array(rgb).copy(interpretation="srgb"), path, compression="deflate")
    model, source = uni2h.build_model(None, test_mode=True)
    emb, info = uni2h.embed(str(path), pd.DataFrame({"x0_px": [0, 224], "y0_px": [0, 0]}), model,
                            device="cpu", batch_size=2, num_workers=0)
    assert emb.shape == (2, 1536) and np.isfinite(emb).all()
    assert source.startswith("test-mode")


def test_rgb_he_input_reads_the_h_and_e_only_bands(tmp_path):
    import pyvips

    from tileparity_lib import channels

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(224, 448, 3), dtype=np.uint8)
    bands = channels.band_order(["rgb"], ["rgb_he"])
    arr = channels.rgb_to_channels_numpy(rgb, 0.3, 0.15, bands)
    path = tmp_path / "channels.tif"
    write_tiled_tiff(pyvips.Image.new_from_array(arr).copy(interpretation="multiband"), path, compression="deflate")
    idx = [bands.index(b) for b in channels.CHANNEL_SETS["rgb_he"]]
    ds = uni2h.TileDataset(str(path), pd.DataFrame({"x0_px": [224], "y0_px": [0]}), 224, idx)
    mean = np.array(uni2h.IMAGENET_MEAN)[:, None, None]
    std = np.array(uni2h.IMAGENET_STD)[:, None, None]
    expected = (channels.rgb_he_numpy(rgb)[0:224, 224:448].transpose(2, 0, 1) / 255.0 - mean) / std
    np.testing.assert_allclose(ds[0].numpy(), expected, atol=1e-5)
    assert uni2h.block_name("rgb_he") == "UNI2H_RGB_HE" and uni2h.columns("rgb")[0] == "uni2h_rgb_0000"
    with pytest.raises(ValueError):
        uni2h.TileDataset(str(path), pd.DataFrame({"x0_px": [0], "y0_px": [0]}), 224, [0, 1])


def test_embedding_script_writes_one_block_per_input_set(tmp_path):
    import json

    import pyvips

    import tp_embed_uni2h
    from tileparity_lib import channels, features

    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, size=(224, 448, 3), dtype=np.uint8)
    write_tiled_tiff(pyvips.Image.new_from_array(rgb).copy(interpretation="srgb"), tmp_path / "s.resampled.tif")
    bands = channels.band_order([], ["rgb", "rgb_he"])
    write_tiled_tiff(pyvips.Image.new_from_array(channels.rgb_to_channels_numpy(rgb, 0.3, 0.15, bands))
                     .copy(interpretation="multiband"), tmp_path / "s.channels.tif")
    (tmp_path / "s.channels.json").write_text(json.dumps({"bands": bands, "stain_vectors": {"method": "test"}}))
    pd.DataFrame({"tile_id": ["a", "b"], "tile_px": [224, 224], "x0_px": [0, 224], "y0_px": [0, 0],
                  "extract": [True, True]}).to_csv(tmp_path / "s.tiles.tsv.gz", sep="\t", index=False)
    tp_embed_uni2h.main(["--image", str(tmp_path / "s.resampled.tif"), "--tiles", str(tmp_path / "s.tiles.tsv.gz"),
                         "--prefix", str(tmp_path / "s"), "--inputs", "rgb,rgb_he", "--channels-image",
                         str(tmp_path / "s.channels.tif"), "--channels-json", str(tmp_path / "s.channels.json"),
                         "--test-mode", "--device", "cpu", "--num-workers", "0", "--batch-size", "2"])
    ids, blocks, attrs = features.read(str(tmp_path / "s.uni2h.h5"))
    assert list(ids) == ["a", "b"] and set(blocks) == {"UNI2H_RGB", "UNI2H_RGB_HE"}
    assert blocks["UNI2H_RGB"][0].shape == (2, 1536)
    assert blocks["UNI2H_RGB_HE"][2]["input"] == "rgb_he" and attrs["inputs"] == ["rgb", "rgb_he"]
    assert not np.allclose(blocks["UNI2H_RGB"][0], blocks["UNI2H_RGB_HE"][0])  # different pixels, different tokens
    with pytest.raises(SystemExit):  # rgb_he without the channels image
        tp_embed_uni2h.main(["--image", str(tmp_path / "s.resampled.tif"), "--tiles", str(tmp_path / "s.tiles.tsv.gz"),
                             "--prefix", str(tmp_path / "x"), "--inputs", "rgb_he", "--test-mode"])
