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
