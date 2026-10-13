"""tileparity: harmonised H&E tiling for fair comparison of image feature extractors.

The library behind the pipeline's processes. Every geometric decision (pixel size,
tissue mask, tile grid, assignment of spatial-transcriptomics units to tiles) is
made once, in :mod:`tileparity_lib.prepare`, and is consumed unchanged by the
feature extractors (:mod:`tileparity_lib.uni2h`, :mod:`tileparity_lib.lbp`) and by
the expression aggregation (:mod:`tileparity_lib.st`).
"""

__version__ = "0.1.0"


def setup_logging(level: str = "INFO") -> None:
    """Log to stderr; keep libvips' own (very chatty) messages at WARNING."""
    import logging

    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("pyvips").setLevel(logging.WARNING)
