# Citations

If you use tileparity, please cite the tools and data it builds on.

* **Nextflow** — Di Tommaso P. et al. Nextflow enables reproducible computational workflows. *Nat Biotechnol* 35, 316–319 (2017). doi:10.1038/nbt.3820
* **nf-core** (conventions) — Ewels P.A. et al. The nf-core framework for community-curated bioinformatics pipelines. *Nat Biotechnol* 38, 276–278 (2020).
* **UNI / UNI2-h** — Chen R.J. et al. Towards a general-purpose foundation model for computational pathology. *Nat Med* 30, 850–862 (2024). Weights: https://huggingface.co/MahmoodLab/UNI2-h (CC-BY-NC-ND 4.0).
* **fastLBP** — https://github.com/imbg-ua/fastLBP (GPL-3.0).
* **Local binary patterns** — Ojala T., Pietikäinen M., Mäenpää T. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE TPAMI* 24, 971–987 (2002).
* **Colour deconvolution** — Ruifrok A.C., Johnston D.A. Quantification of histochemical staining by color deconvolution. *Anal Quant Cytol Histol* 23, 291–299 (2001).
* **Otsu threshold** — Otsu N. A threshold selection method from gray-level histograms. *IEEE Trans Syst Man Cybern* 9, 62–66 (1979).
* **libvips / pyvips** — Martinez K., Cupitt J. VIPS – a highly tuned image processing software architecture. *IEEE ICIP* (2005).
* **scikit-image** — van der Walt S. et al. *PeerJ* 2, e453 (2014).
* **SpatialExperiment** — Righelli D. et al. SpatialExperiment: infrastructure for spatially-resolved transcriptomics data in R using Bioconductor. *Bioinformatics* 38, 3128–3131 (2022).
* **AnnData** — Virshup I. et al. anndata: Access and store annotated data matrices. *JOSS* 9, 4371 (2024).
* **HEST-1k** — Jaume G. et al. HEST-1k: A dataset for spatial transcriptomics and histology image analysis. *NeurIPS* (2024).
* **Visium HD colorectal cancer data** — de Oliveira M.F. et al. High-definition spatial transcriptomic profiling of immune cell populations in colorectal cancer. *Nat Genet* (2025). doi:10.1038/s41588-025-02193-3

Methods used by the options added in 0.2.0:

* **Dice overlap** (mask comparison) — Dice L.R. Measures of the amount of ecologic association between species. *Ecology* 26, 297–302 (1945).
* **Optical-density background threshold** (`--mask_method od`, reference value) — Macenko M. et al. A method for normalizing histology slides for quantitative analysis. *IEEE ISBI*, 1107–1110 (2009).
* **Count splitting** (the multinomial split of Visium spots follows the same principle) — Neufeld A., Gao L.L., Popp J., Battle A., Witten D. Inference after latent variable estimation for single-cell RNA sequencing data. arXiv:2207.00554 (2022); *Biostatistics* (2024).
* **Hellinger transform** (LBP analysis recipe) — Legendre P., Gallagher E.D. Ecologically meaningful transformations for ordination of species data. *Oecologia* 129, 271–280 (2001).
* **Multiple factor analysis** (equal weight per LBP block) — Escofier B., Pagès J. Multiple factor analysis (AFMULT package). *Comput Stat Data Anal* 18, 121–140 (1994).
* **UMAP** (`--qc_umap`) — McInnes L., Healy J., Melville J. UMAP: Uniform Manifold Approximation and Projection for dimension reduction. arXiv:1802.03426 (2018).
* **scikit-learn** (PCA for `--qc_umap`) — Pedregosa F. et al. Scikit-learn: machine learning in Python. *JMLR* 12, 2825–2830 (2011).
* **Plotly** (interactive QC page) — Plotly Technologies Inc. Collaborative data science. Montréal (2015). https://plot.ly

Methods used by the options added in 0.3.0:

* **Stain vectors estimated per slide** (`--stain_vectors slide`) — Macenko M. et al. A method for normalizing histology slides for quantitative analysis. *IEEE ISBI*, 1107–1110 (2009). The H&E-only image (`rgb_he`) uses the colour-deconvolution model of Ruifrok & Johnston (2001), above.
