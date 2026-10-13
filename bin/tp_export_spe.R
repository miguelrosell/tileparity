#!/usr/bin/env Rscript
# Process 4 (R) - one SpatialExperiment per sample and one combined per scale.
#
# Reads exactly the files the AnnData export reads (matched by the prefix <sample>_mpp<scale>):
#   <prefix>.tiles.tsv.gz    tile table from Process 1 (only tiles with keep == TRUE are exported)
#   <prefix>.prepare.json    provenance and thumbnail scale factor
#   <prefix>.thumbnail.png   low-resolution image for imgData()
#   <prefix>.uni2h.h5        Output2a  (optional)
#   <prefix>.lbp.h5          Output2b  (optional)
#   <prefix>.counts.h5       Output3   (genes x tiles, may have 0 genes)
#
# Layout of each object:
#   assay "counts"     genes x tiles (sum of Visium HD bins, or the Visium spot)
#   colData            the tile table (QC metrics, sample metadata, ...)
#   spatialCoords      tile centre in full-resolution pixels (pxl_col_in_fullres, pxl_row_in_fullres)
#   reducedDims        UNI2H (1536), LBP_HE, LBP_GREY, INTENSITY_HE
#   imgData            the thumbnail, with its scale factor to full-resolution pixels
#   metadata$tileparity provenance (parameters, model, LBP radii, ...)

suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(SummarizedExperiment)
    library(S4Vectors)
    library(Matrix)
    library(rhdf5)
    library(jsonlite)
})

parse_args <- function(argv) {
    out <- list(input_dir = ".", combined_name = NULL, no_combined = FALSE)
    i <- 1
    while (i <= length(argv)) {
        a <- argv[[i]]
        if (a == "--input-dir") { out$input_dir <- argv[[i + 1]]; i <- i + 2 }
        else if (a == "--combined-name") { out$combined_name <- argv[[i + 1]]; i <- i + 2 }
        else if (a == "--no-combined") { out$no_combined <- TRUE; i <- i + 1 }
        else stop("unknown argument: ", a)
    }
    if (is.null(out$combined_name)) stop("--combined-name is required")
    out
}

# h5py stores arrays row-major; rhdf5 returns them with reversed dimensions
read_matrix_rows <- function(file, name) {
    x <- h5read(file, name)
    if (is.null(dim(x))) x <- matrix(x, ncol = 1)
    t(x)
}

NUMERIC_COLUMNS <- c("scale_mpp", "tile_px", "tile_um", "row", "col", "x0_px", "y0_px", "cx_px", "cy_px",
                     "cx_fullres", "cy_fullres", "cx_um", "cy_um", "tissue_fraction", "st_coverage",
                     "n_st_units", "lapvar", "pen_fraction", "dark_fraction")
BOOL_COLUMNS <- c("inside_image", "blur_flag", "pen_flag", "dark_flag", "keep")

# Same typing rules as tileparity_lib.tiling.read_tiles(): numbers, booleans, everything else text
read_tiles <- function(path) {
    df <- utils::read.delim(path, check.names = FALSE, stringsAsFactors = FALSE,
                            colClasses = "character", na.strings = character(0))
    for (col in names(df)) {
        if (col %in% BOOL_COLUMNS) {
            df[[col]] <- tolower(df[[col]]) == "true"
        } else if (col %in% NUMERIC_COLUMNS) {
            df[[col]] <- suppressWarnings(as.numeric(ifelse(df[[col]] == "", NA, df[[col]])))
        }
    }
    df[df$keep, , drop = FALSE]
}

read_features <- function(path, tile_ids) {
    ids <- as.character(h5read(path, "tile_id"))
    idx <- match(tile_ids, ids)
    if (anyNA(idx)) stop(basename(path), ": ", sum(is.na(idx)), " kept tiles have no features")
    groups <- h5ls(path)
    block_names <- unique(groups$name[groups$group == "/blocks"])
    out <- list()
    for (b in block_names) {
        X <- read_matrix_rows(path, paste0("/blocks/", b, "/X"))
        colnames(X) <- as.character(h5read(path, paste0("/blocks/", b, "/columns")))
        X <- X[idx, , drop = FALSE]
        rownames(X) <- tile_ids
        storage.mode(X) <- "double"
        out[[b]] <- X
    }
    attrs <- h5readAttributes(path, "/")
    list(blocks = out, attrs = attrs)
}

read_counts <- function(path, tile_ids) {
    n_genes <- as.integer(h5readAttributes(path, "/matrix")$n_genes)
    tiles <- as.character(h5read(path, "/matrix/barcodes"))
    idx <- match(tile_ids, tiles)
    if (anyNA(idx)) stop(basename(path), ": ", sum(is.na(idx)), " kept tiles have no counts")
    if (length(n_genes) == 0 || n_genes == 0) {
        m <- sparseMatrix(i = integer(0), j = integer(0), x = numeric(0), dims = c(0L, length(tile_ids)))
        colnames(m) <- tile_ids
        return(list(counts = m, gene_names = character(0)))
    }
    shape <- as.integer(h5read(path, "/matrix/shape"))
    m <- sparseMatrix(i = as.integer(h5read(path, "/matrix/indices")) + 1L,
                      p = as.integer(h5read(path, "/matrix/indptr")),
                      x = as.numeric(h5read(path, "/matrix/data")),
                      dims = shape)
    ids <- as.character(h5read(path, "/matrix/features/id"))
    gene_names <- as.character(h5read(path, "/matrix/features/name"))
    rownames(m) <- ids
    colnames(m) <- tiles
    list(counts = m[, idx, drop = FALSE], gene_names = gene_names)
}

build_spe <- function(dir, prefix) {
    f <- function(suffix) file.path(dir, paste0(prefix, suffix))
    tiles <- read_tiles(f(".tiles.tsv.gz"))
    tile_ids <- tiles$tile_id
    prep <- fromJSON(f(".prepare.json"), simplifyVector = FALSE)
    sample <- if (nrow(tiles)) tiles$sample_id[1] else prep$sample_id

    rd <- list()
    provenance <- list(prepare = prep)
    if (file.exists(f(".uni2h.h5"))) {
        u <- read_features(f(".uni2h.h5"), tile_ids)
        rd$UNI2H <- u$blocks$UNI2H
        provenance$uni2h <- u$attrs
    }
    if (file.exists(f(".lbp.h5"))) {
        l <- read_features(f(".lbp.h5"), tile_ids)
        if (all(c("LBP_H", "LBP_E") %in% names(l$blocks))) rd$LBP_HE <- cbind(l$blocks$LBP_H, l$blocks$LBP_E)
        if ("LBP_GREY" %in% names(l$blocks)) rd$LBP_GREY <- l$blocks$LBP_GREY
        if ("INTENSITY_HE" %in% names(l$blocks)) rd$INTENSITY_HE <- l$blocks$INTENSITY_HE
        provenance$lbp <- l$attrs
    }
    cnt <- read_counts(f(".counts.h5"), tile_ids)

    cd <- DataFrame(tiles, check.names = FALSE)
    rownames(cd) <- tile_ids
    coords <- as.matrix(tiles[, c("cx_fullres", "cy_fullres")])
    colnames(coords) <- c("pxl_col_in_fullres", "pxl_row_in_fullres")
    rownames(coords) <- tile_ids

    spe <- SpatialExperiment(
        assays = list(counts = cnt$counts),
        colData = cd,
        spatialCoords = coords,
        sample_id = sample,
        reducedDims = rd
    )
    if (nrow(spe) > 0) rowData(spe)$gene_name <- cnt$gene_names
    thumb <- f(".thumbnail.png")
    if (file.exists(thumb) && !is.null(prep$thumbnail$scale_to_fullres)) {
        spe <- addImg(spe, imageSource = thumb, scaleFactor = as.numeric(prep$thumbnail$scale_to_fullres),
                      sample_id = sample, image_id = "thumbnail", load = TRUE)
    }
    metadata(spe)$tileparity <- provenance
    spe
}

combine_spe <- function(spes) {
    # Same rule as the Python export: samples with expression data are combined on their
    # shared genes and image-only samples are left out; with fewer than two expression
    # samples, all samples are combined with image features only.
    spes <- Filter(function(s) ncol(s) > 0, spes)
    has_genes <- vapply(spes, function(s) nrow(s) > 0, logical(1))
    if (sum(has_genes) >= 2) {
        if (any(!has_genes)) message("image-only samples are not included in the combined object")
        spes <- spes[has_genes]
        common <- Reduce(intersect, lapply(spes, rownames))
        if (length(common) == 0) stop("samples share no genes; cannot combine counts")
        spes <- lapply(spes, function(s) s[common, ])
    } else if (any(has_genes)) {
        message("fewer than two samples have expression data: the combined object keeps image features only")
        spes <- lapply(spes, function(s) s[integer(0), ])
    }
    if (length(spes) < 2) return(NULL)
    gene_names <- if (nrow(spes[[1]]) > 0) rowData(spes[[1]])$gene_name else character(0)
    dims <- Reduce(intersect, lapply(spes, reducedDimNames))
    cols <- Reduce(union, lapply(spes, function(s) colnames(colData(s))))
    spes <- lapply(spes, function(s) {
        reducedDims(s) <- reducedDims(s)[dims]
        for (cname in setdiff(cols, colnames(colData(s)))) colData(s)[[cname]] <- NA_character_
        colData(s) <- colData(s)[, cols]
        rowData(s) <- NULL  # gene symbols may differ between annotations; restored below
        metadata(s) <- list()
        s
    })
    combined <- do.call(cbind, unname(spes))
    if (nrow(combined) > 0) rowData(combined)$gene_name <- gene_names
    combined
}

main <- function() {
    args <- parse_args(commandArgs(trailingOnly = TRUE))
    files <- list.files(args$input_dir, pattern = "\\.tiles\\.tsv\\.gz$")
    prefixes <- sort(sub("\\.tiles\\.tsv\\.gz$", "", files))
    if (length(prefixes) == 0) stop("no *.tiles.tsv.gz files found")
    spes <- list()
    for (p in prefixes) {
        spe <- build_spe(args$input_dir, p)
        saveRDS(spe, paste0(p, ".rds"))
        message(sprintf("%s: %d tiles x %d genes; reducedDims: %s", p, ncol(spe), nrow(spe),
                        paste(reducedDimNames(spe), collapse = ", ")))
        spes[[p]] <- spe
    }
    if (!args$no_combined) {
        combined <- combine_spe(spes)
        if (!is.null(combined)) saveRDS(combined, paste0(args$combined_name, ".rds"))
    }
}

main()
