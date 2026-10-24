#!/usr/bin/env Rscript
# Process 4 (R) - one SpatialExperiment per sample and one combined per scale.
#
# Reads exactly the files the AnnData export reads (matched by the prefix <sample>_mpp<scale>):
#   <prefix>.tiles.tsv.gz    tile table from Process 1 (the extracted tiles are exported; column
#                            `keep` marks those that pass every QC threshold)
#   <prefix>.prepare.json    provenance and thumbnail scale factor
#   <prefix>.thumbnail.png   low-resolution image for imgData()
#   <prefix>.uni2h.h5        Output2a  (optional)
#   <prefix>.lbp.h5          Output2b  (optional)
#   <prefix>.counts.h5       Output3   (genes x tiles, may have 0 genes)
#   <prefix>.*.lineage.json  lineage of the steps above
#
# Layout of each object:
#   assay "counts"     genes x tiles (sum of Visium HD bins, Visium spots split by area, or the spot)
#   colData            the tile table (QC metrics, sample metadata, ...)
#   spatialCoords      tile centre in full-resolution pixels (pxl_col_in_fullres, pxl_row_in_fullres)
#   reducedDims        UNI2H (1536), LBP_HE, LBP_GREY, LBP_RGB, LBP_CENTER_*, INTENSITY_HE (those computed)
#   imgData            the thumbnail, with its scale factor to full-resolution pixels
#   metadata$tileparity provenance (parameters, model, LBP radii, ...) and the lineage (JSON text)

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
    out <- list(input_dir = ".", combined_name = NULL, no_combined = FALSE, process = "TILEPARITY_EXPORTSPE",
                tileparity_version = NA_character_)
    i <- 1
    while (i <= length(argv)) {
        a <- argv[[i]]
        if (a == "--input-dir") { out$input_dir <- argv[[i + 1]]; i <- i + 2 }
        else if (a == "--combined-name") { out$combined_name <- argv[[i + 1]]; i <- i + 2 }
        else if (a == "--no-combined") { out$no_combined <- TRUE; i <- i + 1 }
        else if (a == "--process") { out$process <- argv[[i + 1]]; i <- i + 2 }
        else if (a == "--tileparity-version") { out$tileparity_version <- argv[[i + 1]]; i <- i + 2 }
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
                     "n_st_units", "st_units_equiv", "lapvar", "pen_fraction", "dark_fraction")
BOOL_COLUMNS <- c("inside_image", "blur_flag", "pen_flag", "dark_flag", "extract", "keep")
# blocks of one channel set are joined; different channel sets stay separate alternatives
COMBINED_BLOCKS <- list(
    LBP_HE = c("LBP_H", "LBP_E"), LBP_GREY = "LBP_GREY", LBP_RGB = c("LBP_R", "LBP_G", "LBP_B"),
    LBP_CENTER_HE = c("LBP_CENTER_H", "LBP_CENTER_E"), LBP_CENTER_GREY = "LBP_CENTER_GREY",
    LBP_CENTER_RGB = c("LBP_CENTER_R", "LBP_CENTER_G", "LBP_CENTER_B"), INTENSITY_HE = "INTENSITY_HE"
)

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
    sel <- if ("extract" %in% names(df)) df$extract else df$keep
    df[sel, , drop = FALSE]
}

read_features <- function(path, tile_ids) {
    ids <- as.character(h5read(path, "tile_id"))
    idx <- match(tile_ids, ids)
    if (anyNA(idx)) stop(basename(path), ": ", sum(is.na(idx)), " extracted tiles have no features")
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
    if (anyNA(idx)) stop(basename(path), ": ", sum(is.na(idx)), " extracted tiles have no counts")
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

# ---- lineage (same layout as bin/tileparity_lib/lineage.py; checksums are MD5 here) ----------
lineage_parents <- function(dir, prefix) {
    steps <- c("uni2h", "lbp", "st", "prepare")
    files <- file.path(dir, paste0(prefix, ".", steps, ".lineage.json"))
    files <- files[file.exists(files)]
    leaves <- files[!grepl("\\.prepare\\.lineage\\.json$", files)]
    if (length(leaves)) leaves else files
}

build_lineage <- function(step, process, prefix, outputs, parents, version, parameters = list()) {
    ancestors <- setNames(list(), character(0))
    uids <- character(0)
    for (f in parents) {
        p <- fromJSON(f, simplifyVector = FALSE)
        for (k in names(p$ancestors)) ancestors[[k]] <- p$ancestors[[k]]
        ancestors[[p$record$uid]] <- p$record
        uids <- c(uids, p$record$uid)
        if (is.na(version)) version <- p$record$tileparity_version
    }
    out_info <- lapply(outputs, function(f) list(name = basename(f), bytes = unname(file.size(f)),
                                                 md5 = unname(tools::md5sum(f))))
    record <- list(
        step = step, process = process, prefix = prefix,
        created_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
        tileparity_version = version,
        command = as.list(c("tp_export_spe.R", commandArgs(trailingOnly = TRUE))),
        parameters = if (length(parameters)) parameters else setNames(list(), character(0)),
        software = list(R = as.character(getRversion()),
                        SpatialExperiment = as.character(packageVersion("SpatialExperiment")),
                        rhdf5 = as.character(packageVersion("rhdf5"))),
        inputs = setNames(list(), character(0)),
        outputs = out_info,
        parents = as.list(uids)
    )
    tmp <- tempfile(fileext = ".json")
    writeLines(toJSON(record, auto_unbox = TRUE), tmp)
    digest <- substr(unname(tools::md5sum(tmp)), 1, 12)
    unlink(tmp)
    list(schema = "tileparity-lineage/1", record = c(list(uid = paste(step, prefix, digest, sep = ":")), record),
         ancestors = ancestors)
}

write_lineage <- function(lin, path) {
    writeLines(toJSON(lin, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null"), path)
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
        for (name in names(COMBINED_BLOCKS)) {
            parts <- COMBINED_BLOCKS[[name]]
            if (all(parts %in% names(l$blocks))) rd[[name]] <- do.call(cbind, unname(l$blocks[parts]))
        }
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
    # scaleFactor converts spatialCoords (full-resolution px) to thumbnail px
    if (file.exists(thumb) && !is.null(prep$thumbnail$fullres_to_thumbnail)) {
        spe <- addImg(spe, imageSource = thumb, scaleFactor = as.numeric(prep$thumbnail$fullres_to_thumbnail),
                      sample_id = sample, image_id = "thumbnail", load = TRUE)
    }
    parents <- lineage_parents(dir, prefix)
    merged <- build_lineage("sample", "tileparity", prefix, list(), parents, NA_character_)
    provenance$lineage <- as.character(toJSON(merged, auto_unbox = TRUE, null = "null", na = "null"))
    metadata(spe)$tileparity <- provenance
    list(spe = spe, parents = parents)
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
        # logical NA combines with columns of any type (text metadata, numbers) in cbind
        for (cname in setdiff(cols, colnames(colData(s)))) colData(s)[[cname]] <- rep(NA, ncol(s))
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
    leaves <- character(0)
    parents_by_sample <- list()
    for (p in prefixes) {
        built <- build_spe(args$input_dir, p)
        spe <- built$spe
        parents_by_sample[[metadata(spe)$tileparity$prepare$sample_id]] <- built$parents
        out <- paste0(p, ".rds")
        saveRDS(spe, out)
        lin <- build_lineage("spe", args$process, p, list(spe = out), built$parents, args$tileparity_version)
        write_lineage(lin, paste0(p, ".spe.lineage.json"))
        leaves <- c(leaves, paste0(p, ".spe.lineage.json"))
        message(sprintf("%s: %d tiles x %d genes; reducedDims: %s", p, ncol(spe), nrow(spe),
                        paste(reducedDimNames(spe), collapse = ", ")))
        spes[[p]] <- spe
    }
    if (!args$no_combined) {
        combined <- combine_spe(spes)
        if (!is.null(combined)) {
            out <- paste0(args$combined_name, ".rds")
            included <- sort(unique(as.character(combined$sample_id)))
            # the object carries the lineage of every sample it contains, like the per-sample objects
            merged <- build_lineage("combined", "tileparity", args$combined_name, list(),
                                    unlist(parents_by_sample[included], use.names = FALSE), args$tileparity_version)
            metadata(combined)$tileparity <- list(
                version = args$tileparity_version, samples = included,
                lineage = as.character(toJSON(merged, auto_unbox = TRUE, null = "null", na = "null")))
            saveRDS(combined, out)
            write_lineage(build_lineage("spe_combined", args$process, args$combined_name, list(spe = out), leaves,
                                        args$tileparity_version),
                          paste0(args$combined_name, ".spe.lineage.json"))
        }
    }
}

main()
