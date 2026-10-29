// Samplesheet parsing and parameter checks. Plain Nextflow/Groovy, no plugins, so the
// pipeline also starts on machines without internet access.

workflow PIPELINE_INITIALISATION {
    take:
    samplesheet  // path to the samplesheet (CSV), from --input
    scales       // comma-separated target pixel sizes in µm/px, e.g. "0.2857,0.5,1.0"

    main:
    if (!samplesheet) {
        error("Missing --input: a samplesheet CSV is required (see docs/usage.md).")
    }
    validateParams()
    def sheet = file(samplesheet, checkIfExists: true)
    def samples = readSamplesheet(sheet)
    def scaleList = parseScales(scales)
    log.info(summaryText(samples.size(), scaleList))

    ch_inputs = channel.fromList(samples)
        .combine(channel.fromList(scaleList))
        .map { meta, image, spatial, mask, mpp ->
            def tag = scaleTag(mpp)
            [meta + [id: "${meta.sample}_mpp${tag}", scale_mpp: mpp, scale_tag: tag], image, spatial, mask]
        }

    emit:
    inputs = ch_inputs
}

def scaleTag(Double mpp) {
    return String.format(Locale.ROOT, '%.4f', mpp)
}

def numberList(String name, value) {
    def values = value.toString().tokenize(',').collect { v -> v.trim() }.findAll { v -> v }
    values.each { v ->
        if (!(v ==~ /[0-9]*\.?[0-9]+/)) {
            error("--${name}: '${v}' is not a number")
        }
    }
    return values.collect { v -> v as Double }
}

// Checks that are cheap here and expensive to discover inside a task
def validateParams() {
    if (!params.skip_uni2h && (params.tile_px as int) != 224) {
        error("--tile_px ${params.tile_px}: UNI2-h works on 224 x 224 px tiles. Keep --tile_px 224 or add --skip_uni2h.")
    }
    if (!((params.hd_bin_um as int) in [2, 8, 16])) {
        error("--hd_bin_um ${params.hd_bin_um}: Space Ranger writes bins of 2, 8 or 16 µm")
    }
    def sets = inputSets(params.lbp_channels)
    if (!params.skip_lbp && 'grey' in sets) {
        error("--lbp_channels '${params.lbp_channels}': the grey channel set was removed in tileparity 0.3; " +
              "use rgb (R, G, B of the slide) and rgb_he (R, G, B of the slide rebuilt from H and E only)")
    }
    if (!params.skip_lbp && (!sets || (sets - ['rgb', 'rgb_he', 'he']))) {
        error("--lbp_channels '${params.lbp_channels}': use a comma-separated subset of the input sets rgb, rgb_he, he " +
              "('rgb' = R, G, B of the slide; 'rgb_he' = R, G, B of the slide rebuilt from H and E only; " +
              "'he' = the H and E amounts)")
    }
    def inputs = inputSets(params.uni2h_inputs)
    if (!params.skip_uni2h && (!inputs || (inputs - ['rgb', 'rgb_he']))) {
        error("--uni2h_inputs '${params.uni2h_inputs}': use a comma-separated subset of the input sets rgb, rgb_he " +
              "('rgb' = the slide; 'rgb_he' = the slide rebuilt from H and E only)")
    }
    if (!(params.stain_vectors in ['slide', 'standard'])) {
        error("--stain_vectors must be slide (estimated on each slide) or standard (Ruifrok & Johnston's fixed vectors)")
    }
    def minTissue = params.min_tissue as Double
    if (!(minTissue >= 0 && minTissue <= 1)) {
        error("--min_tissue must be between 0 and 1")
    }
    if (params.min_tissue_extract != null) {
        def extract = params.min_tissue_extract as Double
        if (!(extract >= 0 && extract <= minTissue)) {
            error("--min_tissue_extract (${extract}) must be between 0 and --min_tissue (${minTissue})")
        }
    }
    numberList('qc_tissue_thresholds', params.qc_tissue_thresholds).each { v ->
        if (v < 0 || v > 1) {
            error("--qc_tissue_thresholds: ${v} is not between 0 and 1")
        }
    }
    numberList('mask_compare_mpp', params.mask_compare_mpp)
    if (!(params.mask_method in ['saturation', 'grey', 'od'])) {
        error("--mask_method must be saturation, grey or od")
    }
    if (!(params.visium_tiling in ['grid', 'spot'])) {
        error("--visium_tiling must be grid or spot")
    }
    if (!(params.visium_split in ['multinomial', 'fraction'])) {
        error("--visium_split must be multinomial or fraction")
    }
    if (!(params.uni2h_precision in ['fp32', 'fp16', 'bf16'])) {
        error("--uni2h_precision must be fp32, fp16 or bf16")
    }
    if (params.cluster) {
        validateCluster()
    }
    if (params.skip_uni2h && params.skip_lbp) {
        log.warn("--skip_uni2h and --skip_lbp are both set: only tiles and expression per tile will be produced")
    }
    if (params.uni2h_test_mode && !params.skip_uni2h) {
        log.warn("UNI2-h TEST MODE: embeddings come from a random 1-block network and are meaningless")
    }
}

def validateCluster() {
    listParam('cluster_views', params.cluster_views,
              ['all', 'UNI2H_RGB', 'UNI2H_RGB_HE', 'LBP_RGB', 'LBP_RGB_HE', 'LBP_HE', 'INTENSITY_HE', 'EXPRESSION'])
    listParam('cluster_sets', params.cluster_sets, ['keep', 'extended'])
    listParam('cluster_methods', params.cluster_methods, ['leiden', 'kmeans'])
    def ks = (params.cluster_k ?: '').toString().tokenize(',')*.trim().findAll { v -> v }
    if (!ks || ks.any { v -> !(v ==~ /[0-9]+/) || (v as Integer) < 2 }) {
        error("--cluster_k '${params.cluster_k}': use a comma-separated list of whole numbers of at least 2, e.g. '4,8,12'")
    }
    wholeParam('cluster_k_ref', params.cluster_k_ref, 2)
    wholeParam('cluster_n_pcs', params.cluster_n_pcs, 1)
    wholeParam('cluster_n_neighbors', params.cluster_n_neighbors, 2)
    wholeParam('cluster_n_hvg', params.cluster_n_hvg, 10)
    wholeParam('cluster_null_shifts', params.cluster_null_shifts, 0)
    wholeParam('cluster_seed', params.cluster_seed, 0)
    def floor = params.cluster_min_counts.toString()
    if (!(floor ==~ /[0-9]*\.?[0-9]+/) || (floor as Double) > 1) {
        error("--cluster_min_counts '${params.cluster_min_counts}' must be between 0 and 1 (a fraction of the keep tiles' median)")
    }
    if (!((params.cluster_k_ref as Integer) in ks.collect { v -> v as Integer })) {
        log.warn("--cluster_k_ref ${params.cluster_k_ref} is not in --cluster_k '${params.cluster_k}': the figures use the closest k below it")
    }
}

// A comma-separated subset of `allowed` (or 'all' when allowed)
def listParam(String name, value, List allowed) {
    def items = (value ?: '').toString().tokenize(',')*.trim().findAll { v -> v }
    if (items == ['all'] && 'all' in allowed) {
        return items
    }
    def bad = items - allowed
    if (!items || bad) {
        error("--${name} '${value}': use a comma-separated subset of ${(allowed - ['all']).join(', ')}" +
              ('all' in allowed ? ', or all' : ''))
    }
    return items
}

def wholeParam(String name, value, int min) {
    if (!(value.toString() ==~ /[0-9]+/) || (value as Integer) < min) {
        error("--${name} '${value}' must be a whole number of at least ${min}")
    }
}

def inputSets(value) {
    return (value ?: '').toString().toLowerCase().tokenize(',')*.trim().findAll { s -> s && s != 'none' }
}

// Columns of the tile table; samplesheet metadata columns must not reuse these names
def reservedColumns() {
    return ['tile_id', 'sample_id', 'scale_mpp', 'tile_px', 'tile_um', 'unit_type', 'row', 'col', 'barcode',
            'x0_px', 'y0_px', 'cx_px', 'cy_px', 'cx_fullres', 'cy_fullres', 'cx_um', 'cy_um', 'inside_image',
            'tissue_fraction', 'st_coverage', 'n_st_units', 'st_units_equiv', 'lapvar', 'pen_fraction',
            'dark_fraction', 'blur_flag', 'pen_flag', 'dark_flag', 'extract', 'keep', 'qc_fail']
}

// Samplesheet paths: absolute, URLs, '~/...' or relative to the samplesheet's folder
def resolvePath(String p, base) {
    if (p.startsWith('~/')) {
        return file(System.getProperty('user.home') + p.substring(1))
    }
    return (p.startsWith('/') || p.contains('://')) ? file(p) : file(base.resolve(p).toString())
}

def parseScales(String text) {
    def values = text.toString().tokenize(',').collect { v -> v.trim() }.findAll { v -> v }
    if (!values) {
        error("--scales_mpp must list at least one pixel size, e.g. '0.5'")
    }
    return values.collect { v ->
        if (!(v ==~ /[0-9]*\.?[0-9]+/) || (v as Double) <= 0) {
            error("--scales_mpp: '${v}' is not a positive number")
        }
        v as Double
    }.unique()
}

def readSamplesheet(sheet) {
    def known = ['sample', 'image', 'spatial', 'spatial_format', 'tiling', 'mpp', 'mask']
    def formats = ['spaceranger_hd', 'spaceranger', 'hest']
    def tilings = ['auto', 'grid', 'spot']
    def rows = sheet.splitCsv(header: true, strip: true)
    if (!rows) {
        error("Samplesheet ${sheet} has no rows")
    }
    def missing = ['sample', 'image'] - rows[0].keySet()
    if (missing) {
        error("Samplesheet ${sheet} lacks required column(s): ${missing.join(', ')}")
    }
    def clash = rows[0].keySet().intersect(reservedColumns())
    if (clash) {
        error("Samplesheet ${sheet}: column(s) ${clash.join(', ')} would overwrite columns of the tile table; please rename them")
    }
    def base = sheet.parent
    def seen = [] as Set
    return rows.withIndex().collect { row, i ->
        def line = i + 2
        def sample = row.sample ?: ''
        if (!(sample ==~ /[A-Za-z0-9._-]+/)) {
            error("Samplesheet line ${line}: sample '${sample}' must be non-empty and use only letters, digits, '.', '_' or '-'")
        }
        if (!seen.add(sample)) {
            error("Samplesheet line ${line}: sample '${sample}' is not unique")
        }
        def image = resolvePath(row.image ?: '', base)
        if (!row.image || !image.exists()) {
            error("Samplesheet line ${line}: image '${row.image}' not found")
        }
        def spatial = []
        def fmt = (row.spatial_format ?: '').toLowerCase()
        if (row.spatial) {
            spatial = resolvePath(row.spatial, base)
            if (!spatial.exists()) {
                error("Samplesheet line ${line}: spatial '${row.spatial}' not found")
            }
            if (!(fmt in formats)) {
                error("Samplesheet line ${line}: spatial_format must be one of ${formats.join(', ')}")
            }
        } else if (fmt) {
            error("Samplesheet line ${line}: spatial_format given without a spatial path")
        }
        def tiling = (row.tiling ?: 'auto').toLowerCase()
        if (!(tiling in tilings)) {
            error("Samplesheet line ${line}: tiling must be one of ${tilings.join(', ')}")
        }
        if (tiling == 'spot' && !(fmt in ['spaceranger', 'hest'])) {
            error("Samplesheet line ${line}: tiling 'spot' needs Visium spots (spatial_format spaceranger or hest); Visium HD uses 'grid'")
        }
        def mask = []
        if (row.mask) {
            mask = resolvePath(row.mask, base)
            if (!mask.exists()) {
                error("Samplesheet line ${line}: mask '${row.mask}' not found")
            }
        }
        Double mpp = null
        if (row.mpp) {
            if (!(row.mpp ==~ /[0-9]*\.?[0-9]+/) || (row.mpp as Double) <= 0) {
                error("Samplesheet line ${line}: mpp '${row.mpp}' is not a positive number")
            }
            mpp = row.mpp as Double
        }
        // every other column is sample metadata, copied into the tile tables and exports
        // (single quotes are removed because the values are passed as quoted command-line arguments)
        def metadata = row.findAll { k, _v -> !(k in known) }.collectEntries { k, v ->
            [(k.replace("'", '')): (v ?: '').replace("'", '')]
        }
        def meta = [sample: sample, spatial_format: fmt ?: null, tiling: tiling, mpp: mpp, metadata: metadata]
        [meta, image, spatial, mask]
    }
}

def summaryText(int nSamples, List scales) {
    def extract = params.min_tissue_extract != null ? params.min_tissue_extract : params.min_tissue
    return """
    tileparity ${workflow.manifest.version}
    --------------------------------------------------
    samplesheet : ${params.input}
    samples     : ${nSamples}
    scales      : ${scales.collect { s -> "${s} µm/px (tile ${String.format(Locale.ROOT, '%.1f', s * params.tile_px)} µm)" }.join(', ')}
    tissue      : mask '${params.mask_method}' at ${params.mask_mpp} µm/px; features from >= ${extract}, keep >= ${params.min_tissue}${params.mask_compare ? '; mask comparison on' : ''}
    visium      : ${params.visium_tiling} tiling${params.visium_tiling == 'grid' ? ", spots split by area (${params.visium_split})" : ''}
    extractors  : ${[params.skip_uni2h ? null : "UNI2-h (${params.uni2h_inputs})" + (params.uni2h_test_mode ? ' [TEST MODE]' : ''), params.skip_lbp ? null : "fastLBP (${params.lbp_channels}${params.lbp_center ? ', + centre' : ''})"].findAll().join(', ')}
    stains      : H and E vectors ${params.stain_vectors == 'slide' ? 'estimated on each slide (Macenko)' : 'standard (Ruifrok & Johnston)'}
    exports     : ${[params.skip_spe ? null : 'SpatialExperiment', params.skip_anndata ? null : 'AnnData', params.qc_umap ? '3-D UMAP QC' : null].findAll().join(', ')}
    clustering  : ${params.cluster ? "per slide, ${params.cluster_methods} at k = ${params.cluster_k} (figures at k = ${params.cluster_k_ref}), sets ${params.cluster_sets}" : 'off (--cluster)'}
    gpu         : ${params.use_gpu}
    outdir      : ${params.outdir}
    --------------------------------------------------
    """.stripIndent()
}
