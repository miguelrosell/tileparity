// Samplesheet parsing and parameter checks. Plain Nextflow/Groovy, no plugins, so the
// pipeline also starts on machines without internet access.

workflow PIPELINE_INITIALISATION {
    take:
    input   // path to the samplesheet (CSV)
    scales  // comma-separated target pixel sizes in µm/px, e.g. "0.2857,0.5,1.0"

    main:
    if (!input) {
        error "Missing --input: a samplesheet CSV is required (see docs/usage.md)."
    }
    validateParams()
    def sheet = file(input, checkIfExists: true)
    def samples = readSamplesheet(sheet)
    def scaleList = parseScales(scales)
    log.info summaryText(samples.size(), scaleList)

    ch_inputs = Channel.fromList(samples)
        .combine(Channel.fromList(scaleList))
        .map { meta, image, spatial, mpp ->
            def tag = scaleTag(mpp)
            [meta + [id: "${meta.sample}_mpp${tag}", scale_mpp: mpp, scale_tag: tag], image, spatial]
        }

    emit:
    inputs = ch_inputs
}

def scaleTag(Double mpp) {
    return String.format(Locale.ROOT, '%.4f', mpp)
}

// Checks that are cheap here and expensive to discover inside a task
def validateParams() {
    if (!params.skip_uni2h && (params.tile_px as int) != 224) {
        error "--tile_px ${params.tile_px}: UNI2-h works on 224 x 224 px tiles. Keep --tile_px 224 or add --skip_uni2h."
    }
    if (!((params.hd_bin_um as int) in [2, 8, 16])) {
        error "--hd_bin_um ${params.hd_bin_um}: Space Ranger writes bins of 2, 8 or 16 µm"
    }
    def channels = params.lbp_channels.toString().tokenize(',')*.trim().findAll { it }
    if (!params.skip_lbp && (!channels || (channels - ['H', 'E', 'grey']))) {
        error "--lbp_channels '${params.lbp_channels}': use a comma-separated subset of H,E,grey"
    }
    if (!((params.min_tissue as Double) >= 0 && (params.min_tissue as Double) <= 1)) {
        error "--min_tissue must be between 0 and 1"
    }
    if (!(params.uni2h_precision in ['fp32', 'fp16', 'bf16'])) {
        error "--uni2h_precision must be fp32, fp16 or bf16"
    }
    if (params.skip_uni2h && params.skip_lbp) {
        log.warn "--skip_uni2h and --skip_lbp are both set: only tiles and expression per tile will be produced"
    }
    if (params.uni2h_test_mode && !params.skip_uni2h) {
        log.warn "UNI2-h TEST MODE: embeddings come from a random 1-block network and are meaningless"
    }
}

// Columns of the tile table; samplesheet metadata columns must not reuse these names
def reservedColumns() {
    return ['tile_id', 'sample_id', 'scale_mpp', 'tile_px', 'tile_um', 'unit_type', 'row', 'col', 'barcode',
            'x0_px', 'y0_px', 'cx_px', 'cy_px', 'cx_fullres', 'cy_fullres', 'cx_um', 'cy_um', 'inside_image',
            'tissue_fraction', 'st_coverage', 'n_st_units', 'lapvar', 'pen_fraction', 'dark_fraction',
            'blur_flag', 'pen_flag', 'dark_flag', 'keep', 'qc_fail']
}

def parseScales(String text) {
    def values = text.toString().tokenize(',').collect { it.trim() }.findAll { it }
    if (!values) {
        error "--scales_mpp must list at least one pixel size, e.g. '0.5'"
    }
    return values.collect { v ->
        if (!(v ==~ /[0-9]*\.?[0-9]+/) || (v as Double) <= 0) {
            error "--scales_mpp: '${v}' is not a positive number"
        }
        v as Double
    }.unique()
}

def readSamplesheet(sheet) {
    def known = ['sample', 'image', 'spatial', 'spatial_format', 'tiling', 'mpp']
    def formats = ['spaceranger_hd', 'spaceranger', 'hest']
    def tilings = ['auto', 'grid', 'spot']
    def rows = sheet.splitCsv(header: true, strip: true)
    if (!rows) {
        error "Samplesheet ${sheet} has no rows"
    }
    def missing = ['sample', 'image'] - rows[0].keySet()
    if (missing) {
        error "Samplesheet ${sheet} lacks required column(s): ${missing.join(', ')}"
    }
    def clash = rows[0].keySet().intersect(reservedColumns())
    if (clash) {
        error "Samplesheet ${sheet}: column(s) ${clash.join(', ')} would overwrite columns of the tile table; please rename them"
    }
    def base = sheet.parent
    def home = System.getProperty('user.home')
    def resolve = { String p ->
        if (p.startsWith('~/')) {
            return file(home + p.substring(1))
        }
        return (p.startsWith('/') || p.contains('://')) ? file(p) : file(base.resolve(p).toString())
    }
    def seen = [] as Set
    return rows.withIndex().collect { row, i ->
        def line = i + 2
        def sample = row.sample ?: ''
        if (!(sample ==~ /[A-Za-z0-9._-]+/)) {
            error "Samplesheet line ${line}: sample '${sample}' must be non-empty and use only letters, digits, '.', '_' or '-'"
        }
        if (!seen.add(sample)) {
            error "Samplesheet line ${line}: sample '${sample}' is not unique"
        }
        def image = resolve(row.image ?: '')
        if (!row.image || !image.exists()) {
            error "Samplesheet line ${line}: image '${row.image}' not found"
        }
        def spatial = []
        def fmt = (row.spatial_format ?: '').toLowerCase()
        if (row.spatial) {
            spatial = resolve(row.spatial)
            if (!spatial.exists()) {
                error "Samplesheet line ${line}: spatial '${row.spatial}' not found"
            }
            if (!(fmt in formats)) {
                error "Samplesheet line ${line}: spatial_format must be one of ${formats.join(', ')}"
            }
        } else if (fmt) {
            error "Samplesheet line ${line}: spatial_format given without a spatial path"
        }
        def tiling = (row.tiling ?: 'auto').toLowerCase()
        if (!(tiling in tilings)) {
            error "Samplesheet line ${line}: tiling must be one of ${tilings.join(', ')}"
        }
        if (tiling == 'spot' && !(fmt in ['spaceranger', 'hest'])) {
            error "Samplesheet line ${line}: tiling 'spot' needs Visium spots (spatial_format spaceranger or hest); Visium HD uses 'grid'"
        }
        Double mpp = null
        if (row.mpp) {
            if (!(row.mpp ==~ /[0-9]*\.?[0-9]+/) || (row.mpp as Double) <= 0) {
                error "Samplesheet line ${line}: mpp '${row.mpp}' is not a positive number"
            }
            mpp = row.mpp as Double
        }
        // every other column is sample metadata, copied into the tile tables and exports
        // (single quotes are removed because the values are passed as quoted command-line arguments)
        def metadata = row.findAll { k, v -> !(k in known) }.collectEntries { k, v ->
            [(k.replace("'", '')): (v ?: '').replace("'", '')]
        }
        def meta = [sample: sample, spatial_format: fmt ?: null, tiling: tiling, mpp: mpp, metadata: metadata]
        [meta, image, spatial]
    }
}

def summaryText(int nSamples, List scales) {
    return """
    tileparity ${workflow.manifest.version}
    --------------------------------------------------
    samplesheet : ${params.input}
    samples     : ${nSamples}
    scales      : ${scales.collect { "${it} µm/px (tile ${String.format(Locale.ROOT, '%.1f', it * params.tile_px)} µm)" }.join(', ')}
    extractors  : ${[params.skip_uni2h ? null : 'UNI2-h' + (params.uni2h_test_mode ? ' [TEST MODE]' : ''), params.skip_lbp ? null : 'fastLBP'].findAll().join(', ')}
    exports     : ${[params.skip_spe ? null : 'SpatialExperiment', params.skip_anndata ? null : 'AnnData'].findAll().join(', ')}
    gpu         : ${params.use_gpu}
    outdir      : ${params.outdir}
    --------------------------------------------------
    """.stripIndent()
}
