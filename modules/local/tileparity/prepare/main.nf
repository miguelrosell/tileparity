// Process 1 - common preparation. The only process that makes geometric decisions:
// pixel size, tissue mask, tile grid and the assignment of spatial units to tiles.
process TILEPARITY_PREPARE {
    tag "${meta.id}"
    label 'process_medium'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(image), path(spatial), path(mask)

    output:
    tuple val(meta), path("${prefix}.resampled.tif")      , emit: image       , optional: true
    tuple val(meta), path("${prefix}.tiles.tsv.gz")       , emit: tiles       , optional: true
    tuple val(meta), path("${prefix}.assignment.tsv.gz")  , emit: assignment  , optional: true
    tuple val(meta), path("${prefix}.thumbnail.png")      , emit: thumbnail   , optional: true
    tuple val(meta), path("${prefix}.{mask,overlay,tissue_thresholds}.png"), emit: qc_images, optional: true
    tuple val(meta), path("${prefix}.mask_compare.{png,tsv}"), emit: mask_compare, optional: true
    tuple val(meta), path("${prefix}.prepare.json")       , emit: json
    tuple val(meta), path("${prefix}.prepare.lineage.json"), emit: lineage
    path "versions.yml"                                    , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    def spatial_args = spatial ? "--spatial ${spatial} --spatial-format ${meta.spatial_format}" : ''
    def mask_arg = mask ? "--mask ${mask}" : ''
    def tiling_arg = meta.tiling ?: 'auto'
    def mpp_arg = meta.mpp ? "--mpp-native ${meta.mpp}" : ''
    def meta_args = (meta.metadata ?: [:]).collect { k, v -> "--meta '${k}=${v}'" }.join(' ')
    """
    export VIPS_CONCURRENCY=${task.cpus}
    tp_prepare.py \\
        --image ${image} \\
        --sample ${meta.sample} \\
        --mpp-target ${meta.scale_mpp} \\
        --tiling ${tiling_arg} \\
        --prefix ${prefix} \\
        --process ${task.process} \\
        ${mpp_arg} \\
        ${spatial_args} \\
        ${mask_arg} \\
        ${meta_args} \\
        ${args}

    tp_versions.py "${task.process}" pyvips libvips numpy scikit-image pandas pillow > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.resampled.tif ${prefix}.thumbnail.png ${prefix}.mask.png ${prefix}.overlay.png ${prefix}.tissue_thresholds.png
    printf 'tile_id\\textract\\tkeep\\n' | gzip > ${prefix}.tiles.tsv.gz
    printf 'barcode\\ttile_id\\tweight\\n' | gzip > ${prefix}.assignment.tsv.gz
    echo '{"status": "ok"}' > ${prefix}.prepare.json
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.prepare.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
