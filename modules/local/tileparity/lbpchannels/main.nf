// Process 1b - LBP adapter: H and E (colour deconvolution) and, per channel set, grey and
// R/G/B channels, pixel by pixel from the same resampled RGB image that UNI2-h reads.
process TILEPARITY_LBPCHANNELS {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(image), path(lineage)

    output:
    tuple val(meta), path("${prefix}.channels.tif"), path("${prefix}.lbpchannels.lineage.json"), emit: channels
    tuple val(meta), path("${prefix}.channels.json"), emit: json
    path "versions.yml"                              , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    export VIPS_CONCURRENCY=${task.cpus}
    tp_lbp_channels.py \\
        --image ${image} \\
        --prefix ${prefix} \\
        --process ${task.process} \\
        --lineage-parent ${lineage} \\
        ${args}

    tp_versions.py "${task.process}" pyvips libvips scikit-image > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.channels.tif
    echo '{}' > ${prefix}.channels.json
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.lbpchannels.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
