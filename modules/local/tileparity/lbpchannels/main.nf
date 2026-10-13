// Process 1b - LBP adapter: H, E (colour deconvolution) and grey channels, pixel by pixel
// from the same resampled RGB image that UNI2-h reads.
process TILEPARITY_LBPCHANNELS {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(image)

    output:
    tuple val(meta), path("${prefix}.channels.tif") , emit: channels
    tuple val(meta), path("${prefix}.channels.json"), emit: json
    path "versions.yml"                              , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    export VIPS_CONCURRENCY=${task.cpus}
    tp_lbp_channels.py \\
        --image ${image} \\
        --prefix ${prefix} \\
        ${args}

    tp_versions.py "${task.process}" pyvips libvips scikit-image > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.channels.tif
    echo '{}' > ${prefix}.channels.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
