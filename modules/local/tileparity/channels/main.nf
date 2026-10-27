// Process 1b - the input images of both extractors, pixel by pixel from the resampled RGB
// image of Process 1: H and E (colour deconvolution with the slide's own stain vectors), the
// R, G, B copies for fastLBP's 'rgb' set and the image rebuilt from H and E only ('rgb_he',
// for UNI2-h and fastLBP), with a QC figure and table of what 'rgb_he' removes.
process TILEPARITY_CHANNELS {
    tag "${meta.id}"
    label 'process_medium'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(image), path(tiles), path(lineage)

    output:
    tuple val(meta), path("${prefix}.channels.tif"), path("${prefix}.channels.json"), path("${prefix}.channels.lineage.json"), emit: channels
    tuple val(meta), path("${prefix}.rgb_he_qc.{png,tsv}"), optional: true, emit: qc
    path "versions.yml"                                                , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    export VIPS_CONCURRENCY=${task.cpus}
    tp_channels.py \\
        --image ${image} \\
        --tiles ${tiles} \\
        --prefix ${prefix} \\
        --process ${task.process} \\
        --lineage-parent ${lineage} \\
        ${args}

    tp_versions.py "${task.process}" pyvips libvips scikit-image > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.channels.tif ${prefix}.rgb_he_qc.png ${prefix}.rgb_he_qc.tsv
    echo '{}' > ${prefix}.channels.json
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.channels.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
