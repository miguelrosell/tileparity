// Process 2b - LBP histograms per tile with fastLBP's LBP kernel (+ H/E intensity statistics).
process FASTLBP_EXTRACT {
    tag "${meta.id}"
    label 'process_high'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(channels), path(tiles)

    output:
    tuple val(meta), path("${prefix}.lbp.h5"), emit: features
    path "versions.yml"                       , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    tp_extract_lbp.py \\
        --channels-image ${channels} \\
        --tiles ${tiles} \\
        --prefix ${prefix} \\
        --ncpus ${task.cpus} \\
        ${args}

    tp_versions.py "${task.process}" fastlbp numpy pyvips > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.lbp.h5
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
