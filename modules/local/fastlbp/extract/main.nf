// Process 2b - LBP histograms per tile with fastLBP's LBP kernel (+ H/E intensity statistics,
// optional central-pixel block and the irregular-pattern QC figure).
process FASTLBP_EXTRACT {
    tag "${meta.id}"
    label 'process_high'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(channels), path(lineage), path(tiles)

    output:
    tuple val(meta), path("${prefix}.lbp.h5"), path("${prefix}.lbp.lineage.json"), emit: features
    tuple val(meta), path("${prefix}.lbp_qc.{png,tsv}")                          , emit: qc
    path "versions.yml"                                                          , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    tp_extract_lbp.py \\
        --channels-image ${channels} \\
        --tiles ${tiles} \\
        --prefix ${prefix} \\
        --ncpus ${task.cpus} \\
        --process ${task.process} \\
        --lineage-parent ${lineage} \\
        ${args}

    tp_versions.py "${task.process}" fastlbp numpy pyvips > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.lbp.h5 ${prefix}.lbp_qc.png ${prefix}.lbp_qc.tsv
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.lbp.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
