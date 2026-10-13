// Process 3 - expression per tile, Y = X A, using the unit -> tile assignment of Process 1.
process TILEPARITY_AGGREGATEST {
    tag "${meta.id}"
    label 'process_medium'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(tiles), path(assignment), path(spatial)

    output:
    tuple val(meta), path("${prefix}.counts.h5")         , emit: counts
    tuple val(meta), path("${prefix}.counts_summary.tsv"), emit: summary
    tuple val(meta), path("${prefix}.aggregate.json")    , emit: json
    path "versions.yml"                                   , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    def spatial_args = spatial ? "--spatial ${spatial} --spatial-format ${meta.spatial_format}" : ''
    """
    tp_aggregate_st.py \\
        --tiles ${tiles} \\
        --assignment ${assignment} \\
        --prefix ${prefix} \\
        ${spatial_args} \\
        ${args}

    tp_versions.py "${task.process}" scipy h5py anndata > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.counts.h5 ${prefix}.counts_summary.tsv
    echo '{}' > ${prefix}.aggregate.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
