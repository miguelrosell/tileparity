// Optional (--cluster) - every feature view of one slide and scale clustered on its own
// (Leiden and k-means at the same numbers of clusters), and how much the views agree
// (adjusted Rand index, NMI, spatial null), with figures.
process TILEPARITY_CLUSTER {
    tag "${meta.id}"
    label 'process_medium'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(files)

    output:
    tuple val(meta), path("${prefix}.clusters.tsv.gz"), path("${prefix}.cluster.lineage.json"), emit: labels
    tuple val(meta), path("${prefix}.cluster_agreement.tsv"), path("${prefix}.cluster.json")     , emit: agreement
    tuple val(meta), path("${prefix}.cluster_*.png")                       , optional: true       , emit: figures
    path "versions.yml"                                                                        , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    export NUMBA_CACHE_DIR=\${PWD}/.numba_cache
    export MPLCONFIGDIR=\${PWD}/.mplconfig
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus} NUMBA_NUM_THREADS=${task.cpus}
    tp_cluster.py \\
        --input-dir . \\
        --prefix ${prefix} \\
        --process ${task.process} \\
        ${args}

    tp_versions.py "${task.process}" scikit-learn igraph umap-learn > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    printf 'tile_id\\tin_keep\\tin_extended\\n' | gzip > ${prefix}.clusters.tsv.gz
    touch ${prefix}.cluster_agreement.tsv ${prefix}.cluster_agreement_keep.png ${prefix}.cluster_maps_keep.png
    echo '{}' > ${prefix}.cluster.json
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.cluster.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
