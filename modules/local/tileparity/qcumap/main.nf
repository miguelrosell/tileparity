// Optional QC (--qc_umap) - interactive 3-D UMAP of every feature view, one HTML page per scale.
process TILEPARITY_QCUMAP {
    tag "${meta.id}"
    label 'process_medium'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(files)

    output:
    tuple val(meta), path("${prefix}.qc_umap.{html,tsv.gz}"), emit: html
    tuple val(meta), path("${prefix}.qc_umap.lineage.json") , emit: lineage
    path "versions.yml"                                     , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    export NUMBA_CACHE_DIR=\${PWD}/.numba_cache
    export MPLCONFIGDIR=\${PWD}/.mplconfig
    tp_qc_umap.py \\
        --input-dir . \\
        --name ${prefix} \\
        --process ${task.process} \\
        ${args}

    tp_versions.py "${task.process}" umap-learn numba scikit-learn plotly > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.qc_umap.html ${prefix}.qc_umap.tsv.gz
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.qc_umap.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
