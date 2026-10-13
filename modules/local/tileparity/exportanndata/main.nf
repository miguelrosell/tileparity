// Process 4 (Python) - one AnnData (.h5ad) per sample and one combined per scale.
process TILEPARITY_EXPORTANNDATA {
    tag "${meta.id}"
    label 'process_medium'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(files)

    output:
    tuple val(meta), path("*.h5ad"), emit: h5ad
    path "versions.yml"            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    tp_export_anndata.py \\
        --input-dir . \\
        --combined-name ${prefix} \\
        ${args}

    tp_versions.py "${task.process}" anndata > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.h5ad
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
