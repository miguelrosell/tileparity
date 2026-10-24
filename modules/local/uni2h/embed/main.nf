// Processes 1a + 2a - UNI2-h adapter (cut 224 px tiles, ImageNet normalisation, no resize)
// and embedding (1536-d class token per tile).
process UNI2H_EMBED {
    tag "${meta.id}"
    label 'process_gpu'

    conda "${projectDir}/containers/uni2h/environment.yml"
    container "${params.container_uni2h}"

    input:
    tuple val(meta), path(image), path(tiles), path(lineage)
    path weights

    output:
    tuple val(meta), path("${prefix}.uni2h.h5"), path("${prefix}.uni2h.lineage.json"), emit: features
    path "versions.yml"                                                              , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    def weights_arg = weights ? "--weights ${weights}" : ''
    def workers = Math.max(0, (task.cpus as int) - 1)
    """
    export VIPS_CONCURRENCY=1
    export HF_HOME=\${PWD}/.hf_home
    tp_embed_uni2h.py \\
        --image ${image} \\
        --tiles ${tiles} \\
        --prefix ${prefix} \\
        --num-workers ${workers} \\
        --process ${task.process} \\
        --lineage-parent ${lineage} \\
        ${weights_arg} \\
        ${args}

    tp_versions.py "${task.process}" torch timm pyvips > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.uni2h.h5
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.uni2h.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
