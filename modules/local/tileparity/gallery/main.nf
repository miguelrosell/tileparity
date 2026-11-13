// Optional (--cluster with --cluster_gallery, on by default) - for every view of one slide
// and scale, the tiles that show what it groups: the most typical tiles of every cluster at
// the k shown and the tiles at both ends of the first principal components, cut from the
// image the extractors read.
process TILEPARITY_GALLERY {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(image), path(tiles), path(prepare_lineage), path(representatives), path(cluster_json), path(cluster_lineage)

    output:
    tuple val(meta), path("${prefix}.gallery_*.png"), optional: true, emit: figures
    tuple val(meta), path("${prefix}.gallery.lineage.json")         , emit: lineage
    path "versions.yml"                                            , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    tp_gallery.py \\
        --input-dir . \\
        --prefix ${prefix} \\
        --process ${task.process} \\
        ${args}

    tp_versions.py "${task.process}" pyvips pillow > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.gallery_UNI2H_RGB.png
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.gallery.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
