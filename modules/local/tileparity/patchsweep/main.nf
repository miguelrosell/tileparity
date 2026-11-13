// Optional (--patch_sweep, Visium HD slides only) - what LBP knows about the expression as a
// function of the patch size: LBP codes counted once per 8 um cell, summed into patches of
// 16 to 224 um, expression from the 8 um bins summed the same way, with a split-half noise
// ceiling (tileparity_lib/patchsweep.py).
process TILEPARITY_PATCH_SWEEP {
    tag "${meta.id}"
    label 'process_high'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(channels), path(channels_json), path(channels_lineage), path(tiles), path(prepare_json), path(spatial)

    output:
    tuple val(meta), path("${prefix}.patch_sweep{.tsv,_radius.tsv,.json}"), emit: tables
    tuple val(meta), path("${prefix}.patch_sweep.png")                     , optional: true, emit: figure
    tuple val(meta), path("${prefix}.patch_sweep.lineage.json")            , emit: lineage
    path "versions.yml"                                                    , emit: versions

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    export NUMBA_CACHE_DIR=\${PWD}/.numba_cache
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    tp_patch_sweep.py \\
        --channels-image ${channels} \\
        --channels-json ${channels_json} \\
        --tiles ${tiles} \\
        --prepare-json ${prepare_json} \\
        --spatial ${spatial} \\
        --prefix ${prefix} \\
        --ncpus ${task.cpus} \\
        --process ${task.process} \\
        --lineage-parent ${channels_lineage} \\
        ${args}

    tp_versions.py "${task.process}" fastlbp scikit-learn > versions.yml
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.patch_sweep.tsv ${prefix}.patch_sweep_radius.tsv
    echo '{}' > ${prefix}.patch_sweep.json
    echo '{"schema": "tileparity-lineage/1"}' > ${prefix}.patch_sweep.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
