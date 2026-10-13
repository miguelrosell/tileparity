// Process 4 (R) - one SpatialExperiment (.rds) per sample and one combined per scale.
process TILEPARITY_EXPORTSPE {
    tag "${meta.id}"
    label 'process_medium'

    conda "${projectDir}/containers/r/environment.yml"
    container "${params.container_r}"

    input:
    tuple val(meta), path(files)

    output:
    tuple val(meta), path("*.rds"), emit: rds
    path "versions.yml"           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    tp_export_spe.R \\
        --input-dir . \\
        --combined-name ${prefix} \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        r-base: \$(Rscript -e 'cat(as.character(getRversion()))')
        SpatialExperiment: \$(Rscript -e 'cat(as.character(packageVersion("SpatialExperiment")))')
        rhdf5: \$(Rscript -e 'cat(as.character(packageVersion("rhdf5")))')
    END_VERSIONS
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.rds
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
