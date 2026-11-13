// With --cluster (--cohort_summary, on by default) - one task per scale: every slide's
// clustering, redundancy and relevance results gathered into a long table, a table of key
// numbers per slide and view, cohort figures, comparisons between the groups of a
// samplesheet column, and one HTML page.
process TILEPARITY_COHORT {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/containers/python/environment.yml"
    container "${params.container_python}"

    input:
    tuple val(meta), path(files), val(samples_tsv)

    output:
    tuple val(meta), path("${meta.id}.{long.tsv.gz,key.tsv,groups.tsv,genes.tsv.gz}"), optional: true, emit: tables
    tuple val(meta), path("${meta.id}.*.png")                                         , optional: true, emit: figures
    tuple val(meta), path("${meta.id}.report.html")                                   , emit: report
    tuple val(meta), path("${meta.id}.cohort.lineage.json")                          , emit: lineage
    path "versions.yml"                                                               , emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    cat > samples.tsv <<'TILEPARITY_SAMPLES'
${samples_tsv}
TILEPARITY_SAMPLES
    tp_cohort.py \\
        --input-dir . \\
        --prefix ${meta.id} \\
        --samples samples.tsv \\
        --process ${task.process} \\
        ${args}

    tp_versions.py "${task.process}" scipy pillow > versions.yml
    """

    stub:
    """
    touch ${meta.id}.key.tsv ${meta.id}.report.html
    echo '{"schema": "tileparity-lineage/1"}' > ${meta.id}.cohort.lineage.json
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
