#!/usr/bin/env nextflow
/*
 * tileparity - harmonised H&E tiling so that UNI2-h embeddings, fastLBP texture
 * histograms and spatial-transcriptomics counts all describe the same tiles.
 *
 * https://github.com/miguelrosell/tileparity
 */

include { TILEPARITY_PIPELINE } from './workflows/tileparity'

workflow {
    TILEPARITY_PIPELINE()

    // Registered inside the workflow block and written as a function call, so that it is
    // valid in both the classic and the strict (Nextflow >= 25.10) syntax.
    workflow.onComplete { completionMessage() }
}

def completionMessage() {
    if (workflow.success) {
        log.info("tileparity finished: results in ${params.outdir}")
    } else {
        log.error("tileparity failed: see ${workflow.launchDir}/.nextflow.log and the work directory of the failing task")
    }
}
