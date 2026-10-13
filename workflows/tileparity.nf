include { PIPELINE_INITIALISATION } from '../subworkflows/local/utils_tileparity/main'
include { TILEPARITY              } from '../subworkflows/local/tileparity/main'

workflow TILEPARITY_PIPELINE {
    main:
    PIPELINE_INITIALISATION(params.input, params.scales_mpp)
    TILEPARITY(PIPELINE_INITIALISATION.out.inputs)

    TILEPARITY.out.versions
        .unique()
        .collectFile(name: 'software_versions.yml', storeDir: "${params.outdir}/pipeline_info", sort: true, newLine: true)
}
