// The tileparity subworkflow: one preparation, two feature extractors on the same tiles,
// expression per tile, and export.
//
//   Input1 --[1 PREPARE]--> Output1 = Input2 (resampled image, tile table, unit->tile assignment)
//                              |-- [1a+2a UNI2H_EMBED] ----------------> Output2a (tiles x 1536)
//                              |-- [1b LBPCHANNELS] -> [2b FASTLBP_EXTRACT] -> Output2b (tiles x d)
//                              |-- [3 AGGREGATEST] ------------------> Output3 (genes x tiles)
//                              '-- [4 EXPORTSPE / EXPORTANNDATA] ----> SpatialExperiment, AnnData
//
// Only PREPARE makes geometric decisions; every later step reads its tile table.

include { TILEPARITY_PREPARE       } from '../../../modules/local/tileparity/prepare/main'
include { TILEPARITY_LBPCHANNELS   } from '../../../modules/local/tileparity/lbpchannels/main'
include { FASTLBP_EXTRACT          } from '../../../modules/local/fastlbp/extract/main'
include { UNI2H_DOWNLOAD           } from '../../../modules/local/uni2h/download/main'
include { UNI2H_EMBED              } from '../../../modules/local/uni2h/embed/main'
include { TILEPARITY_AGGREGATEST   } from '../../../modules/local/tileparity/aggregatest/main'
include { TILEPARITY_EXPORTANNDATA } from '../../../modules/local/tileparity/exportanndata/main'
include { TILEPARITY_EXPORTSPE     } from '../../../modules/local/tileparity/exportspe/main'

workflow TILEPARITY {
    take:
    ch_input   // channel: [ meta, image, spatial (path or []) ] - one item per sample x scale

    main:
    ch_versions = Channel.empty()

    // ---- Process 1: common preparation ---------------------------------------------------
    TILEPARITY_PREPARE(ch_input)
    ch_versions = ch_versions.mix(TILEPARITY_PREPARE.out.versions.first())

    TILEPARITY_PREPARE.out.json
        .map { meta, json -> [meta, new groovy.json.JsonSlurper().parse(json.toFile())] }
        .filter { meta, report -> report.status == 'skipped' }
        .subscribe { meta, report -> log.warn("${meta.id}: scale skipped - ${report.reason}") }

    ch_image = TILEPARITY_PREPARE.out.image
    ch_tiles = TILEPARITY_PREPARE.out.tiles

    // ---- Processes 1b + 2b: fastLBP branch -------------------------------------------------
    if (params.skip_lbp) {
        ch_lbp = ch_tiles.map { meta, tiles -> [meta, []] }
    } else {
        TILEPARITY_LBPCHANNELS(ch_image)
        FASTLBP_EXTRACT(TILEPARITY_LBPCHANNELS.out.channels.join(ch_tiles))
        ch_lbp = FASTLBP_EXTRACT.out.features
        ch_versions = ch_versions
            .mix(TILEPARITY_LBPCHANNELS.out.versions.first())
            .mix(FASTLBP_EXTRACT.out.versions.first())
    }

    // ---- Processes 1a + 2a: UNI2-h branch --------------------------------------------------
    if (params.skip_uni2h) {
        ch_uni = ch_tiles.map { meta, tiles -> [meta, []] }
    } else {
        if (params.uni2h_test_mode) {
            ch_weights = Channel.value([])
        } else if (params.uni2h_weights) {
            ch_weights = Channel.value(file(params.uni2h_weights, checkIfExists: true))
        } else {
            UNI2H_DOWNLOAD()
            ch_weights = UNI2H_DOWNLOAD.out.weights.first()
            ch_versions = ch_versions.mix(UNI2H_DOWNLOAD.out.versions)
        }
        UNI2H_EMBED(ch_image.join(ch_tiles), ch_weights)
        ch_uni = UNI2H_EMBED.out.features
        ch_versions = ch_versions.mix(UNI2H_EMBED.out.versions.first())
    }

    // ---- Process 3: expression per tile ------------------------------------------------------
    ch_spatial = ch_input.map { meta, image, spatial -> [meta, spatial] }
    TILEPARITY_AGGREGATEST(ch_tiles.join(TILEPARITY_PREPARE.out.assignment).join(ch_spatial))
    ch_versions = ch_versions.mix(TILEPARITY_AGGREGATEST.out.versions.first())

    // ---- Process 4: export, one task per scale with all samples ------------------------------
    ch_bundle = ch_tiles
        .join(TILEPARITY_PREPARE.out.json)
        .join(TILEPARITY_PREPARE.out.thumbnail)
        .join(ch_uni)
        .join(ch_lbp)
        .join(TILEPARITY_AGGREGATEST.out.counts)
        .map { meta, tiles, json, thumb, uni, lbp, counts ->
            [meta.scale_tag, [tiles, json, thumb, uni, lbp, counts].flatten()]
        }
        .groupTuple()
        .map { tag, files -> [[id: "combined_mpp${tag}", scale_tag: tag], files.flatten()] }

    ch_h5ad = Channel.empty()
    if (!params.skip_anndata) {
        TILEPARITY_EXPORTANNDATA(ch_bundle)
        ch_h5ad = TILEPARITY_EXPORTANNDATA.out.h5ad
        ch_versions = ch_versions.mix(TILEPARITY_EXPORTANNDATA.out.versions.first())
    }
    ch_rds = Channel.empty()
    if (!params.skip_spe) {
        TILEPARITY_EXPORTSPE(ch_bundle)
        ch_rds = TILEPARITY_EXPORTSPE.out.rds
        ch_versions = ch_versions.mix(TILEPARITY_EXPORTSPE.out.versions.first())
    }

    emit:
    tiles      = ch_tiles                          // [ meta, tiles.tsv.gz ]
    uni2h      = ch_uni                            // [ meta, uni2h.h5 ]
    lbp        = ch_lbp                            // [ meta, lbp.h5 ]
    counts     = TILEPARITY_AGGREGATEST.out.counts // [ meta, counts.h5 ]
    h5ad       = ch_h5ad                           // [ meta, [*.h5ad] ]
    rds        = ch_rds                            // [ meta, [*.rds] ]
    versions   = ch_versions
}
