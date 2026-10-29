// The tileparity subworkflow: one preparation, two feature extractors on the same tiles and
// the same input images, expression per tile, and export.
//
//   Input1 --[1 PREPARE]--> Output1 = Input2: resampled image (input set rgb), tile table,
//                              |              unit->tile assignment
//                              |-- [1b CHANNELS] --------------------> channels image: H, E, R, G, B and
//                              |                                       the image rebuilt from H and E
//                              |                                       only (input set rgb_he)
//                              |-- [2b FASTLBP_EXTRACT] (channels) --> Output2b (LBP per channel)
//                              |-- [1a+2a UNI2H_EMBED] (rgb, rgb_he) -> Output2a (tiles x 1536 per set)
//                              |-- [3 AGGREGATEST] ------------------> Output3 (genes x tiles)
//                              |-- [CLUSTER] (optional, per slide) --> clusters of every view, ARI between views
//                              |-- [4 EXPORTSPE / EXPORTANNDATA] ----> SpatialExperiment, AnnData
//                              '-- [QCUMAP] (optional) --------------> 3-D UMAP of every view
//
// Only PREPARE makes geometric decisions; every later step reads its tile table. CHANNELS
// computes every input image once, so UNI2-h and fastLBP read exactly the same pixels. Every
// step writes a lineage file that contains the lineage of its inputs, so the export's lineage
// holds the whole route from the original image.

include { TILEPARITY_PREPARE       } from '../../../modules/local/tileparity/prepare/main'
include { TILEPARITY_CHANNELS      } from '../../../modules/local/tileparity/channels/main'
include { FASTLBP_EXTRACT          } from '../../../modules/local/fastlbp/extract/main'
include { UNI2H_DOWNLOAD           } from '../../../modules/local/uni2h/download/main'
include { UNI2H_EMBED              } from '../../../modules/local/uni2h/embed/main'
include { TILEPARITY_AGGREGATEST   } from '../../../modules/local/tileparity/aggregatest/main'
include { TILEPARITY_EXPORTANNDATA } from '../../../modules/local/tileparity/exportanndata/main'
include { TILEPARITY_EXPORTSPE     } from '../../../modules/local/tileparity/exportspe/main'
include { TILEPARITY_QCUMAP        } from '../../../modules/local/tileparity/qcumap/main'
include { TILEPARITY_CLUSTER       } from '../../../modules/local/tileparity/cluster/main'

workflow TILEPARITY {
    take:
    ch_input   // channel: [ meta, image, spatial (path or []), mask (path or []) ] - one item per sample x scale

    main:
    ch_versions = channel.empty()

    // ---- Process 1: common preparation ---------------------------------------------------
    TILEPARITY_PREPARE(ch_input)
    ch_versions = ch_versions.mix(TILEPARITY_PREPARE.out.versions.first())

    TILEPARITY_PREPARE.out.json
        .map { meta, json -> [meta, new groovy.json.JsonSlurper().parse(json.toFile())] }
        .filter { _meta, report -> report.status == 'skipped' }
        .subscribe { meta, report -> log.warn("${meta.id}: scale skipped - ${report.reason}") }

    ch_image   = TILEPARITY_PREPARE.out.image
    ch_tiles   = TILEPARITY_PREPARE.out.tiles
    ch_lineage = TILEPARITY_PREPARE.out.lineage

    // ---- Process 1b: the input images of both extractors -------------------------------------
    def uni2h_sets = params.uni2h_inputs.toString().toLowerCase().tokenize(',')*.trim()
    def need_rgb_he = !params.skip_uni2h && ('rgb_he' in uni2h_sets)
    if (!params.skip_lbp || need_rgb_he) {
        TILEPARITY_CHANNELS(ch_image.join(ch_tiles).join(ch_lineage))
        ch_channels = TILEPARITY_CHANNELS.out.channels   // [ meta, channels.tif, channels.json, lineage ]
        ch_channels_report = ch_channels.map { meta, _image, report, _lineage -> [meta, report] }
        ch_versions = ch_versions.mix(TILEPARITY_CHANNELS.out.versions.first())
    } else {
        ch_channels = channel.empty()
        ch_channels_report = ch_tiles.map { meta, _tiles -> [meta, []] }
    }

    // ---- Process 2b: fastLBP branch ----------------------------------------------------------
    if (params.skip_lbp) {
        ch_lbp = ch_tiles.map { meta, _tiles -> [meta, [], []] }
    } else {
        FASTLBP_EXTRACT(ch_channels.join(ch_tiles))
        ch_lbp = FASTLBP_EXTRACT.out.features
        ch_versions = ch_versions.mix(FASTLBP_EXTRACT.out.versions.first())
    }

    // ---- Processes 1a + 2a: UNI2-h branch --------------------------------------------------
    if (params.skip_uni2h) {
        ch_uni = ch_tiles.map { meta, _tiles -> [meta, [], []] }
    } else {
        if (params.uni2h_test_mode) {
            ch_weights = channel.value([])
        } else if (params.uni2h_weights) {
            ch_weights = channel.value(file(params.uni2h_weights, checkIfExists: true))
        } else {
            UNI2H_DOWNLOAD()
            ch_weights = UNI2H_DOWNLOAD.out.weights.first()
            ch_versions = ch_versions.mix(UNI2H_DOWNLOAD.out.versions)
        }
        ch_uni_in = ch_image.join(ch_tiles).join(ch_lineage)
        if (need_rgb_he) {
            ch_uni_in = ch_uni_in.join(ch_channels)
        } else {
            ch_uni_in = ch_uni_in.map { meta, image, tiles, lineage -> [meta, image, tiles, lineage, [], [], []] }
        }
        UNI2H_EMBED(ch_uni_in, ch_weights)
        ch_uni = UNI2H_EMBED.out.features
        ch_versions = ch_versions.mix(UNI2H_EMBED.out.versions.first())
    }

    // ---- Process 3: expression per tile ------------------------------------------------------
    ch_spatial = ch_input.map { meta, _image, spatial, _mask -> [meta, spatial] }
    TILEPARITY_AGGREGATEST(ch_tiles.join(TILEPARITY_PREPARE.out.assignment).join(ch_spatial).join(ch_lineage))
    ch_versions = ch_versions.mix(TILEPARITY_AGGREGATEST.out.versions.first())

    // ---- every output of one sample and scale -------------------------------------------------
    ch_sample = ch_tiles
        .join(TILEPARITY_PREPARE.out.json)
        .join(TILEPARITY_PREPARE.out.thumbnail)
        .join(ch_lineage)
        .join(ch_uni)
        .join(ch_lbp)
        .join(TILEPARITY_AGGREGATEST.out.counts)
        .join(ch_channels_report)
        .map { meta, tiles, json, thumb, prep_lineage, uni, uni_lineage, lbp, lbp_lineage, counts, st_lineage, channels_report ->
            [meta, [tiles, json, thumb, prep_lineage, uni, uni_lineage, lbp, lbp_lineage, counts, st_lineage, channels_report].flatten()]
        }

    // ---- optional: clustering of every view, per slide and scale ------------------------------------
    ch_clusters = channel.empty()
    ch_sample_export = ch_sample
    if (params.cluster) {
        TILEPARITY_CLUSTER(ch_sample)
        ch_clusters = TILEPARITY_CLUSTER.out.agreement
        ch_versions = ch_versions.mix(TILEPARITY_CLUSTER.out.versions.first())
        // the exports add the Leiden labels of the set keep to the tile table
        ch_sample_export = ch_sample
            .join(TILEPARITY_CLUSTER.out.labels)
            .map { meta, files, labels, cluster_lineage -> [meta, files + [labels, cluster_lineage]] }
    }

    // ---- Process 4: export, one task per scale with all samples ------------------------------
    ch_bundle = byScale(ch_sample_export)

    ch_h5ad = channel.empty()
    if (!params.skip_anndata) {
        TILEPARITY_EXPORTANNDATA(ch_bundle)
        ch_h5ad = TILEPARITY_EXPORTANNDATA.out.h5ad
        ch_versions = ch_versions.mix(TILEPARITY_EXPORTANNDATA.out.versions.first())
    }
    ch_rds = channel.empty()
    if (!params.skip_spe) {
        TILEPARITY_EXPORTSPE(ch_bundle)
        ch_rds = TILEPARITY_EXPORTSPE.out.rds
        ch_versions = ch_versions.mix(TILEPARITY_EXPORTSPE.out.versions.first())
    }

    // ---- optional QC: 3-D UMAP of every view, one page per scale -------------------------------
    ch_umap = channel.empty()
    if (params.qc_umap) {
        TILEPARITY_QCUMAP(byScale(ch_sample))
        ch_umap = TILEPARITY_QCUMAP.out.html
        ch_versions = ch_versions.mix(TILEPARITY_QCUMAP.out.versions.first())
    }

    emit:
    tiles      = ch_tiles                          // [ meta, tiles.tsv.gz ]
    uni2h      = ch_uni                            // [ meta, uni2h.h5, lineage ]
    lbp        = ch_lbp                            // [ meta, lbp.h5, lineage ]
    counts     = TILEPARITY_AGGREGATEST.out.counts // [ meta, counts.h5, lineage ]
    h5ad       = ch_h5ad                           // [ meta, [*.h5ad] ]
    rds        = ch_rds                            // [ meta, [*.rds] ]
    qc_umap    = ch_umap                           // [ meta, [html, tsv.gz] ]
    clusters   = ch_clusters                       // [ meta, cluster_agreement.tsv, cluster.json ]
    versions   = ch_versions
}

// One item per scale with the files of all its samples: [ [id: combined_mpp<scale>, scale_tag], files ]
def byScale(ch) {
    return ch
        .map { meta, files -> [meta.scale_tag, files] }
        .groupTuple()
        .map { tag, files -> [[id: "combined_mpp${tag}", scale_tag: tag], files.flatten()] }
}
