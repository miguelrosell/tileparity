// Download the gated UNI2-h weights once per run (needs HF_TOKEN and internet access).
// Prefer downloading them once with scripts/download_uni2h_weights.sh and passing
// --uni2h_weights, especially on clusters whose compute nodes have no internet.
process UNI2H_DOWNLOAD {
    tag "MahmoodLab/UNI2-h"
    label 'process_single'

    conda "${projectDir}/containers/uni2h/environment.yml"
    container "${params.container_uni2h}"

    output:
    path "uni2h/pytorch_model.bin", emit: weights
    path "versions.yml"            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    if [ -z "\${HF_TOKEN:-}" ]; then
        echo "HF_TOKEN is not set: export a Hugging Face token with access to MahmoodLab/UNI2-h, or pass --uni2h_weights" >&2
        exit 1
    fi
    export HF_HOME=\${PWD}/.hf_home
    tp_download_uni2h.py --outdir uni2h

    tp_versions.py "${task.process}" huggingface_hub > versions.yml
    """

    stub:
    """
    mkdir -p uni2h && touch uni2h/pytorch_model.bin
    echo '"${task.process}": {tileparity: stub}' > versions.yml
    """
}
