# Build:  docker build -t ghcr.io/miguelrosell/tileparity-uni2h:0.2.0 containers/uni2h
# The same environment.yml is used by -profile conda, so both routes install the same software.
FROM mambaorg/micromamba:2.0.5
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes
# Make the environment usable without activation (Nextflow overrides the entrypoint)
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONNOUSERSITE=1
LABEL org.opencontainers.image.source="https://github.com/miguelrosell/tileparity" \
      org.opencontainers.image.description="tileparity uni2h environment" \
      org.opencontainers.image.licenses="MIT"
