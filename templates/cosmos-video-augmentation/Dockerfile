# Template image: Ray plus the isolated NVIDIA Cosmos Transfer 2.5 environment.
# Ray and the notebooks use the base Python; Cosmos runs in /opt/cosmos as a subprocess of each GPU actor.
ARG BASE_IMAGE=anyscale/ray:2.60.0-py313-cu130
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt "uv==0.9.5"

# Cosmos Transfer 2.5 v1.5.4 (commit 45c6f0fb), locked dependencies, CUDA 13.0 wheels.
COPY scripts/build_cosmos_env.sh /tmp/build_cosmos_env.sh
RUN sudo mkdir -p /opt/cosmos && sudo chown -R "$(id -u):$(id -g)" /opt/cosmos && \
    OFFROAD_ROOT=/opt/cosmos UV_CACHE_DIR=/tmp/uv-cache bash /tmp/build_cosmos_env.sh && \
    rm -rf /tmp/uv-cache

ENV COSMOS_ENV_ROOT=/opt/cosmos \
    OFFROAD_IMAGE_DEPS=1 \
    COSMOS_SHARED=/mnt/user_storage/cosmos-video-augmentation
