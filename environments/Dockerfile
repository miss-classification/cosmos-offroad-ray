# cosmos-offroad-ray runtime image for the B200 cloud (x86_64, host NVIDIA driver >= 580 for CUDA 13).
#
# OS dependencies + orchestration (Ray-side) Python deps only. The Cosmos environments are NOT baked in:
# they are built once onto the mounted root, exactly as on the A100 cluster, so they survive container restarts:
#   bash environments/build_cosmos_env.sh     -> $OFFROAD_ROOT/envs/ct25-v1.5.4-45c6f0fb-cu130-py3.13 (edge + depth)
#   bash environments/build_cosmos3_env.sh    -> $OFFROAD_ROOT/envs/cosmos3-<diffusers sha12>-cu130-py3.13
# No model downloads, credentials, ENTRYPOINT or CMD (log in to Hugging Face at run time).
#
#   docker build -t offroad-cosmos:b200 environments/     (no COPY, so it also passes the Anyscale image builder)
FROM anyscale/ray:2.60.0-py313-cu130

RUN sudo apt-get update && \
    sudo env DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs rsync unzip ffmpeg \
        build-essential pkg-config \
        libx11-dev libgl1 libglib2.0-0 libegl1 libgomp1 && \
    sudo rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OFFROAD_ROOT=/mnt/cluster_storage/cosmos-offroad \
    HF_HOME=/mnt/cluster_storage/cosmos-offroad/hf_home \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Orchestration deps. No COPY allowed (Anyscale image builder), so pins are inlined here.
# Must match pyproject.toml [project.dependencies] + [test]; tests/test_dockerfile.py checks this.
RUN pip install --no-cache-dir \
        boto3==1.29.7 \
        numpy==2.2.6 \
        opencv-python-headless==4.12.0.88 \
        pandas==2.3.3 \
        pyarrow==23.0.1 \
        pydantic==2.12.3 \
        pyyaml==6.0.3 \
        pillow==11.3.0 \
        jinja2==3.1.6 \
        matplotlib==3.10.8 \
        nvidia-ml-py==13.580.82 \
        pytest==8.4.2

# Build-time checks (no GPU needed).
RUN python -c "import sys, ray, pandas, pyarrow, cv2, pydantic; \
assert sys.version_info[:2] == (3, 13); assert ray.__version__ == '2.60.0'; print(sys.version, 'Ray', ray.__version__)"
RUN ffmpeg -version && ffprobe -version && git lfs version && rsync --version && uv --version
# B200, like A100, has no NVENC: outputs are encoded on CPU with libx264.
RUN ffmpeg -hide_banner -loglevel error -f lavfi -i "color=c=black:s=128x128:r=16" \
    -frames:v 2 -c:v libx264 -pix_fmt yuv420p -f null -
