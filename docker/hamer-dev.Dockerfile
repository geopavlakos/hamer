ARG BASE=nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM ${BASE} as hamer

# Install OS dependencies:
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y --no-install-recommends \
    gcc g++ \
    make \
    python3 python3-dev python3-pip python3-venv python3-wheel \
    espeak-ng libsndfile1-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libglfw3-dev libgles2-mesa-dev -y

# Install hamer:
WORKDIR /app
COPY . .

# Create virtual environment
RUN python3 -m venv /opt/venv

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Activate virtual environment and install dependencies:
# REVIEW: We need to install/upgrade wheel and setuptools first because otherwise installation fails:
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade wheel setuptools

# Install torch and torchaudio
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# REVIEW: Numpy is installed separately because otherwise installation fails:
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy

# Install project dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .[all]

# Install ViTPose
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -v -e third-party/ViTPose

# Install gdown
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install gdown

# Acquire the example data
# RUN /bin/bash fetch_demo_data.sh
