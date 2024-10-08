ARG BASE=nvidia/cuda:12.6.1-base-ubuntu24.04
FROM ${BASE}

# Install OS dependencies:
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    gcc g++ \
    make \
    python3 python3-dev python3-pip python3-venv python3-wheel \
    espeak-ng libsndfile1-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install hamer:
COPY . /hamer
WORKDIR /hamer

# Create virtual environment:
RUN python3 -m venv .venv

# Activate virtual environment and install dependencies:
# REVIEW: We need to install/upgrade wheel and setuptools first because otherwise installation fails:
RUN --mount=type=cache,target=/root/.cache/pip \
    /bin/bash -c "source .venv/bin/activate && pip install --upgrade wheel setuptools"
RUN --mount=type=cache,target=/root/.cache/pip \
    /bin/bash -c "source .venv/bin/activate && pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118"

RUN ls -la /hamer

# REVIEW: Numpy is installed separately because otherwise installation fails:
RUN --mount=type=cache,target=/root/.cache/pip \
    /bin/bash -c "source .venv/bin/activate && pip install numpy"

RUN --mount=type=cache,target=/root/.cache/pip \
    /bin/bash -c "source .venv/bin/activate && pip install -e .[all]"
RUN --mount=type=cache,target=/root/.cache/pip \
    /bin/bash -c "source .venv/bin/activate && pip install -v -e third-party/ViTPose"

# Acquire the example data:
RUN /bin/bash fetch_demo_data.sh