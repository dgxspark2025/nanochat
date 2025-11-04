FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    curl \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-ubuntu2404.pin && \
    mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb && \
    dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb && \
    cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-13-0 && \
    rm -f cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml README.md ./
COPY rustbpe ./rustbpe

RUN uv sync --extra gpu --group dev

RUN uv run maturin develop --manifest-path rustbpe/Cargo.toml

COPY . .

CMD ["/bin/bash"]
