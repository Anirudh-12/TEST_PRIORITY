FROM ubuntu:20.04

# Avoid prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for building Python 3.6 and general tools
RUN apt-get update -qq && \
    apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git jq && \
    rm -rf /var/lib/apt/lists/*

# Build Python 3.6.15 from source
RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz >/dev/null 2>&1 && \
    tar xzf Python-3.6.15.tgz && \
    cd Python-3.6.15 && \
    ./configure --enable-optimizations >/dev/null 2>&1 && \
    make altinstall >/dev/null 2>&1 && \
    cd .. && rm -rf Python-3.6.15*

# Build Python 3.7.17 from source
RUN wget https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz >/dev/null 2>&1 && \
    tar xzf Python-3.7.17.tgz && \
    cd Python-3.7.17 && \
    ./configure --enable-optimizations >/dev/null 2>&1 && \
    make altinstall >/dev/null 2>&1 && \
    cd .. && rm -rf Python-3.7.17*

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

# Install modern python versions instantly using uv pre-compiled binaries
RUN uv python install 3.8 3.9 3.10

# Set workspace environment variables
ENV BUGSINPY_WORKSPACE=/workspace/bugsinpy_workspace
ENV HOME=/root
WORKDIR /workspace
