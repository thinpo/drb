FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install development tools and dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    gdb \
    vim \
    openssh-client \
    curl \
    wget \
    sudo \
    pkg-config \
    clang \
    clang-format \
    clang-tidy \
    ccache \
    libgtest-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up C++ development environment
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CCACHE_DIR=/ccache

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Configure git
RUN git config --global core.editor "vim" && \
    git config --global init.defaultBranch main
