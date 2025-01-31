FROM rockylinux/rockylinux:9

# Install development tools and dependencies
RUN dnf -y update && \
    dnf -y install dnf-plugins-core epel-release && \
    dnf -y config-manager --set-enabled crb && \
    dnf -y groupinstall "Development Tools" && \
    dnf -y install \
    cmake \
    gcc-c++ \
    make \
    git \
    ninja-build \
    python3 \
    python3-pip \
    gdb \
    vim \
    openssh-clients \
    which \
    tar \
    clang \
    clang-tools-extra \
    ccache \
    sudo && \
    dnf clean all

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
