FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install development tools and dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
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
    libgtest-dev \
    cmake-data \
    gcc-13 \
    g++-13 \
    zsh \
    fonts-powerline && \
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
ENV CC=/usr/bin/gcc-13
ENV CXX=/usr/bin/g++-13
ENV CCACHE_DIR=/ccache

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install Oh My Zsh and plugins
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="agnoster"/' ~/.zshrc && \
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting docker cmake)/' ~/.zshrc

# Set zsh as default shell
USER root
RUN chsh -s /usr/bin/zsh $USERNAME
USER $USERNAME

# Configure git
RUN git config --global core.editor "vim" && \
    git config --global init.defaultBranch main

# Set default shell to zsh
ENV SHELL=/usr/bin/zsh
