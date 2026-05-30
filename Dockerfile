# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build-base
FROM ubuntu:24.04 AS build-base
RUN userdel -r ubuntu

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-color \
    LANGUAGE=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LC_CTYPE=en_US.UTF-8 \
    LC_MESSAGES=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        fonts-liberation \
        fonts-noto-color-emoji \
        git \
        gpg \
        gpg-agent \
        less \
        libasound2t64 \
        libatk-bridge2.0-0t64 \
        libatk1.0-0t64 \
        libatspi2.0-0t64 \
        libbz2-dev \
        libcairo2 \
        libcups2t64 \
        libdbus-1-3 \
        libdrm2 \
        libffi-dev \
        libgbm1 \
        libgl1 \
        libglib2.0-0t64 \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libnspr4 \
        libnss3 \
        libpango-1.0-0 \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxkbcommon0 \
        libxrandr2 \
        llvm \
        locales \
        openssh-server \
        poppler-utils \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        sudo \
        tk-dev \
        tzdata \
        unzip \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
        zstd \
    && sed -i "s/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g" /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && pip install uv --break-system-packages \
    && apt-get clean

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

ARG USERNAME=devpod
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID ${USERNAME} \
    && useradd --uid $USER_UID --gid $USER_GID -ms /bin/bash ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && echo 'export PATH=${PATH}:~/.local/bin' >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /home/${USERNAME}

ENV UV_LINK_MODE=copy

ENV VIRTUAL_ENV=/venv
RUN sudo mkdir "$VIRTUAL_ENV" \
    && sudo chown -R ${USERNAME}:${USERNAME} "$VIRTUAL_ENV"

ENV BUILD_DIR=/app
COPY --chown=${USERNAME}:${USERNAME} . "$BUILD_DIR"

WORKDIR "${BUILD_DIR}"
RUN uv lock && uv sync --active && uv cache clean
RUN $VIRTUAL_ENV/bin/playwright install chromium
WORKDIR /home/${USERNAME}


###########################################################
FROM build-base AS build-dev

USER root
RUN /venv/bin/playwright install-deps
USER ${USERNAME}

RUN curl -fsSL https://claude.ai/install.sh | bash

CMD ["bash"]
