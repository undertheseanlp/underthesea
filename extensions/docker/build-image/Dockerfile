FROM almalinux:8

RUN dnf install -y wget bzip2 ca-certificates \
    git openssl-devel gcc make curl

# Install Rust
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH $CARGO_HOME/bin:$PATH

RUN mkdir -p "$CARGO_HOME" && mkdir -p "$RUSTUP_HOME" && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable && \
    chmod -R a=rwX $CARGO_HOME

RUN cargo --version

# Install Miniconda (auto-detect architecture for x86_64/aarch64 support)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    echo "Downloading Miniconda for $ARCH from $MINICONDA_URL" && \
    wget --quiet "$MINICONDA_URL" -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda update -y conda
ENV PATH /opt/conda/envs/python3.10/bin:$PATH
ENV PATH /opt/conda/envs/python3.11/bin:$PATH
ENV PATH /opt/conda/envs/python3.12/bin:$PATH
ENV PATH /opt/conda/envs/python3.13/bin:$PATH
ENV PATH /opt/conda/envs/python3.14/bin:$PATH

# python 3.10
RUN conda create -y -n python3.10 python=3.10
RUN ln -s /opt/conda/envs/python3.10/bin/python3.10 /usr/bin/python3.10
RUN /opt/conda/envs/python3.10/bin/python -m pip install --upgrade pip
RUN python3.10 -m pip install poetry

# python 3.11
RUN conda create -y -n python3.11 python=3.11
RUN ln -s /opt/conda/envs/python3.11/bin/python3.11 /usr/bin/python3.11
RUN /opt/conda/envs/python3.11/bin/python -m pip install --upgrade pip
RUN python3.11 -m pip install poetry

# python 3.12
RUN conda create -y -n python3.12 python=3.12
RUN ln -s /opt/conda/envs/python3.12/bin/python3.12 /usr/bin/python3.12
RUN /opt/conda/envs/python3.12/bin/python -m pip install --upgrade pip
RUN python3.12 -m pip install poetry

# python 3.13
RUN conda create -y -n python3.13 python=3.13
RUN ln -s /opt/conda/envs/python3.13/bin/python3.13 /usr/bin/python3.13
RUN /opt/conda/envs/python3.13/bin/python -m pip install --upgrade pip
RUN python3.13 -m pip install poetry

# python 3.14
RUN conda create -y -n python3.14 python=3.14
RUN ln -s /opt/conda/envs/python3.14/bin/python3.14 /usr/bin/python3.14
RUN /opt/conda/envs/python3.14/bin/python -m pip install --upgrade pip
RUN python3.14 -m pip install poetry
