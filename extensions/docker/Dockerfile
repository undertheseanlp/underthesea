FROM centos:7

RUN yum install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion openssl-devel

# Install Anaconda & Python
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Install Rust
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH $CARGO_HOME/bin:$PATH

RUN mkdir -p "$CARGO_HOME" && mkdir -p "$RUSTUP_HOME" && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable && \
    chmod -R a=rwX $CARGO_HOME

RUN cargo --version

# Funny stuff !!!
RUN mkdir underthesea

RUN conda update conda
ENV PATH /opt/conda/envs/python3.7/bin:$PATH
RUN conda create -f -y -n python3.7 python=3.7
RUN ln -s /opt/conda/envs/python3.7/bin/python3.7 /usr/bin/python3.7
RUN source activate python3.7 && conda install -y pip
RUN python3.7 -m pip install poetry
# RUN conda run --no-capture-output -n python3.7 conda install pip

# RUN conda create -f -y -n python3.8 python=3.8
# RUN ln -s /opt/conda/envs/python3.8/bin/python3.8 /usr/bin/python3.8
# RUN python3.8 -m ensurepip --upgrade
