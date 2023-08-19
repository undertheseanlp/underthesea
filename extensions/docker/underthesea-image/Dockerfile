FROM alpine:3.16
# Install required packages
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    build-base \
    pkgconfig \
    openssl-dev \
    libressl-dev \
    bash \
    rust \
    cargo \
    py3-scikit-learn && \
    pip3 install maturin patchelf
# Install underthesea_core
RUN curl -O https://files.pythonhosted.org/packages/b3/7f/8d136185263619e25473af7584bb224918a3546975afeb375b87b3bcae20/underthesea_core-1.0.4.tar.gz && \
    tar -xvf underthesea_core-1.0.4.tar.gz && \
    rm underthesea_core-1.0.4.tar.gz && \
    cd underthesea_core-1.0.4 && \
    maturin build && \
    pip3 install target/wheels/*.whl
# Install underthesea_core
RUN pip install underthesea