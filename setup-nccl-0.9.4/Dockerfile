# 如因网络问题无法编译，则从bin目录直接拷贝二进制文件
# FROM lukemathwalker/cargo-chef:latest-rust-1.71 AS chef

# RUN git clone https://github.com/huggingface/text-generation-inference.git -b v0.9.4
# WORKDIR $pwd/text-generation-inference
# COPY launcher/src/main.rs launcher/src/main.rs
# RUN cd launcher && cargo build --release

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS cuda

FROM sakurahua/lm_inference:open-tgi-v0.9.4 AS base
WORKDIR /app
ENV CONDA_PACKAGES_PATH=/opt/conda/lib/python3.9/site-packages

# COPY --from=chef /text-generation-inference/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher-mpi
COPY bin/text-generation-launcher /usr/local/bin/text-generation-launcher-mpi
COPY comm_lib comm_lib

# 配置apt
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wget git vim ssh libxml2 && \
    rm -rf /etc/apt/sources.list.d
# 安装openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.bz2 && \
    tar -xjvf openmpi-4.1.6.tar.bz2 && \
    cd openmpi-4.1.6 && ./configure --prefix=/usr/local && make all && make install
# 安装nccl
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt install libnccl2=2.17.1-1+cuda12.1 libnccl-dev=2.17.1-1+cuda12.1 && \
    rm -rf /var/lib/apt/lists/*
RUN pip install mpi4py

FROM base AS optim
WORKDIR /app
COPY --from=cuda /usr/local/cuda-11.8 /usr/local/cuda-11.8

# 安装通信优化py包并测试
RUN cd comm_lib && python setup.py install

FROM base
WORKDIR /app
ARG PY_DIR=server/text_generation_server
COPY --from=optim $CONDA_PACKAGES_PATH/comm_lib* $CONDA_PACKAGES_PATH
COPY $PY_DIR/server.py $CONDA_PACKAGES_PATH/text_generation_server/
COPY $PY_DIR/models/flash_llama.py $CONDA_PACKAGES_PATH/text_generation_server/models/
COPY $PY_DIR/models/custom_modeling/flash_llama_modeling.py $CONDA_PACKAGES_PATH/text_generation_server/models/custom_modeling/
COPY $PY_DIR/utils/layers.py $PY_DIR/utils/mpi_dist.py $CONDA_PACKAGES_PATH/text_generation_server/utils/

# 删除过期文件，减小镜像大小
RUN mv /usr/local/bin/text-generation-launcher /usr/local/bin/text-generation-launcher-ori && \
    mv /usr/local/bin/text-generation-launcher-mpi /usr/local/bin/text-generation-launcher && \
    chmod +x /usr/local/bin/text-generation-launcher && \
    rm -rf /app
WORKDIR /usr/src
