FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

SHELL [ "/bin/bash", "-c"]
# Install required packages
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -yq software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt install -yq apt-transport-https \
    gcc \
    g++ \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/xview3
WORKDIR /home/xview3

ADD environment.yml environment.yml

# Changing parameter back to default
ENV DEBIAN_FRONTEND=newt
# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda update -n base -c defaults conda \
    && conda env create -f ./environment.yml \
    && conda init bash

RUN mkdir /input
RUN mkdir /output

# https://github.com/pytorch/pytorch/issues/27971
ENV LRU_CACHE_CAPACITY 1

# set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1

# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1


ADD xview3 /home/xview3/xview3
ADD run_inference.sh /home/xview3/
ADD docker_inference.py /home/xview3/
RUN chmod +x /home/xview3/run_inference.sh

ADD submissions/1128_b5_b4_vs2_fliplr/traced_ensemble.jit \
    /home/xview3/submissions/1128_b5_b4_vs2_fliplr/traced_ensemble.jit

ADD submissions/1128_b5_b4_vs2_fliplr/1128_b5_b4_vs2_fliplr_test__config_0.300_vsl_0.338_fsh_0.350_step_1536_tta_fliplr.yaml \
    /home/xview3/config.yaml

# Set entrypoint
ENTRYPOINT [ "/home/xview3/run_inference.sh" ]
