# docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t spc -f Dockerfile .
# go into https://catalog.ngc.nvidia.com/containers to fine the compatibale docker image base.
FROM nvcr.io/nvidia/tensorflow:21.02-tf1-py3

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  libsdl-sge-dev python3 python3-pip

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
# RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash
RUN apt-get install -y python3-requests
RUN rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools psutil

# Setup the football env
RUN python -m pip install git+https://github.com/ltzheng/football.git

# Python packages we use (or used at one point...)
COPY ./dockerfile_requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

# Install torch with cuda
RUN python -m pip install ray==2.0.1
RUN python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

ARG USERNAME=rundong
ARG USER_UID=1011
ARG USER_GID=$USER_UID

RUN groupadd -g $USER_GID $USERNAME && useradd -u $USER_UID -g $USER_GID -s /bin/bash $USERNAME && \
    groupadd -r docker && adduser $USERNAME docker && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    mkdir -p /etc/sudoers.d && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

ENV SC2PATH /StarCraftII
COPY ./SC2.4.10.zip SC2.4.10.zip
COPY ./scripts/install_sc2.sh install_sc2.sh
RUN sh install_sc2.sh && rm requirements.txt install_sc2.sh
RUN chmod -R 777 $SC2PATH

USER $USERNAME

WORKDIR /home/football

# how to run nvidia-docker
# nvidia-docker run -it --rm --cpus=24 --memory 32gb --gpus all --name spc_exp -v /home/longtao/Code/CurriculumMARL:/home/football spc bash
# run_experiments.bash python train.py -f configs