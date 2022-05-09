from nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 

WORKDIR /cbert

COPY requirements.txt requirements.txt
COPY code code

# Disable nvidia repos as their GPG key isn't recognized
RUN mv /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda.list.bak
RUN mv /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/nvidia-ml.list.bak

RUN apt-get update &&\
    apt-get install --no-install-recommends --no-install-suggests -y gnupg2 \
                                                                     ca-certificates \
                                                                     git \
                                                                     build-essential \
                                                                     python3-dev \
                                                                     python3-pip \
                                                                     wget \
								     curl \
                                                                     rsync


RUN pip install -r requirements.txt
