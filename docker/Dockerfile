FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install software-properties-common && \
    apt-get -y install language-pack-ja-base language-pack-ja ibus-mozc && \
    apt-get update 

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
RUN bash Anaconda3-2019.10-Linux-x86_64.sh -b

ENV PATH /root/Anaconda3/bin:$PATH
RUN echo $PATH

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
RUN apt-get update
RUN apt-get -y install cuda-drivers

RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
RUN curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

RUN apt-get update && apt-get install -y nvidia-container-toolkit
RUN systemctl restart docker

RUN pip install chainer
