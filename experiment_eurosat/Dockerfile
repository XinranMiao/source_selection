#docker run -it --rm=true xinranmiao/cuda_transfer:220717 /bin/bash
#export PATH=$PWD/python/bin:$PATH
#export PYTHONPATH=$PWD/packages
#export HOME=$PWD
#source activate example-environment
FROM nvidia/cuda:11.4.0-base-ubuntu20.04
# Add some dependencies
RUN apt-get clean && apt-get update -y -qq \
  && DEBIAN_FRONTEND="noninteractive" apt-get install --yes --no-install-recommends curl git build-essential vim ffmpeg libsm6 libxext6\
  && apt-get install wget \
  && apt-get install unzip

RUN useradd --shell /bin/bash --create-home --home-dir /home/xinranmiao xinranmiao
USER xinranmiao
WORKDIR /home/xinranmiao

# Setup for conda installation
ENV PATH=${PATH}:/home/xinranmiao/miniconda3/bin/
COPY environment.yml .



RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
  && bash Miniconda3-py38_4.8.2-Linux-x86_64.sh -b \
  && conda env create -f environment.yml



ENV CONDA_DEFAULT_ENV $example-environment
