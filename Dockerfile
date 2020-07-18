# Dockerfile for nnenum
#
# To build image:
# docker build . -t nnenum_image
#
# To get a shell after building the image:
# docker run -ir nnenum_image bash

FROM tensorflow/tensorflow:2.2.0

COPY ./requirements.txt /work/requirements.txt

# set working directory
WORKDIR /work

# install python package dependencies
RUN pip3 install -r requirements.txt

# set environment variables
ENV PYTHONPATH=$PYTHONPATH:/work/nnenum
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# copy remaining files to docker
COPY . /work