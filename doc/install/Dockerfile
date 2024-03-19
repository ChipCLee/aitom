# BASE IMAGE
FROM continuumio/miniconda3

SHELL ["/bin/bash","-c"]

WORKDIR /opt
ADD . /opt
RUN apt-get update && apt-get -y install git gcc g++ liblapack-dev libblas-dev libboost-dev libarmadillo-dev libfftw3-dev \
    && pip install -r requirements.txt \
    && bash build.sh

EXPOSE 8888

