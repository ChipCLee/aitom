# BASE IMAGE
FROM continuumio/miniconda3:24.1.2-0

SHELL ["/bin/bash","-c"]

WORKDIR /opt
ADD . /opt
RUN apt-get update && apt-get -y install git gcc g++ liblapack-dev libblas-dev libboost-dev libarmadillo-dev libfftw3-dev
RUN pip install -r requirements.txt
RUN bash build.sh

EXPOSE 8888

