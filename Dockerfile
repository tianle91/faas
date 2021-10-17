FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install -y python3.7 python3-pip
RUN apt-get install --no-install-recommends -y openjdk-11-jdk-headless ca-certificates-java

RUN ln -fs /usr/bin/python3.7 /usr/bin/python
RUN python -m pip install -U pip

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

ENV PYSPARK_PYTHON=/usr/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python
