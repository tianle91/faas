FROM python:3.9

RUN apt-get update -y
RUN apt-get install --no-install-recommends -y openjdk-11-jdk-headless

RUN pip install -U pip
RUN pip install pyspark==3.2.0

ARG PYTHON_PATH=/usr/local/bin/python
ENV PYSPARK_PYTHON=$PYTHON_PATH
ENV PYSPARK_DRIVER_PYTHON=$PYTHON_PATH

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
