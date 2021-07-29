FROM python:3.9.6-buster

WORKDIR /ws
COPY ./requirements.txt /ws/requirements.txt
RUN pip install -U pip && \
    pip install -r requirements.txt
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*
COPY . /ws
