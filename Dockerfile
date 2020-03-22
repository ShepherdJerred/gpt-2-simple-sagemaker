FROM tensorflow/tensorflow:1.15.2-gpu-py3

ENV LANG=C.UTF-8

RUN mkdir /opt/program/
WORKDIR /opt/program/

COPY requirements.txt /opt/program
RUN pip3 install -r requirements.txt

COPY . /opt/program

ENV PYTHONPATH="/opt/program/src/"
ENV PATH="/opt/program/:${PATH}"

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

