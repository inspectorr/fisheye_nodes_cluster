FROM python:3.9

RUN python -m pip install tensorflow numpy

WORKDIR /app

ADD . .
