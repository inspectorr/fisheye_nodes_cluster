# TODO

FROM python:3.9

WORKDIR /app

ADD . .

RUN python -m pip install -r requirements.txt
