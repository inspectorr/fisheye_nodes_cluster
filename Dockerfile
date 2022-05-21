FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app

COPY . .

RUN python -m pip install -r requirements.txt
RUN pip install -e .
