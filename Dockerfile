FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

ADD . .

RUN pip install -e .

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:5000 app:app
