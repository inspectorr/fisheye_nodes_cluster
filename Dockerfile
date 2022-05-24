FROM python:3.9

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
RUN pip install -e .

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:5000 app:app
