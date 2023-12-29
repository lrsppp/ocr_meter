FROM python:3.11-slim

WORKDIR /ocr_meter
ENV DOCKER_CONTAINER=true

COPY . /ocr_meter

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1

EXPOSE 8001
EXPOSE 8000