FROM python:3.11-slim AS builder

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

FROM python:3.11-slim

WORKDIR /ocr_meter
ENV DOCKER_CONTAINER=true

COPY --from=builder /ocr_meter .

EXPOSE 8001
EXPOSE 8000

CMD ["python", "app.py", "&", "streamlit", "run", "stream.py"]