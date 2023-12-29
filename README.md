# OCR-Meter

Tool to extract digits from meter images.

## Installation

The package can be installed using `poetry install` or simply call `pip install .`.

## Usage

The API is defined in `app.py`:

```
python app.py
```

To start the data app defined in `stream.py` run:
```
streamlit run stream.py
```

Or use `docker` to build the image and/or use `docker-compose`:

```
sudo docker build -t ocr-meter .
sudo docker-compose build
```

## ImageProcessor

Define `opencv-python` pipelines using `ocr_meter.image_processor`.