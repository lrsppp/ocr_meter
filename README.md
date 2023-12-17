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

Or use `docker` to build the image and run the docker container:

```
sudo docker build -t ocr-meter .
sudo docker run -p 8001:8001 ocr-meter
```

##