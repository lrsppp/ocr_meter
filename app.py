import os
import logging
from logging.config import dictConfig

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from ocr_meter.const import API_PORT, CONFIG
from ocr_meter.models import ImageModel, LogConfig
from ocr_meter.pipeline import pipeline

dictConfig(LogConfig().model_dump())
logger = logging.getLogger("ocr_meter")
logger.info("Logging configured successfully on startup")

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-image")
async def upload_image(image_data: dict):
    try:
        image_data = image_data.get("image_data")
        image_model = ImageModel(image_data=image_data)

        logger.info("File uploaded.")
        digit = pipeline(image_model=image_model, config=CONFIG)
        return JSONResponse(content={"digit": digit})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    if os.environ.get("DOCKER_CONTAINER"):
        uvicorn.run(app, host="0.0.0.0", port=API_PORT)
    else:
        uvicorn.run(app, host="localhost", port=API_PORT)
