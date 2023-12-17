from const import CONFIG
from pipeline import pipeline
from models import ImageModel, LogConfig

from logging.config import dictConfig
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import uvicorn


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
    uvicorn.run(app, host="localhost", port=8002)
