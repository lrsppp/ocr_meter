from const import CONFIG
from pipeline import pipeline
from models import ImageModel

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        image_model = ImageModel(image_data=image.tolist())
        digit = pipeline(image_model=image_model, config=CONFIG)
        return JSONResponse(content={"file_name": file.filename, "digit": digit})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
