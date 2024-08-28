from fastapi import FastAPI, status, UploadFile
from fastapi.responses import FileResponse
from typing import Annotated
from pydantic import BaseModel
from PIL import Image
from helper import *
from subprocess import Popen
import io
import os

home = os.environ['HOME']
available_dcu = -1

app = FastAPI()

class GenerateBody(BaseModel):
    file: UploadFile

class IdResponse(BaseModel):
    id: int

@app.get("/avail")
async def check_avail():
    available, avail = check_dcu_status()
    global available_dcu
    available_dcu = avail
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{avail}"
    return {"available": available}

@app.post("/generate", status_code=status.HTTP_200_OK, response_model=IdResponse)
async def generate(file: UploadFile):
    contents = await file.read()
    try:
        read_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        print(str(e))
    id = next_id()
    read_image.save(f"./inits/{id}.png")
    infer(id)
    return {"id": id}

@app.get("/check")
async def check_finished(id: int):
    results = os.listdir("./results")
    if f"{id}.png" in results:
        return {"finished": True}
    else:
        return {"finished": False}

@app.get("/fetch")
async def fetch_image(id: int):
    return FileResponse(f"./results/{id}.png")
