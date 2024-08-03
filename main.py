from fastapi import FastAPI
from subprocess import check_output
from helper import *
import os

home = os.environ['HOME']

app = FastAPI()

@app.get("/avail")
async def check_avail():
    available, avail = check_dcu_status()
    return available
