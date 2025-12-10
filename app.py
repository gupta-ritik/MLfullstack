import os
import sys
import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

import pymongo
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException

from fastapi import FastAPI
import uvicorn   # <-- FIXED HERE
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, Request
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client['NetworkSecurity']
collection = database['Predictions']

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return "Training successful!!"
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# OPTIONAL MAIN RUNNER (KEEP ONLY IF YOU HAD THIS)
if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)   # <-- FIXED
