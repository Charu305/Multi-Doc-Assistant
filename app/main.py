from fastapi import FastAPI, UploadFile, BackgroundTasks
import os
from config import DATA_PATH
from pdf_ingestion import process_pdf
from rag import ask_question
from stream import stream_response

app = FastAPI(title="Multi Document Assistant")

os.makedirs(DATA_PATH, exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile, bg: BackgroundTasks):
    path=f"{DATA_PATH}/{file.filename}"
    with open(path,"wb") as f:
        f.write(await file.read())
    bg.add_task(process_pdf,path)
    return {"status":"processing"}

@app.post("/ask")
def ask(q:dict):
    return ask_question(q["question"])

@app.get("/ask-stream")
def ask_stream(question:str):
    return stream_response(question)

@app.get("/")
def home():
    return {"status":"running"}
