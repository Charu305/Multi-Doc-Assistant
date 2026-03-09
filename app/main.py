from fastapi import FastAPI, UploadFile, BackgroundTasks
import os
from app.config import DATA_PATH
from app.pdf_ingestion import process_pdf
from app.rag import ask_question
from app.stream import stream_response

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
