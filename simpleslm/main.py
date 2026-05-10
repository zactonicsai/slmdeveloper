from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import chromadb
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

client = chromadb.HttpClient(host="chroma", port=8000)
embeddings = OllamaEmbeddings(model="phi4-mini:3.8b", base_url="http://ollama:11434")
vectorstore = Chroma(collection_name="templates", embedding_function=embeddings, client=client)
llm = OllamaLLM(model="phi4-mini:3.8b", base_url="http://ollama:11434")

@app.get("/")
async def index():
    return HTMLResponse(open("static/index.html").read())

@app.post("/upload")
async def upload(file: UploadFile):
    text = (await file.read()).decode()
    vectorstore.add_texts([text])
    return {"status": "added"}

@app.post("/chat")
async def chat(msg: str):
    docs = vectorstore.similarity_search(msg, k=3)
    context = "\n".join(d.page_content for d in docs)
    prompt = PromptTemplate.from_template("Summarize/fill template using context:\n{context}\n\nUser: {msg}")
    chain = prompt | llm
    return {"reply": chain.invoke({"context": context, "msg": msg})}