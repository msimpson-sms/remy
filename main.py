from fastapi import FastAPI, Request
import openai, pinecone, os
from pydantic import BaseModel

# Config
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index = pinecone.Index("chat-logs")

app = FastAPI()

# Define schemas
class ChatLog(BaseModel):
    user_id: str
    message: str

@app.post("/log")
def log_message(data: ChatLog):
    # Save to vector DB
    emb = openai.Embedding.create(input=data.message, model="text-embedding-3-large")["data"][0]["embedding"]
    index.upsert([(data.user_id + "-" + str(hash(data.message)), emb, {"message": data.message})])
    return {"status": "ok"}

@app.post("/recall")
def recall(data: ChatLog):
    emb = openai.Embedding.create(input=data.message, model="text-embedding-3-large")["data"][0]["embedding"]
    res = index.query(vector=emb, top_k=5, include_metadata=True)
    messages = [match["metadata"]["message"] for match in res["matches"]]
    return {"matches": messages}
