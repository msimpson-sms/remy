import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai
from pinecone import Pinecone, ServerlessSpec

# Load API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone v3 client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "chat-logs"

# Optional: create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region="us-west1")
    )

index = pc.Index(index_name)

# Initialize FastAPI app
app = FastAPI()

# Define request model
class ChatLog(BaseModel):
    user_id: str
    message: str

# Log message endpoint
@app.post("/log")
def log_message(data: ChatLog):
    emb = openai.Embedding.create(
        input=data.message,
        model="text-embedding-3-large"
    )["data"][0]["embedding"]

    vector_id = f"{data.user_id}-{hash(data.message)}"
    index.upsert(vectors=[(vector_id, emb, {"message": data.message})])

    return {"status": "ok"}

# Recall endpoint
@app.post("/recall")
def recall(data: ChatLog):
    emb = openai.Embedding.create(
        input=data.message,
        model="text-embedding-3-large"
    )["data"][0]["embedding"]

    res = index.query(
        vector=emb,
        top_k=5,
        include_metadata=True
    )

    messages = [match["metadata"]["message"] for match in res["matches"]]
    return {"matches": messages}
