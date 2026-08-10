from __future__ import annotations
import os
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

import llm  # noqa: E402
import embeddings  # noqa: E402
import singlestore  # noqa: E402
from api import router as api_router  # noqa: E402


app = FastAPI(title="SingleStore Demo Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3011",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3011",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    model: Literal["opus", "sonnet", "haiku"] = "sonnet"
    messages: list[Message]
    system: str | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class ChatResponse(BaseModel):
    reply: str


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int


@app.get("/health")
def health() -> dict:
    out: dict = {"ok": True, "demo": os.environ.get("DEMO_NAME", "")}
    try:
        out["singlestore"] = singlestore.ping()
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["singlestore"] = {"ok": False, "error": str(e)}
    return out


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        text = llm.chat(req.model, [m.model_dump() for m in req.messages], system=req.system, max_tokens=req.max_tokens)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")
    return ChatResponse(reply=text)


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    try:
        vectors = embeddings.embed(req.texts)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")
    return EmbedResponse(vectors=vectors, dim=len(vectors[0]) if vectors else 0)
