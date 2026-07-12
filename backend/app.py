import os
import re
import json
import uuid

import chromadb
from chromadb.utils import embedding_functions
import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Hallucination Detective API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="knowledge_base", embedding_function=embed_fn)

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge. You will be given a CLAIM and a set of EVIDENCE chunks retrieved from a knowledge base.

Your job: decide whether the CLAIM is supported by the EVIDENCE.

Rules:
- Base your verdict ONLY on the provided evidence. Ignore anything you personally know that isn't in the evidence.
- If the evidence clearly supports the claim, verdict is GROUNDED.
- If the evidence clearly contradicts the claim, verdict is HALLUCINATED.
- If the evidence is missing, unrelated, or insufficient to decide either way, verdict is UNCERTAIN. Do not guess.
- Quote the specific evidence (by chunk source) that drove your decision.

Respond ONLY with valid JSON, no markdown fences, no preamble, matching exactly this schema:
{
  "verdict": "GROUNDED" | "HALLUCINATED" | "UNCERTAIN",
  "confidence": <float 0-1>,
  "reasoning": "<2-3 sentence explanation>",
  "supporting_evidence": ["<short quote or paraphrase from evidence>", ...]
}"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


class IngestRequest(BaseModel):
    text: str
    source: str = "uploaded_doc"


class CheckRequest(BaseModel):
    claim: str
    top_k: int = 4


@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": collection.count()}


@app.post("/ingest")
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    if not chunks:
        return {"chunks_added": 0, "total_documents": collection.count()}
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": req.source, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return {"chunks_added": len(chunks), "total_documents": collection.count()}


@app.post("/check")
def check(req: CheckRequest):
    results = collection.query(query_texts=[req.claim], n_results=req.top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    evidence = [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "relevance_score": round(1 - dist, 4),
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]

    if not evidence:
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "reasoning": "No evidence was retrieved from the knowledge base for this claim.",
            "supporting_evidence": [],
            "evidence_used": [],
        }

    evidence_block = "\n\n".join(
        f"[Evidence {i+1} | source: {e['source']} | relevance: {e['relevance_score']}]\n{e['text']}"
        for i, e in enumerate(evidence)
    )
    user_prompt = f"CLAIM:\n{req.claim}\n\nEVIDENCE:\n{evidence_block}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "reasoning": f"Judge returned unparseable output: {raw[:200]}",
            "supporting_evidence": [],
        }

    result["evidence_used"] = evidence
    return result
