# app.py
import os
import re
import uuid
import tempfile
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# PDF loader
from PyPDF2 import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# DB
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    Text,
    ARRAY,
)
from sqlalchemy.sql import select, insert

# LLM - your existing wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql://user:pass@localhost:5432/pdfdb
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL env var")
if not GEMINI_API_KEY:
    # we won't raise here to allow analyze endpoint to be set up, but you need it for chat
    pass

# ----------------------------
# Embedding model (sentence-transformers)
# ----------------------------
EMBED_MODEL_NAME = "all-mpnet-base-v2"
EMBED_DIM = 768  # model produces 768-dim vectors
print("Loading embedding model:", EMBED_MODEL_NAME)
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = EMBED_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # ensure float32
    return vecs.astype("float32")

def cosine_sim(a: np.ndarray, b: np.ndarray):
    # a: (d,), b: (d,)
    # return cosine similarity
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ----------------------------
# DB setup (SQLAlchemy, simple table)
# ----------------------------
engine = create_engine(DATABASE_URL, future=True)
metadata = MetaData()

# Table to store docs (meta)
pdf_docs = Table(
    "pdf_docs",
    metadata,
    Column("id", String, primary_key=True),
    Column("filename", String),
    Column("summary", Text),
    Column("num_chunks", Integer),
    Column("name", String, nullable=True),   # optional structured fields (resume)
    Column("email", String, nullable=True),
    Column("phone", String, nullable=True),
)

# Table to store chunks (one row per chunk, with embedding as float[] (ARRAY(Float)))
pdf_chunks = Table(
    "pdf_chunks",
    metadata,
    Column("id", String, primary_key=True),
    Column("doc_id", String, nullable=False, index=True),
    Column("chunk_id", String, nullable=False),
    Column("page", Integer, nullable=True),
    Column("text", Text, nullable=False),
    Column("embedding", ARRAY(Float, dimensions=1), nullable=False),  # stores float[] in Postgres
)

# Create tables if not exist
metadata.create_all(engine)

# ----------------------------
# FastAPI models
# ----------------------------
class Topic(BaseModel):
    topic: str
    text: str

class AnalyzeResponse(BaseModel):
    doc_id: str
    summary: str
    topics: List[Topic] = []
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class ChatRequest(BaseModel):
    doc_id: str
    question: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

# ----------------------------
# Utilities: PDF extraction, chunking, structured extraction
# ----------------------------
def extract_text_from_pdf(path: str) -> tuple[str, list[dict]]:
    """
    Return full_text and a list of per-page dicts: [{'page': i, 'text': '...'}]
    """
    reader = PdfReader(path)
    pages = []
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text})
        texts.append(text)
    full_text = "\n".join(texts)
    return full_text, pages

def chunk_text(full_text: str, page_texts: list[dict], chunk_size_chars=2000, overlap_chars=200):
    """
    Create chunks with small overlap. For each chunk we attempt to preserve page info by scanning pages.
    Return list of dicts: {'chunk_id', 'text', 'page'}
    """
    chunks = []
    start = 0
    L = len(full_text)
    chunk_id = 0
    while start < L:
        end = min(L, start + chunk_size_chars)
        chunk_text = full_text[start:end]
        # try to find middle page for this chunk
        page_num = None
        # naive mapping: find which page block contains most of this substring
        max_overlap = 0
        for p in page_texts:
            # overlap length = number of shared tokens (simple heuristic)
            overlap = len(set(chunk_text.split()).intersection(set(p["text"].split()))) if p["text"] else 0
            if overlap > max_overlap:
                max_overlap = overlap
                page_num = p["page"]
        chunks.append({"chunk_id": f"c{chunk_id}", "text": chunk_text.strip(), "page": page_num})
        chunk_id += 1
        start = max(end - overlap_chars, end)  # make sure progress
    return chunks

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4})")

def extract_structured_fields(text: str):
    email_m = EMAIL_RE.search(text)
    phone_m = PHONE_RE.search(text)
    # For name heuristic: look in first 500 chars, take capitalized lines and simple heuristics
    first_chunk = text[:1500]
    lines = [l.strip() for l in first_chunk.splitlines() if l.strip()]
    name = None
    # heuristic: first line with 2 words and capitalized
    for l in lines[:8]:
        parts = l.split()
        if 1 < len(parts) <= 4 and all(p[0].isupper() for p in parts if p):
            name = l
            break
    return {
        "name": name or None,
        "email": email_m.group(0) if email_m else None,
        "phone": phone_m.group(0) if phone_m else None,
    }

# heading extractor (your existing function adapted)
def extract_topics_from_text(full_text: str):
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    topics = []
    current_topic = None
    current_text = []
    heading_pattern = re.compile(r"^[A-Z][A-Z\s\.\'\-:,0-9]+$")
    for line in lines:
        if heading_pattern.match(line) and len(line.split()) <= 8:
            if current_topic:
                topics.append({"topic": current_topic.strip(), "text": " ".join(current_text).strip()})
                current_text = []
            current_topic = line
        else:
            current_text.append(line)
    if current_topic and current_text:
        topics.append({"topic": current_topic.strip(), "text": " ".join(current_text).strip()})
    return topics

# ----------------------------
# LLM wrapper helper
# ----------------------------
def call_gemini_answer(system_prompt: str, user_prompt: str) -> str:
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured in environment")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    # The wrapper you used previously uses `invoke`. Adjust if your wrapper differs.
    resp = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
    return resp.content

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="PDF RAG Chat API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "ok", "message": "PDF RAG Chat API running"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    # save temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        full_text, pages = extract_text_from_pdf(tmp_path)
        topics_raw = extract_topics_from_text(full_text)
        # simple summary via Gemini (you can keep or change)
        # We'll call Gemini to generate a concise summary of the doc
        try:
            summary_prompt = "Summarize the following document clearly and concisely, highlighting key ideas and names:\n\n" + full_text[:15000]
            summary = call_gemini_answer("You are a concise summarizer.", summary_prompt)
        except Exception as e:
            summary = "Summary generation failed: " + str(e)

        # chunk text and embed
        chunks = chunk_text(full_text, pages, chunk_size_chars=2000, overlap_chars=200)
        texts = [c["text"] for c in chunks]
        if len(texts) == 0:
            raise HTTPException(status_code=400, detail="No extractable text from PDF")
        vectors = embed_texts(texts)  # numpy (n, dim)

        # structured extraction (resume heuristics)
        s = extract_structured_fields(full_text)

        # store doc metadata + chunks in Postgres
        doc_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(
                insert(pdf_docs).values(
                    id=doc_id,
                    filename=file.filename,
                    summary=summary,
                    num_chunks=len(chunks),
                    name=s["name"],
                    email=s["email"],
                    phone=s["phone"],
                )
            )
            for i, c in enumerate(chunks):
                chunk_db_id = str(uuid.uuid4())
                # convert vector to python list for storage
                vec_list = vectors[i].astype(float).tolist()
                conn.execute(
                    insert(pdf_chunks).values(
                        id=chunk_db_id,
                        doc_id=doc_id,
                        chunk_id=c["chunk_id"],
                        page=c["page"],
                        text=c["text"],
                        embedding=vec_list,
                    )
                )

        topics = [Topic(**t) for t in topics_raw]
        return AnalyzeResponse(
            doc_id=doc_id,
            summary=summary,
            topics=topics,
            name=s["name"],
            email=s["email"],
            phone=s["phone"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/chat", response_model=ChatResponse)
async def chat_pdf(req: ChatRequest):
    # Validate doc exists - use .first() then ._mapping (safe)
    with engine.begin() as conn:
        doc_row = conn.execute(select(pdf_docs).where(pdf_docs.c.id == req.doc_id)).first()

    if not doc_row:
        raise HTTPException(status_code=404, detail="doc_id not found")

    # Access fields via _mapping for safety (Row or RowMapping)
    try:
        doc_meta = doc_row._mapping
    except Exception:
        # fallback: treat as tuple with known column order (less ideal)
        # but prefer the mapping; here we'll try a conservative approach
        doc_meta = {
            "id": doc_row[0],
            "filename": doc_row[1],
            "summary": doc_row[2],
            "num_chunks": doc_row[3],
            "name": doc_row[4],
            "email": doc_row[5],
            "phone": doc_row[6],
        }

    # 1) quick check structured fields for resume-like question heuristics
    q_lower = req.question.strip().lower()
    if q_lower.startswith("who ") or "candidate" in q_lower or "who is" in q_lower:
        # if doc has name metadata, answer immediately
        name = doc_meta.get("name")
        if name:
            return ChatResponse(answer=f"The candidate appears to be: {name}", sources=[{"type": "metadata", "field": "name"}])

    # 2) embed the question
    q_vec = embed_texts([req.question])[0]  # (dim,)

    # 3) fetch all chunks for this doc and compute similarity in Python
    with engine.begin() as conn:
        rows = conn.execute(select(pdf_chunks).where(pdf_chunks.c.doc_id == req.doc_id)).all()

    if not rows:
        raise HTTPException(status_code=404, detail="No chunks found for this document")

    sims = []
    for r in rows:
        # robustly access embedding (works if row is RowMapping or plain Row)
        try:
            emb_list = r._mapping["embedding"]
        except Exception:
            # fallback: try column-based access
            try:
                emb_list = r[pdf_chunks.c.embedding]
            except Exception:
                # final fallback: assume embedding is at known index (last column)
                emb_list = r[-1]

        emb = np.array(emb_list, dtype="float32")
        sim = cosine_sim(q_vec, emb)
        sims.append((sim, r))

    # sort by similarity descending
    sims_sorted = sorted(sims, key=lambda x: x[0], reverse=True)
    top_k = req.top_k or 5
    top_results = sims_sorted[:top_k]

    # build context text by concatenating top chunks (truncate if necessary)
    contexts = []
    sources = []
    for idx, (score, row) in enumerate(top_results):
        # access chunk text, id, page robustly
        try:
            row_map = row._mapping
            text = row_map.get("text")
            chunk_id = row_map.get("chunk_id")
            page = row_map.get("page")
        except Exception:
            # fallback to column access
            text = row[pdf_chunks.c.text]
            chunk_id = row[pdf_chunks.c.chunk_id]
            page = row[pdf_chunks.c.page]

        contexts.append(f"Source {idx+1} (chunk_id={chunk_id} page={page} score={score:.4f}):\n{text}\n---")
        sources.append({"chunk_id": chunk_id, "page": page, "score": float(score)})

    context_text = "\n".join(contexts)
    # Build grounded prompt
    system_prompt = (
        "You are an assistant that answers questions using ONLY the provided document excerpts. "
        "If the answer is not present in the excerpts, reply: 'I don't know from the document.' "
        "Keep the answer concise and cite sources as (Source 1) etc."
    )
    user_prompt = f"Document excerpts:\n{context_text}\n\nQuestion: {req.question}\nAnswer:"

    try:
        answer = call_gemini_answer(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return ChatResponse(answer=answer, sources=sources)
