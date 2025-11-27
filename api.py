import os
import re
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# ------------------- ENV SETUP -------------------
load_dotenv()



def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Don't kill the app on startup; just fail when someone calls the endpoint
        raise RuntimeError(
            "GEMINI_API_KEY"
        )
    return api_key


# ------------------- CORE LOGIC (reuse from app.py) -------------------
def extract_topics_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]

    topics = []
    current_topic = None
    current_text = []
    heading_pattern = re.compile(r"^[A-Z][A-Z\s\.\'\-:,0-9]+$")

    for line in lines:
        if heading_pattern.match(line) and len(line.split()) <= 8:
            if current_topic:
                topics.append(
                    {
                        "topic": current_topic.strip(),
                        "text": " ".join(current_text).strip(),
                    }
                )
                current_text = []
            current_topic = line
        else:
            current_text.append(line)

    if current_topic and current_text:
        topics.append(
            {
                "topic": current_topic.strip(),
                "text": " ".join(current_text).strip(),
            }
        )

    return topics, full_text


def summarize_with_gemini(full_text: str) -> str:
    api_key = get_api_key()  # ✅ check inside the function, not at import time

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )
    prompt = (
        "Summarize the following document clearly and concisely. "
        "Highlight key ideas, important names, and any factual details. "
        "Make it structured, readable, and professional.\n\n"
        f"{full_text[:15000]}"
    )
    response = llm.invoke(prompt)
    return response.content


# ------------------- FASTAPI MODELS -------------------
class Topic(BaseModel):
    topic: str
    text: str


class AnalyzeResponse(BaseModel):
    topics: List[Topic]
    summary: str


# ------------------- FASTAPI APP -------------------
app = FastAPI(
    title="Smart AI Document Content Extractor API",
    description="REST API for PDF topic extraction + Gemini summary",
    version="1.0.0",
)

# CORS so Android app can call this from device/emulator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; later restrict to your app/domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "PDF AI API is running"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    # 1) Validate file
    if file.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # 2) Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        # 3) Extract topics + text
        topics_raw, full_text = extract_topics_from_pdf(temp_path)

        # 4) Summarize with Gemini
        summary = summarize_with_gemini(full_text)

        # 5) Convert to response
        topics = [Topic(**t) for t in topics_raw]

        return AnalyzeResponse(
            topics=topics,
            summary=summary,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    finally:
        # 6) Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
