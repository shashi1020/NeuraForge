# 🚀 NeuraForgeAI – PDF RAG Chat API

**AI-powered PDF summarizer + Q&A with Retrieval-Augmented Generation (RAG)**
Built with **FastAPI, PostgreSQL, OpenAI embeddings, Gemini LLM**, and fully **Railway-ready**.

This backend allows users to:

* Upload a PDF
* Extract summary, topics, and structured metadata (name/email/phone)
* Ask questions based on the PDF content
* Receive accurate, citation-backed answers via RAG
* Zero persistence for user sessions — pure "upload → chat → done" service

---

## 📁 Project Structure

```
/
├── app.py               # Main FastAPI application
├── requirements.txt     # Python dependencies
├── Procfile             # Railway process declaration
├── .gitignore
├── README.md
└── (optional) .env      # Local environment variables
```

---

## ✨ Features

### 📄 PDF Analysis (`/analyze`)

* Extracts full PDF text using PyPDF2
* Generates a concise summary using **Gemini 2.5 Flash**
* Extracts headings/topics
* Extracts name/email/phone (resume-style parsing)
* Splits text into chunks for RAG
* Stores chunks + embeddings in PostgreSQL

### 💬 Chat With PDF (`/chat`)

* Accepts `doc_id` + user question
* Retrieves top-K relevant chunks using vector similarity
* Constructs a grounded prompt for Gemini
* Returns answer + sources (chunk_id, page number, similarity)

### ⚙️ Embeddings

Uses **OpenAI text-embedding-3-small** — lightweight, fast, reliable, and perfect for Railway free tier.

---

## 🛠️ Tech Stack

| Component      | Technology            |
| -------------- | --------------------- |
| Backend API    | FastAPI               |
| Database       | PostgreSQL (Railway)  |
| Embeddings     | OpenAI                |
| LLM            | Gemini Flash 2.5      |
| Vector Storage | Postgres ARRAY(Float) |
| Deployment     | Railway               |

---

## ▶️ API Endpoints

### **1️⃣ Upload PDF & Analyze**

```
POST /analyze
```

**Body (multipart/form-data):**

* `file`: PDF file

**Response:**

```json
{
  "doc_id": "f45e9d0c-23f4-4d9b-a2d0-aa6cd8cbcc12",
  "summary": "...",
  "topics": [],
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+91 9876543210"
}
```

---

### **2️⃣ Chat With PDF**

```
POST /chat
```

**Request:**

```json
{
  "doc_id": "f45e9d0c-23f4-4d9b-a2d0-aa6cd8cbcc12",
  "question": "What is the title of the document?",
  "top_k": 5
}
```

**Response:**

```json
{
  "answer": "The title is 'Precision Farming for Rural India'.",
  "sources": [
    { "chunk_id": "c0", "page": 1, "score": 0.081 },
    { "chunk_id": "c1", "page": 1, "score": 0.076 }
  ]
}
```

---

## ⚙️ Environment Variables

Create a local `.env` file:

```
DATABASE_URL=postgresql://user:password@host:port/dbname
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

Railway will supply `DATABASE_URL` after adding a Postgres plugin.

---

## 📦 Installation (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuraForgeAI.git
cd NeuraForgeAI
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run server

```bash
uvicorn app:app --reload
```

### 5. Test locally

#### Analyze PDF

```bash
curl -F "file=@sample.pdf" http://localhost:8000/analyze
```

#### Chat

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"doc_id":"...", "question":"What is the title?"}' \
  http://localhost:8000/chat
```

---

## 🗄️ Database Schema

### `pdf_docs`

| Column     | Type       |
| ---------- | ---------- |
| id         | UUID (str) |
| filename   | str        |
| summary    | text       |
| num_chunks | int        |
| name       | str?       |
| email      | str?       |
| phone      | str?       |

### `pdf_chunks`

| Column    | Type    |
| --------- | ------- |
| id        | UUID    |
| doc_id    | UUID    |
| chunk_id  | str     |
| page      | int     |
| text      | text    |
| embedding | float[] |

Tables auto-create via `metadata.create_all(engine)`.

---

## 🚀 Deployment on Railway (Quick Steps)

1. Push this repo to GitHub.
2. Create new Railway project → **Deploy from GitHub**.
3. Add **PostgreSQL plugin**.
4. Add environment variables:

   * `DATABASE_URL` (auto-generated)
   * `OPENAI_API_KEY`
   * `GEMINI_API_KEY`
5. Add `Procfile` containing:

```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

6. Deploy and test via the public URL.

---

## 💡 Notes

* Railway free tier has limited RAM → avoid local ML models.
* OpenAI embeddings are optimal for fast PDF chat.
* Gemini Flash handles summary + grounded answers.
* Stateless API: no login, no session history.

---
