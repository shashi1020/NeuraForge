# NeuraForgeAI – PDF RAG Chat Backend
frontend: https://github.com/shashi1020/NeuraForgeAI-Frontend

NeuraForgeAI is a robust FastAPI-based backend designed for **PDF understanding**, **document chunking**, **embedding-based retrieval**, and **chat over documents using RAG (Retrieval-Augmented Generation)**. It extracts PDF text, generates a concise summary, stores chunks with embeddings in PostgreSQL, and allows users to chat with grounded, citation-based answers using Gemini.

---

## 🚀 Features

* **Upload & Analyze PDFs**

  * Extracts per-page text using PyPDF2
  * Creates smart overlapping chunks
  * Generates vector embeddings using Sentence Transformers (MPNet)
  * Stores chunks & metadata in PostgreSQL
  * Auto-extracts email, phone, and name (resume mode)
  * Detects document headings/topics
  * Generates summary using Gemini

* **Chat Over Documents**

  * Embeds the question
  * Performs similarity search over document chunks
  * Sends only relevant excerpts to Gemini
  * Returns grounded answers + source citations

* **Tech Stack**

  * **FastAPI**
  * **PostgreSQL** (with array embedding fields)
  * **SQLAlchemy**
  * **Sentence-Transformers** (MPNet)
  * **Gemini 2.5 Flash** API

---

## 🛠️ Project Structure

```
app.py            # Main FastAPI backend
.env              # Environment variables
requirements.txt  # Python dependencies
```

---

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/NeuraForgeAI.git
cd NeuraForgeAI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create `.env` file:

```
DATABASE_URL=postgresql://user:password@localhost:5432/neura_db
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

### 5. Setup PostgreSQL

Ensure PostgreSQL is running and create database:

```bash
psql -U postgres
CREATE DATABASE neura_db;
```

The backend automatically creates tables on first run.

---

## ▶️ Running the Backend

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

### API Docs

FastAPI auto-generates interactive Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

## 📁 API Endpoints

### **1. POST /analyze** — Upload & Process PDF

**Body:** multipart/form-data (PDF file)

**Response:**

* doc_id
* summary
* extracted topics
* structured metadata

### **2. POST /chat** — Ask Questions About PDF

**Body:** JSON

```json
{
  "doc_id": "<uuid>",
  "question": "What is the main idea?",
  "top_k": 5
}
```

**Response:**

* grounded answer
* source chunks with scores

---

## 🧪 Testing

Use `curl` or tools like Postman.

### Upload PDF

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -F "file=@sample.pdf"
```

### Chat with PDF

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"<id>", "question":"What does the document talk about?"}'
```

---

## 📌 Notes

* This backend uses **CPU-based embeddings** — GPU not required.
* Gemini API key is required for summary & chat generation.
* PostgreSQL stores embeddings as float arrays.

---

## 🧭 Roadmap

* [ ] Switch to **pgvector** for fast similarity search
* [ ] Add PDF image OCR support
* [ ] Add authentication (JWT)
* [ ] Deploy on Render / Railway / GCP
* [ ] Docker support

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue to discuss.

---

Built with ❤️ for smarter document intelligence.
