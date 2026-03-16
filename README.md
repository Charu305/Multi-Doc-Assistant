# 📚 Multi-Doc Assistant — RAG with Persistent User Memory

> **A multi-document RAG application** that lets users upload and query multiple documents simultaneously — with **persistent user memory** that remembers preferences, past interactions, and context across sessions, containerised with Docker for deployment.

---

## 📌 Project Overview

This project is a direct architectural evolution beyond [Simple-RAG](https://github.com/Charu305/Simple-RAG). While Simple-RAG answered questions from a single PDF within one session, the **Multi-Doc Assistant** introduces two critical upgrades:

1. **Multi-document support** — users can upload and query across multiple documents at once, with the system intelligently retrieving from the right source
2. **Persistent user memory** (`user_memory.json`) — the assistant remembers user preferences, previously asked questions, and interaction context *across sessions*, not just within a single conversation

Together, these make the system feel less like a search engine and more like a personal knowledge assistant that gets better the more you use it.

---

## 🎯 Problem Statement

> *Allow users to query across multiple uploaded documents simultaneously — with a persistent memory layer that retains user context, preferences, and history between sessions.*

**Real-world applications:**
- **Legal teams** — query across multiple contracts, policies, and case documents at once
- **Research assistants** — ask questions across a library of papers and reports
- **Enterprise knowledge bases** — navigate multiple internal documents (HR policies, product manuals, SOPs) in one interface
- **Personal assistants** — the system remembers your preferences and past questions, eliminating repetition

---

## 🏗️ System Architecture

```
User Session
(may be returning — memory loaded)
            │
            ▼
┌──────────────────────────────────────────┐
│         app/  (Application Layer)        │
│  Streamlit UI — upload docs, chat,       │
│  view memory, manage session             │
└─────────┬────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│         Memory Layer                     │
│         user_memory.json                 │
│                                          │
│  Loads on session start:                 │
│  • User preferences                      │
│  • Past Q&A history                      │
│  • Previously referenced documents       │
│                                          │
│  Updates on session end:                 │
│  • New interactions appended             │
│  • Preferences updated                   │
└─────────┬────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│         Multi-Document RAG Pipeline      │
│                                          │
│  1. Parse all uploaded PDFs              │
│  2. Chunk each document                  │
│  3. Embed all chunks (with doc tags)     │
│  4. Store in unified vector index        │
│                                          │
│  At query time:                          │
│  5. Embed user question                  │
│  6. Retrieve Top-K chunks across         │
│     ALL documents                        │
│  7. Augment prompt with retrieved        │
│     chunks + memory context              │
│  8. LLM generates grounded answer        │
│  9. Update user_memory.json              │
└──────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
Multi-Doc-Assistant/
│
├── app/                    # Core application module
│   ├── main.py             # App entry point — Streamlit UI
│   ├── rag_pipeline.py     # Multi-document RAG logic
│   ├── memory.py           # Memory read/write operations
│   └── utils.py            # PDF parsing, chunking, embedding helpers
├── user_memory.json        # Persistent user memory store
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## 🔬 Technical Deep Dive

### 1. Multi-Document RAG Pipeline (`app/rag_pipeline.py`)

The core upgrade from single-document RAG is handling **multiple documents** in a unified vector index while preserving **document attribution** — so the system knows which document each answer came from.

**Document tagging during indexing:**

```python
# Each chunk is stored with its source document as metadata
for doc in uploaded_documents:
    chunks = chunk_document(doc)
    for chunk in chunks:
        vectorstore.add(
            text=chunk.text,
            metadata={"source": doc.filename, "page": chunk.page}
        )
```

**Cross-document retrieval at query time:**

```python
# Single similarity search retrieves across ALL documents
results = vectorstore.similarity_search(user_query, k=5)

# Each result carries its source document
for r in results:
    print(f"From: {r.metadata['source']} — {r.text[:100]}...")
```

This means a question like *"How do the leave policies in Document A and Document B differ?"* can be answered by retrieving relevant chunks from both documents simultaneously.

### 2. Persistent User Memory (`user_memory.json`)

`user_memory.json` is the feature that distinguishes this from every other RAG project. It stores structured user context that **persists across sessions**:

```json
{
  "user_id": "charu_001",
  "preferences": {
    "response_style": "concise",
    "preferred_language": "English",
    "show_sources": true
  },
  "interaction_history": [
    {
      "timestamp": "2024-10-15T14:23:11",
      "question": "What is the maternity leave duration?",
      "answer": "According to the HR Handbook, maternity leave is...",
      "documents_referenced": ["HR_Policy_Handbook.pdf"]
    }
  ],
  "frequently_referenced_docs": ["HR_Policy_Handbook.pdf"],
  "session_count": 7
}
```

**How memory improves the experience:**
- **Continuity** — returning users don't start from scratch; the assistant remembers what was already discussed
- **Personalisation** — response style, verbosity, and source citation preferences are remembered
- **History-aware answers** — the system can reference previous interactions: *"As we discussed last session, the policy states..."*
- **Document familiarity** — frequently referenced documents can be prioritised in retrieval

**Memory lifecycle:**
```
Session Start → Load user_memory.json → inject relevant history into prompt
During Session → answer questions with memory context
Session End → append new Q&A pairs → update user_memory.json
```

### 3. Memory-Augmented Prompt Construction

Unlike Simple-RAG where only retrieved chunks are injected, here the prompt includes **both retrieved document chunks AND relevant memory**:

```
System: You are a helpful document assistant. You have access to the
        user's interaction history and their uploaded documents.

User Memory Context:
  - Previously asked about: maternity leave, remote work policy
  - Preferred response style: concise with bullet points
  - Last session: 2 days ago

Retrieved Document Context:
  [chunk from HR_Policy.pdf]
  [chunk from Remote_Work_Guide.pdf]

Question: {user_question}

Answer:
```

This enables the LLM to give genuinely personalised, context-aware responses — not just stateless retrieval answers.

### 4. Modular Application Structure (`app/`)

The `app/` package separates concerns cleanly across modules:

| Module | Responsibility |
|---|---|
| `main.py` | Streamlit UI — file uploads, chat interface, session management |
| `rag_pipeline.py` | Document processing, vector indexing, retrieval logic |
| `memory.py` | Read/write `user_memory.json`, format memory for prompt injection |
| `utils.py` | PDF parsing, text chunking, embedding generation |

Each module can be developed, tested, and replaced independently — the architecture supports swapping the vector store, LLM, or memory backend without touching the UI.

### 5. Docker Containerisation (`Dockerfile`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501"]
```

The `user_memory.json` file is persisted via a **Docker volume mount** in production — ensuring memory survives container restarts.

---

## 💬 Example Interactions

**Session 1 (new user):**
```
User:  "What does the HR policy say about remote work?"
Bot:   According to the Remote_Work_Guide.pdf, employees may work
       remotely up to 3 days per week provided...
       [memory updated: asked about remote work]
```

**Session 2 (returning user, 3 days later):**
```
Bot:   Welcome back! Last time you asked about remote work policy.
       You have 2 new documents uploaded since your last session.

User:  "And what about sick leave — is it the same policy?"
Bot:   The sick leave policy (HR_Policy.pdf, Section 4) is separate
       from the remote work guidelines. Unlike the remote work policy
       we discussed previously, sick leave is...
       [cross-references previous session memory]
```

---

## 📊 Evolution from Simple-RAG

| Feature | Simple-RAG | Multi-Doc Assistant |
|---|---|---|
| Documents supported | 1 PDF | Multiple PDFs simultaneously |
| Session memory | None — stateless | Persistent across sessions |
| Personalisation | None | Response style, history-aware |
| Source attribution | Single source | Per-chunk multi-source attribution |
| Cross-doc queries | Not possible | Natively supported |
| Returning user experience | Starts fresh every time | Remembers context and preferences |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| LLM | OpenAI GPT / Google Gemini |
| RAG Framework | LangChain |
| Vector Store | FAISS / ChromaDB |
| Embeddings | Sentence Transformers / OpenAI Embeddings |
| Memory Store | JSON (`user_memory.json`) |
| PDF Parsing | pdfplumber / PyPDF2 |
| Web UI | Streamlit |
| Containerisation | Docker |

---

## 🚀 How to Run

### Option A — Local

```bash
git clone https://github.com/Charu305/Multi-Doc-Assistant.git
cd Multi-Doc-Assistant

pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"

streamlit run app/main.py
# Open http://localhost:8501
```

### Option B — Docker

```bash
docker build -t multi-doc-assistant .

# Mount user_memory.json as a volume so memory persists across container restarts
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your-api-key" \
  -v $(pwd)/user_memory.json:/app/user_memory.json \
  multi-doc-assistant
```

---

## 💡 Key Learnings & Takeaways

- **Document metadata tagging is what makes multi-doc RAG work** — storing the source filename and page number with every chunk enables both cross-document retrieval and transparent source attribution. Without metadata, you get answers but no way to trace where they came from.
- **Persistent memory transforms a tool into an assistant** — the difference between a stateless search tool and a true assistant is memory. `user_memory.json` is a simple but effective implementation of this — storing it as JSON means it's human-readable, debuggable, and portable without a database.
- **Memory-augmented prompts require careful context selection** — injecting the full interaction history into every prompt would exceed the LLM's context window. Selecting only the *most relevant* recent interactions (by topic similarity to the current question) is the right approach.
- **Modular `app/` structure scales** — separating UI, RAG pipeline, memory, and utilities into distinct modules means the system can grow (new document types, new vector stores, new memory backends) without code entanglement.
- **Docker volume mounts are essential for stateful containers** — a container that resets `user_memory.json` on every restart defeats the purpose of memory. Mounting the file as a volume ensures persistence across deployments.

---

## 🔮 Potential Enhancements

- **Vector-based memory retrieval** — instead of injecting all history, embed past Q&A pairs and retrieve only the most semantically relevant ones for the current question
- **Database-backed memory** — replace `user_memory.json` with PostgreSQL or Redis for multi-user support and concurrent access
- **Document management UI** — let users see, remove, and re-index uploaded documents between sessions
- **Cross-document summarisation** — a dedicated mode that synthesises information across all uploaded documents on a given topic
- **Streaming responses** — stream LLM output token-by-token for a more responsive chat experience

---

## 👩‍💻 Author

**Charunya**
🔗 [GitHub Profile](https://github.com/Charu305)

---

## 📄 License

This project is developed for educational and research purposes.
