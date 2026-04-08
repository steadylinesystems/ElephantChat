# LMChat

A local AI chat interface powered by LM Studio. 100% local — no data leaves your machine.

## Features

- **Multiple conversations** with persistent history
- **Persistent memory** — facts injected into every chat automatically
- **RAG** — upload PDF/TXT/MD documents and search them per conversation
- **Streaming responses** from LM Studio
- **Configurable** — LM Studio URL, temperature, max tokens, system prompts
- **All data in SQLite** — one file, fully portable, no cloud

## Requirements

- Python 3.10+ (`brew install python` if not installed)
- LM Studio running with a model loaded (default: http://localhost:1234)

## Setup (macOS)

```bash
chmod +x start.sh
./start.sh
```

Then open **http://localhost:8099** in your browser.

If you hit a pip permissions error, use a virtual environment instead:

```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-multipart pypdf httpx aiofiles rank-bm25
python3 -m uvicorn backend:app --host 0.0.0.0 --port 8099 --reload
```

## First Run

1. Open Settings (⚙️ in the sidebar) and confirm your LM Studio URL
2. Select a model from the dropdown in the chat footer
3. Click ✏️ New Chat and start talking

## RAG Usage

1. Click **📚 Documents** in the sidebar and upload PDF, TXT, or MD files
2. Open a conversation → click ⚙ (conversation settings) → enable RAG
3. Or click the **📎 RAG** toggle directly in the chat footer
4. Relevant document chunks will be injected into your prompts automatically

## Memory

Click **🧠 Memory** to add persistent facts (e.g. "User works in enterprise software"). These are injected into every conversation's system prompt when memory is enabled.

## Data

- `lmchat.db` — all conversations, messages, memory, and document indexes
- `uploads/` — your uploaded files

Both live in the same directory as this README. Safe to back up, move, or copy.

## Ports

- LMChat: `8099`
- LM Studio: `1234` (configurable in Settings)

## Architecture

```
frontend.html  ←→  backend.py (FastAPI)  ←→  lmchat.db (SQLite FTS5)
                         ↕
                  LM Studio (localhost:1234)
```

RAG uses SQLite's built-in FTS5 virtual table (Porter stemmer + BM25 ranking). No Chroma, no Qdrant, no external vector database.
