"""
LMChat Backend - FastAPI server for local LM Studio chat interface
All data stored in local SQLite. Nothing leaves your machine.
"""

import asyncio
import json
import math
import os
import re
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path("lmchat.db")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_LM_STUDIO_URL = "http://localhost:8080"
CHUNK_SIZE = 500          # words per RAG chunk
CHUNK_OVERLAP = 50        # word overlap between chunks
MAX_RAG_CHUNKS = 5        # how many chunks to inject per message

# Global queue for memory extraction jobs — ensures we never hit llama-server
# while it's busy with a chat request
memory_queue: asyncio.Queue = asyncio.Queue()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL DEFAULT 'New Conversation',
            model      TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            system_prompt TEXT DEFAULT '',
            rag_enabled   INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            token_count     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS memory_entries (
            id         TEXT PRIMARY KEY,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_conversation_id TEXT
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            file_type   TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS doc_chunks (
            id          TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content     TEXT NOT NULL,
            -- Simple TF-IDF/BM25 search terms stored as JSON for fallback
            tokens      TEXT DEFAULT '[]'
        );

        -- FTS5 virtual table for full-text search on chunks
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
            chunk_id UNINDEXED,
            content,
            tokenize='porter unicode61'
        );
    """)

    # Default settings
    defaults = {
        "lm_studio_url": DEFAULT_LM_STUDIO_URL,
        "backend_type": "llamacpp",
        "default_model": "",
        "default_system_prompt": "You are a helpful, thoughtful assistant.",
        "memory_enabled": "true",
        "temperature": "0.7",
        "max_tokens": "4096",
        "context_window": "8192",
    }
    for key, value in defaults.items():
        c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))

    conn.commit()
    conn.close()


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Text processing / RAG
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 fallback."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z]{2,}\b', text)
    stopwords = {'the','a','an','and','or','but','in','on','at','to','for',
                 'of','with','by','from','is','was','are','were','be','been',
                 'have','has','had','do','does','did','will','would','could',
                 'should','may','might','shall','can','this','that','these',
                 'those','it','its','i','we','you','he','she','they','them'}
    return [t for t in tokens if t not in stopwords]


def bm25_score(query_tokens: list[str], doc_tokens: list[str], 
               avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """Compute BM25 score for a single document."""
    dl = len(doc_tokens)
    doc_tf: dict[str, int] = {}
    for t in doc_tokens:
        doc_tf[t] = doc_tf.get(t, 0) + 1
    
    score = 0.0
    for qt in query_tokens:
        if qt not in doc_tf:
            continue
        tf = doc_tf[qt]
        idf = math.log(1 + (1 / (tf + 0.5)))  # simplified IDF
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += idf * numerator / denominator
    return score


def search_chunks(query: str, conn: sqlite3.Connection, top_k: int = MAX_RAG_CHUNKS) -> list[dict]:
    """Search document chunks using FTS5, fall back to BM25."""
    results = []
    
    # Try FTS5 first
    try:
        rows = conn.execute("""
            SELECT dc.id, dc.content, dc.document_id, d.name as doc_name,
                   bm25(doc_chunks_fts) as score
            FROM doc_chunks_fts f
            JOIN doc_chunks dc ON dc.id = f.chunk_id
            JOIN documents d ON d.id = dc.document_id
            WHERE doc_chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, top_k)).fetchall()
        
        for row in rows:
            results.append({
                "chunk_id": row["id"],
                "content": row["content"],
                "document_id": row["document_id"],
                "doc_name": row["doc_name"],
                "score": row["score"],
            })
    except Exception:
        # FTS fallback: manual BM25
        query_tokens = tokenize(query)
        all_chunks = conn.execute("""
            SELECT dc.id, dc.content, dc.document_id, dc.tokens, d.name as doc_name
            FROM doc_chunks dc
            JOIN documents d ON d.id = dc.document_id
        """).fetchall()
        
        if all_chunks:
            avg_dl = sum(len(json.loads(r["tokens"])) for r in all_chunks) / len(all_chunks)
            scored = []
            for row in all_chunks:
                doc_tokens = json.loads(row["tokens"])
                score = bm25_score(query_tokens, doc_tokens, avg_dl)
                if score > 0:
                    scored.append((score, row))
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, row in scored[:top_k]:
                results.append({
                    "chunk_id": row["id"],
                    "content": row["content"],
                    "document_id": row["document_id"],
                    "doc_name": row["doc_name"],
                    "score": score,
                })
    
    return results


def build_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = ["<context>"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(f'<source name="{chunk["doc_name"]}" chunk="{i}">')
        parts.append(chunk["content"])
        parts.append("</source>")
    parts.append("</context>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Start the background memory extraction worker
    worker_task = asyncio.create_task(memory_queue_worker())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="LMChat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------
@app.get("/api/settings")
def get_settings():
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


@app.post("/api/settings")
async def update_settings(request: Request):
    data = await request.json()
    conn = get_db()
    for key, value in data.items():
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, str(value))
        )
    conn.commit()
    conn.close()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Models endpoint (proxy to LM Studio)
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def get_models():
    conn = get_db()
    url = conn.execute("SELECT value FROM settings WHERE key='lm_studio_url'").fetchone()["value"]
    conn.close()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/v1/models")
            data = resp.json()
            return {"models": [m["id"] for m in data.get("data", [])]}
    except Exception as e:
        return {"models": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@app.post("/api/models/load")
async def load_model(request: Request):
    data = await request.json()
    model = data.get("model", "")
    if not model:
        raise HTTPException(400, "model required")
    conn = get_db()
    rows = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings").fetchall()}
    url = rows.get("lm_studio_url", DEFAULT_LM_STUDIO_URL)
    backend_type = rows.get("backend_type", "llamacpp")
    conn.close()

    # llama-server loads one model at startup — no dynamic loading API
    # Just verify the model is already loaded and ready
    if backend_type == "llamacpp":
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{url}/v1/models")
                data = resp.json()
                loaded = [m["id"] for m in data.get("data", [])]
                if any(model in m or m in model for m in loaded):
                    return {"ok": True, "model": model, "note": "already loaded"}
                return {"ok": True, "model": loaded[0] if loaded else model,
                        "note": "llama-server has a fixed model — using whatever is loaded"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # LM Studio dynamic loading
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{url}/api/v0/models/load",
                json={"identifier": model, "preset": "default"},
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code in (200, 201):
                return {"ok": True, "model": model}
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}
            return {"ok": False, "error": str(body)}
    except httpx.TimeoutException:
        return {"ok": True, "model": model, "note": "timeout — model may still be loading"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
@app.get("/api/conversations")
def list_conversations():
    conn = get_db()
    rows = conn.execute("""
        SELECT c.*, COUNT(m.id) as message_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/conversations")
async def create_conversation(request: Request):
    data = await request.json()
    conn = get_db()
    settings = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings").fetchall()}
    
    cid = new_id()
    now = now_iso()
    conn.execute("""
        INSERT INTO conversations (id, title, model, created_at, updated_at, system_prompt, rag_enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        cid,
        data.get("title", "New Conversation"),
        data.get("model", settings.get("default_model", "")),
        now, now,
        data.get("system_prompt", settings.get("default_system_prompt", "")),
        int(data.get("rag_enabled", False)),
    ))
    conn.commit()
    row = conn.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    conn.close()
    return dict(row)


@app.get("/api/conversations/{cid}")
def get_conversation(cid: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    if not row:
        raise HTTPException(404, "Conversation not found")
    messages = conn.execute(
        "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at", (cid,)
    ).fetchall()
    conn.close()
    return {**dict(row), "messages": [dict(m) for m in messages]}


@app.patch("/api/conversations/{cid}")
async def update_conversation(cid: str, request: Request):
    data = await request.json()
    conn = get_db()
    fields = []
    values = []
    allowed = ["title", "model", "system_prompt", "rag_enabled"]
    for f in allowed:
        if f in data:
            fields.append(f"{f}=?")
            values.append(data[f])
    if fields:
        fields.append("updated_at=?")
        values.append(now_iso())
        values.append(cid)
        conn.execute(f"UPDATE conversations SET {', '.join(fields)} WHERE id=?", values)
        conn.commit()
    conn.close()
    return {"ok": True}


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str):
    conn = get_db()
    conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
    conn.commit()
    conn.close()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat / Streaming
# ---------------------------------------------------------------------------
@app.post("/api/conversations/{cid}/chat")
async def chat(cid: str, request: Request):
    data = await request.json()
    user_content: str = data.get("content", "")
    
    conn = get_db()
    conv = conn.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    if not conv:
        conn.close()
        raise HTTPException(404, "Conversation not found")
    
    settings = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings").fetchall()}
    lm_url = settings.get("lm_studio_url", DEFAULT_LM_STUDIO_URL)
    temperature = float(settings.get("temperature", 0.7))
    max_tokens = int(settings.get("max_tokens", 4096))
    memory_enabled = settings.get("memory_enabled", "true") == "true"
    
    # Build messages list
    history = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at",
        (cid,)
    ).fetchall()
    
    # Build system prompt
    system_parts = []
    if conv["system_prompt"]:
        system_parts.append(conv["system_prompt"])
    
    # Inject memory
    if memory_enabled:
        memories = conn.execute(
            "SELECT content FROM memory_entries ORDER BY updated_at DESC LIMIT 20"
        ).fetchall()
        if memories:
            mem_text = "\n".join(f"- {m['content']}" for m in memories)
            system_parts.append(f"\n<memory>\nHere is what you know about the user:\n{mem_text}\n</memory>")
    
    # RAG
    rag_context = ""
    if conv["rag_enabled"]:
        chunks = search_chunks(user_content, conn)
        if chunks:
            rag_context = build_rag_context(chunks)
            system_parts.append(f"\n{rag_context}\n\nUse the above context to inform your response when relevant.")
    
    system_prompt = "\n".join(system_parts)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})
    
    # Save user message
    user_msg_id = new_id()
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_msg_id, cid, "user", user_content, now_iso())
    )
    
    # Auto-title after first message
    if len(list(history)) == 0:
        title = user_content[:60] + ("..." if len(user_content) > 60 else "")
        conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, now_iso(), cid))
    else:
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now_iso(), cid))
    
    conn.commit()
    
    model = conv["model"] or settings.get("default_model", "")
    
    assistant_msg_id = new_id()
    
    async def stream_response() -> AsyncGenerator[str, None]:
        full_content = ""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{lm_url}/v1/chat/completions",
                                         json=payload,
                                         headers={"Content-Type": "application/json"}) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        yield f"data: {json.dumps({'error': error_text.decode()})}\n\n"
                        return
                    
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta and delta["content"]:
                                token = delta["content"]
                                full_content += token
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        except Exception:
                            continue
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to LM Studio. Is it running?'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        
        # Save assistant message
        db = get_db()
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (assistant_msg_id, cid, "assistant", full_content, now_iso())
        )
        db.commit()
        db.close()
        
        yield f"data: {json.dumps({'done': True, 'message_id': assistant_msg_id})}\n\n"
    
    conn.close()
    return StreamingResponse(stream_response(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# Auto Memory Extraction — queue-based so we never hit llama-server concurrently
# ---------------------------------------------------------------------------
MEMORY_EXTRACTION_PROMPT = (
    "You are a memory extractor. Your only job is to read a conversation and output a JSON array "
    "of facts about the user worth remembering permanently.\n\n"
    "CAPTURE EVERYTHING including:\n"
    "- Name, age, location, family, pets\n"
    "- Job, employer, skills, tools, side projects\n"
    "- Food & drink preferences (e.g. how they take coffee, dietary restrictions, favorites)\n"
    "- Hobbies, interests, sports, entertainment preferences\n"
    "- Opinions, values, things they like or dislike\n"
    "- Goals, plans, things they are working on\n"
    "- Health, constraints, allergies\n"
    "- Anything the user explicitly asks to be remembered\n\n"
    "FORMAT RULES:\n"
    "- Every memory starts with 'User'\n"
    "- One fact per string, concise and specific\n"
    "- Output ONLY the raw JSON array — no thinking, no explanation, no markdown fences\n"
    "- If the user said nothing personal at all, output exactly: []\n\n"
    "EXAMPLES:\n"
    "'I take my coffee black'  →  'User takes their coffee black'\n"
    "'please remember I hate mornings'  →  'User dislikes mornings'\n"
    "'I'm a mainframe developer at Northwestern Mutual'  →  'User is a mainframe developer at Northwestern Mutual'\n"
    "'my dog is named Max'  →  'User has a dog named Max'\n\n"
    "OUTPUT FORMAT: [\"User fact one\", \"User fact two\"]"
)


async def wait_for_server_free(lm_url: str, max_wait: int = 30) -> bool:
    """Poll /health until llama-server reports status=ok, up to max_wait seconds."""
    deadline = time.time() + max_wait
    async with httpx.AsyncClient(timeout=3) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{lm_url}/health")
                data = r.json()
                status = data.get("status", "")
                if status == "ok":
                    return True
                # "loading model" or "no slot available" — wait and retry
                print(f"[memory] server status: {status!r}, waiting...")
            except Exception:
                pass
            await asyncio.sleep(1)
    return False


async def run_memory_extraction(cid: str, lm_url: str, model: str):
    """The actual extraction logic — runs inside the queue worker."""
    import re as _re

    conn = get_db()
    history = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at",
        (cid,)
    ).fetchall()

    if not history:
        conn.close()
        return

    transcript_lines = []
    for msg in history:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        transcript_lines.append(f"{prefix}: {msg['content']}")
    transcript = "\n".join(transcript_lines)

    existing = [r["content"] for r in conn.execute(
        "SELECT content FROM memory_entries ORDER BY updated_at DESC LIMIT 100"
    ).fetchall()]
    existing_str = "\n".join(f"- {m}" for m in existing) if existing else "None"

    extraction_prompt = (
        f"Full conversation:\n{transcript}\n\n"
        f"Already stored (do not repeat these):\n{existing_str}\n\n"
        "Extract every new fact worth remembering from the USER's messages above."
    )

    print(f"[memory] running extraction for conv {cid[:8]}...")

    try:
        raw = ""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
                {"role": "user", "content": extraction_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
            "stream": True,
        }
        print(f"[memory] posting to {lm_url}/v1/chat/completions with model={model!r}")
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{lm_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as resp:
                print(f"[memory] HTTP status: {resp.status_code}")
                reasoning = ""
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk_str = line[6:].strip()
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        delta = chunk["choices"][0]["delta"]
                        # Collect both content and reasoning_content
                        # Qwen3 thinking mode puts everything in reasoning_content
                        for field in ("content", "reasoning_content"):
                            val = delta.get(field)
                            if val and isinstance(val, str):
                                if field == "content":
                                    raw += val
                                else:
                                    reasoning += val
                    except Exception:
                        continue
                # If content is empty but reasoning has the answer, extract from reasoning
                if not raw and reasoning:
                    print(f"[memory] no content field, searching reasoning for JSON array...")
                    # Find the last JSON array in the reasoning output
                    matches = list(_re.finditer(r"\[.*?\]", reasoning, _re.DOTALL))
                    if matches:
                        raw = matches[-1].group(0)
                        print(f"[memory] found array in reasoning: {raw!r}")
                print(f"[memory] collected content: {raw!r}")

        print(f"[memory] raw output: {raw!r}")

        # Strip <think> blocks from reasoning models
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.MULTILINE)
        raw = _re.sub(r"```\s*$", "", raw, flags=_re.MULTILINE)
        raw = raw.strip()

        # Find JSON array anywhere in output
        match = _re.search(r"\[.*?\]", raw, _re.DOTALL)
        if match:
            raw = match.group(0)

        memories = json.loads(raw)
        if not isinstance(memories, list):
            print("[memory] not a list, skipping")
            conn.close()
            return

        print(f"[memory] extracted {len(memories)} candidates: {memories}")

        saved = []
        now = now_iso()
        existing_lower = [e.lower() for e in existing]
        for mem in memories:
            mem = mem.strip()
            if not mem:
                continue
            if mem.lower() in existing_lower:
                print(f"[memory] duplicate, skipping: {mem!r}")
                continue
            mid = new_id()
            conn.execute(
                "INSERT INTO memory_entries (id, content, created_at, updated_at, source_conversation_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (mid, mem, now, now, cid)
            )
            saved.append(mem)
            existing_lower.append(mem.lower())
            print(f"[memory] saved: {mem!r}")

        conn.commit()
        conn.close()
        print(f"[memory] done — saved {len(saved)} memories")

    except Exception as e:
        print(f"[memory] ERROR during extraction: {e}")
        try:
            conn.close()
        except Exception:
            pass


async def memory_queue_worker():
    """Background worker — processes extraction jobs one at a time."""
    print("[memory] queue worker started")
    while True:
        try:
            job = await memory_queue.get()
            cid, lm_url, model = job
            print(f"[memory] got job for conv {cid[:8]}, waiting for server free...")
            free = await wait_for_server_free(lm_url, max_wait=60)
            if free:
                await run_memory_extraction(cid, lm_url, model)
            else:
                print(f"[memory] server never became free, dropping job for {cid[:8]}")
            memory_queue.task_done()
        except asyncio.CancelledError:
            print("[memory] queue worker stopped")
            break
        except Exception as e:
            print(f"[memory] queue worker error: {e}")


@app.post("/api/conversations/{cid}/extract-memory")
async def extract_memory(cid: str, request: Request):
    """Enqueue a memory extraction job — returns immediately, runs in background."""
    conn = get_db()
    settings = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings").fetchall()}
    memory_enabled = settings.get("memory_enabled", "true") == "true"

    if not memory_enabled:
        conn.close()
        return {"memories": [], "queued": False}

    lm_url = settings.get("lm_studio_url", DEFAULT_LM_STUDIO_URL)
    conv = conn.execute("SELECT model FROM conversations WHERE id=?", (cid,)).fetchone()
    model = conv["model"] if conv else settings.get("default_model", "")
    conn.close()

    await memory_queue.put((cid, lm_url, model))
    print(f"[memory] queued extraction for conv {cid[:8]}, queue size: {memory_queue.qsize()}")
    return {"memories": [], "queued": True}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@app.get("/api/memory")
def list_memory():
    conn = get_db()
    rows = conn.execute("SELECT * FROM memory_entries ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/memory")
async def create_memory(request: Request):
    data = await request.json()
    conn = get_db()
    mid = new_id()
    now = now_iso()
    conn.execute(
        "INSERT INTO memory_entries (id, content, created_at, updated_at, source_conversation_id) VALUES (?, ?, ?, ?, ?)",
        (mid, data["content"], now, now, data.get("source_conversation_id"))
    )
    conn.commit()
    row = conn.execute("SELECT * FROM memory_entries WHERE id=?", (mid,)).fetchone()
    conn.close()
    return dict(row)


@app.patch("/api/memory/{mid}")
async def update_memory(mid: str, request: Request):
    data = await request.json()
    conn = get_db()
    conn.execute(
        "UPDATE memory_entries SET content=?, updated_at=? WHERE id=?",
        (data["content"], now_iso(), mid)
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.delete("/api/memory/{mid}")
def delete_memory(mid: str):
    conn = get_db()
    conn.execute("DELETE FROM memory_entries WHERE id=?", (mid,))
    conn.commit()
    conn.close()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Documents / RAG
# ---------------------------------------------------------------------------
def extract_text_from_file(path: Path, file_type: str) -> str:
    if file_type == "pdf":
        try:
            import pypdf
            text_parts = []
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as e:
            raise HTTPException(500, f"PDF extraction failed: {e}")
    else:
        # txt, md
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


@app.get("/api/documents")
def list_documents():
    conn = get_db()
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    if ext not in ("pdf", "txt", "md"):
        raise HTTPException(400, "Only PDF, TXT, and MD files are supported")
    
    doc_id = new_id()
    save_path = UPLOAD_DIR / f"{doc_id}.{ext}"
    
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    text = extract_text_from_file(save_path, ext)
    chunks = chunk_text(text)
    
    conn = get_db()
    now = now_iso()
    conn.execute(
        "INSERT INTO documents (id, name, file_type, created_at, chunk_count) VALUES (?, ?, ?, ?, ?)",
        (doc_id, filename, ext, now, len(chunks))
    )
    
    for i, chunk in enumerate(chunks):
        chunk_id = new_id()
        tokens = tokenize(chunk)
        conn.execute(
            "INSERT INTO doc_chunks (id, document_id, chunk_index, content, tokens) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, doc_id, i, chunk, json.dumps(tokens))
        )
        # FTS index
        conn.execute(
            "INSERT INTO doc_chunks_fts (chunk_id, content) VALUES (?, ?)",
            (chunk_id, chunk)
        )
    
    conn.commit()
    conn.close()
    return {"id": doc_id, "name": filename, "chunk_count": len(chunks)}


@app.delete("/api/documents/{did}")
def delete_document(did: str):
    conn = get_db()
    # Remove from FTS first
    chunk_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM doc_chunks WHERE document_id=?", (did,)
    ).fetchall()]
    for cid in chunk_ids:
        conn.execute("DELETE FROM doc_chunks_fts WHERE chunk_id=?", (cid,))
    conn.execute("DELETE FROM documents WHERE id=?", (did,))
    conn.commit()
    conn.close()
    # Clean up file
    for ext in ("pdf", "txt", "md"):
        p = UPLOAD_DIR / f"{did}.{ext}"
        if p.exists():
            p.unlink()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path("frontend.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Frontend not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8099, reload=True)
