import os
import zipfile
from dotenv import load_dotenv

# ---------- Load .env ----------
load_dotenv(dotenv_path=".env", override=True)

# app.py — Stately local “AI memory” with LanceDB + OpenAI
import uuid
from typing import List, Optional

import pyarrow as pa
import lancedb
from fastapi import FastAPI, UploadFile, File, Form, Request, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI
import httpx

# ---------- OpenAI Client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env file")

oa = OpenAI(api_key=OPENAI_API_KEY)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")


def embed(texts: List[str]) -> List[List[float]]:
    """Return 1536-dim embeddings for a list of texts."""
    resp = oa.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


# ---------- LanceDB (explicit schema) ----------
# On Render this points at your persistent disk
DB_PATH = "/data/stately_lancedb"
TABLE = "stately_brain"

db = lancedb.connect(DB_PATH)

schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("project", pa.string()),
        pa.field("source", pa.string()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("vector", pa.list_(pa.float32(), 1536)),
    ]
)


def open_or_create_table():
    if TABLE in db.table_names():
        try:
            t = db.open_table(TABLE)
            # if previous bad schema exists, drop & recreate
            col_types = {f.name: str(f.type) for f in t.schema}
            if col_types.get("project") == "null" or col_types.get("source") == "null":
                db.drop_table(TABLE)
                t = db.create_table(TABLE, schema=schema)
        except Exception:
            try:
                db.drop_table(TABLE)
            except Exception:
                pass
            t = db.create_table(TABLE, schema=schema)
    else:
        t = db.create_table(TABLE, schema=schema)

    # optional: try to create an ANN index (safe if it already exists)
    try:
        t.create_index(column="vector", metric="L2")
    except Exception:
        pass
    return t


t = open_or_create_table()

# ---------- FastAPI app ----------
app = FastAPI()

# ---------- Slack endpoints ----------


def _run_brain_and_reply(question: str, response_url: str, user_id: str):
    """Background job for Q&A via /brain."""
    try:
        req = AskReq(question=question, project=None, k=6)
        result = ask(req)

        if isinstance(result, dict) and "error" in result:
            msg = f"Stately Brain error: {result.get('error')}"
        else:
            msg = (
                result.get("answer_draft")
                if isinstance(result, dict)
                else str(result)
            )
        if not msg:
            msg = "I didn't get a usable answer back from Stately Brain."
    except Exception as e:
        msg = f"Stately Brain error while processing your question: {e}"

    try:
        httpx.post(
            response_url,
            json={"response_type": "ephemeral", "text": msg},
            timeout=30.0,
        )
    except Exception as e:
        print("SLACK SEND ERROR (Q&A):", repr(e))


def _run_ingest_and_reply(raw_text: str, response_url: str, user_id: str):
    """Background job for ingestion via text: `/brain ingest ...`."""
    try:
        text = raw_text.strip()
        parts = text.split()
        if not parts or parts[0].lower() != "ingest":
            msg = "To ingest, start your command with `ingest`."
        else:
            tokens = parts[1:]
            project = None
            tags_list = []
            body_tokens = []

            for tok in tokens:
                lower = tok.lower()
                if lower.startswith("project="):
                    project = tok.split("=", 1)[1]
                elif lower.startswith("tags="):
                    tags_raw = tok.split("=", 1)[1]
                    tags_list = [
                        t.strip() for t in tags_raw.split(",") if t.strip()
                    ]
                else:
                    body_tokens.append(tok)

            body = " ".join(body_tokens).strip()

            if not body:
                msg = (
                    "I didn't see any note text to ingest.\n\n"
                    "Example:\n"
                    "`/brain ingest project=Casterline tags=allowance,plumbing "
                    "Toilet allowance is $350...`"
                )
            else:
                req = IngestReq(
                    text=body,
                    project=project,
                    source=f"slack:{user_id}",
                    tags=tags_list or None,
                )
                result = ingest(req)
                if isinstance(result, dict) and result.get("ok"):
                    proj_name = req.project or "General"
                    msg = (
                        f"Saved note into project \"{proj_name}\" "
                        f"(id: {result.get('id')})."
                    )
                else:
                    msg = f"Ingest error: {result}"
    except Exception as e:
        msg = f"Stately Brain error while ingesting: {e}"

    try:
        httpx.post(
            response_url,
            json={"response_type": "ephemeral", "text": msg},
            timeout=30.0,
        )
    except Exception as e:
        print("SLACK SEND ERROR (INGEST):", repr(e))


def _ingest_slack_file(event: dict):
    """
    Background job for file uploads:
    - Download the file from Slack
    - Extract text with _read_text_from_upload
    - Ingest chunks into LanceDB
    - Post a confirmation message to the channel
    """
    if not SLACK_BOT_TOKEN:
        print("SLACK FILE INGEST SKIPPED: SLACK_BOT_TOKEN not set")
        return

    try:
        file_id = event.get("file_id") or event.get("file", {}).get("id")
        channel_id = event.get("channel_id") or event.get("channel")
        user_id = event.get("user_id") or event.get("user")

        if not file_id or not channel_id:
            print("SLACK FILE INGEST: missing file_id or channel_id:", event)
            return

        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}

        # 1) Look up file info
        info_resp = httpx.get(
            "https://slack.com/api/files.info",
            params={"file": file_id},
            headers=headers,
            timeout=30.0,
        )
        info = info_resp.json()
        if not info.get("ok"):
            print("files.info error:", info)
            return

        file_obj = info.get("file", {})
        download_url = file_obj.get("url_private_download")
        filename = file_obj.get("name") or file_obj.get("title") or "slack_file"
        mimetype = file_obj.get("mimetype") or ""

        if not download_url:
            print("No url_private_download for file:", file_id)
            return

        # 2) Download file bytes
        file_resp = httpx.get(download_url, headers=headers, timeout=60.0)
        data = file_resp.content

        # 3) Only handle text-like docs for now
        lower_name = filename.lower()
        if not (
            lower_name.endswith(".pdf")
            or lower_name.endswith(".txt")
            or lower_name.endswith(".md")
            or "text" in mimetype
        ):
            print("Skipping non-text file:", filename, mimetype)
            return

        text = _read_text_from_upload(filename, data)
        if not text:
            print("No text extracted from file:", filename)
            return

        # Use channel id as project label
        proj_name = f"slack:{channel_id}"

        saved = []
        for part in chunk_text(text):
            if not part.strip():
                continue

            vec = embed([part])[0]
            row = {
                "id": str(uuid.uuid4()),
                "text": part,
                "project": proj_name,
                "source": filename,
                "tags": ["slack-upload"],
                "vector": vec,
            }
            t.add([row])
            saved.append({"id": row["id"], "source": row["source"]})

        # 4) Post confirmation message back to the channel
        try:
            httpx.post(
                "https://slack.com/api/chat.postMessage",
                headers=headers,
                json={
                    "channel": channel_id,
                    "text": (
                        f"Ingested *{filename}* into project `{proj_name}` "
                        f"({len(saved)} chunks)."
                    ),
                },
                timeout=30.0,
            )
        except Exception as e:
            print("SLACK CONFIRM SEND ERROR:", repr(e))

    except Exception as e:
        print("SLACK FILE INGEST ERROR:", repr(e))


@app.post("/slack/command")
async def slack_command(request: Request, background_tasks: BackgroundTasks):
    """
    Handle the /brain slash command.

    Modes:
    - `/brain <question>`          → Q&A via ask()
    - `/brain ingest ...`          → ingest text into memory
    """
    form = await request.form()

    text = (form.get("text") or "").strip()
    user_id = form.get("user_id") or ""
    response_url = form.get("response_url") or ""

    if not text:
        return {
            "response_type": "ephemeral",
            "text": (
                "Use `/brain` followed by a question, e.g. "
                "`/brain summarize the latest client notes`,\n"
                "or ingest notes with `/brain ingest project=Name your note...`."
            ),
        }

    if not response_url:
        return {
            "response_type": "ephemeral",
            "text": (
                "Slack did not provide a response_url, so I can't send the full "
                "answer back."
            ),
        }

    first_word = text.split()[0].lower()
    if first_word == "ingest":
        background_tasks.add_task(_run_ingest_and_reply, text, response_url, user_id)
        ack_msg = "Got it, ingesting this note into Stately Brain..."
    else:
        background_tasks.add_task(_run_brain_and_reply, text, response_url, user_id)
        ack_msg = f"Got it, thinking about: `{text}`"

    return {"response_type": "ephemeral", "text": ack_msg}


@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    # 1) Slack URL verification
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    event = payload.get("event") or {}
    etype = event.get("type")
    subtype = event.get("subtype")

    # Minimal logging so we can see what Slack is actually sending
    try:
        print(
            "SLACK EVENT TYPE:",
            etype,
            "SUBTYPE:",
            subtype,
            "KEYS:",
            list(event.keys()),
        )
    except Exception as e:
        print("SLACK EVENT LOG ERROR:", repr(e))

    # 2) Handle pure file_shared events (from the file_shared subscription)
    if etype == "file_shared":
        background_tasks.add_task(_ingest_slack_file, event)
        return {"ok": True}

    # 3) Handle message events that include a file (subtype=file_share)
    if etype == "message" and subtype == "file_share":
        files = event.get("files") or []
        if files:
            file_obj = files[0]
            file_id = file_obj.get("id")
            channel_id = event.get("channel")
            user_id = event.get("user") or event.get("user_id")

            ingest_event = {
                "file_id": file_id,
                "channel_id": channel_id,
                "user_id": user_id,
            }
            background_tasks.add_task(_ingest_slack_file, ingest_event)
        return {"ok": True}

    # 4) Everything else (mentions etc.) – just ack for now
    return {"ok": True}

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow everything during development
    allow_credentials=True,
    allow_methods=["*"],  # fixes the OPTIONS error
    allow_headers=["*"],
)


# ---------- Pydantic models ----------
class IngestReq(BaseModel):
    text: str
    project: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = []


class AskReq(BaseModel):
    question: str
    project: Optional[str] = None
    k: int = 6


# ---------- Basic health / diag ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/diag")
def diag():
    try:
        v = embed(["hi"])[0]
        return {"openai_ok": True, "len": len(v)}
    except Exception as e:
        return {"openai_ok": False, "error": str(e)}


# ---------- /ingest ----------
@app.post("/ingest")
def ingest(req: IngestReq):
    try:
        vec = embed([req.text])[0]
        row = {
            "id": str(uuid.uuid4()),
            "text": req.text,
            "project": req.project or "General",
            "source": req.source or "",
            "tags": req.tags or [],
            "vector": vec,
        }
        t.add([row])
        return {"ok": True, "id": row["id"]}
    except Exception as e:
        print("INGEST ERROR:", repr(e))
        return {"error": str(e)}


# ---------- helper: build full text context from hits ----------
def build_context(hits: List[dict]) -> str:
    """Turn LanceDB hits into a single context string for the LLM."""
    parts = []
    for i, h in enumerate(hits, start=1):
        src = h.get("source", "unknown")
        proj = h.get("project", "")
        text = h.get("text", "")
        parts.append(f"[{i}] Project: {proj} | Source: {src}\n{text}")
    return "\n\n---\n\n".join(parts)


# ---------- /ask ----------
@app.post("/ask")
def ask(req: AskReq):
    try:
        project = req.project or "General"
        qvec = embed([req.question])[0]

        # 1) search inside the project first
        q = t.search(qvec, vector_column_name="vector")
        q = q.where(f"project = '{project}'")
        q = q.limit(max(1, min(req.k, 20)))
        hits = q.to_list()

        # 2) if no hits, fall back to all projects
        if not hits:
            q = t.search(qvec, vector_column_name="vector")
            q = q.limit(max(1, min(req.k, 20)))
            hits = q.to_list()
            if not hits:
                return {
                    "answer_draft": "I couldn't find anything matching that question.",
                    "citations": [],
                }

        # 3) build context text from hits
        context = build_context(hits)

        # 4) ask OpenAI to answer from that context
        prompt = f"""
You are Stately's estimating assistant.

User question:
{req.question}

Here are project notes and estimate excerpts:

{context}

Answer using ONLY the information in the context above.

Rules:
1. Never guess or estimate numbers. Only use quantities, prices, dates, and names that are explicitly written in the context.
2. If the question is about tile quantities or square footage, list each relevant line item separately with its quantity and units exactly as shown. 
   Do NOT add them together unless the user explicitly asks for a total.
3. If multiple different numbers could apply, list them all and explain what each one refers to.
4. If you are asked whether the estimate is signed, treat the presence of the client's name and a date near the end of the document as an indication that it IS signed. 
   In that case answer like: "Yes, it appears signed by David Casterline on Jun 16, 2025."
5. If the context does not give you enough information, say you are not sure instead of guessing.

Give a short, direct answer.
"""

        resp = oa.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        # Take the first text output from the Responses API
        answer = resp.output[0].content[0].text

        return {
            "answer_draft": answer.strip(),
            "citations": hits,
        }

    except Exception as e:
        print("ASK ERROR:", repr(e))
        return {"error": str(e)}


# ---------- /projects ----------
@app.get("/projects")
def list_projects():
    """
    Return the distinct list of project names currently in the LanceDB table.
    Used by chat.html to populate the project dropdown.
    """
    try:
        df = t.to_pandas()

        if df.empty or "project" not in df.columns:
            return {"projects": []}

        raw_projects = df["project"].fillna("General").tolist()
        projects = sorted({p.strip() for p in raw_projects if p and p.strip()})

        return {"projects": projects}
    except Exception as e:
        print("PROJECTS ERROR:", repr(e))
        return {"error": str(e), "projects": []}


# ---------- helper: split large text into chunks ----------
def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """
    Split a long string into chunks of at most max_chars characters.

    We use characters as a rough proxy for tokens. 6,000 characters keeps us
    safely under the ~8,192 token limit of text-embedding-3-small even for
    messy text.
    """
    text = text or ""
    if not text:
        return []
    return [text[i: i + max_chars] for i in range(0, len(text), max_chars)]


# ---------- helper: read text from a single uploaded file ----------
def _read_text_from_upload(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    # Simple heuristics by extension
    if name.endswith(".pdf"):
        try:
            import io

            reader = PdfReader(io.BytesIO(data))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages).strip()
        except Exception:
            # Fallback: store a note saying PDF couldn't be parsed
            return "(PDF text could not be extracted)"
    else:
        # .txt, .md, .csv, etc. — best effort decode
        try:
            return data.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""


# ---------- /memory/upload (single or multiple files) ----------
@app.post("/memory/upload")
async def memory_upload(
    files: List[UploadFile] = File(
        ..., description="One or more .txt/.md/.pdf files to ingest"
    ),
    project: Optional[str] = Form(
        None, description='Project name, e.g., "Morgan_Kitchen" (defaults to "General")'
    ),
    tags: Optional[str] = Form(
        None, description='Comma-separated tags, e.g., "schedule,cabinets"'
    ),
):
    saved = []
    proj = project or "General"
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] or []

    try:
        for f in files:
            raw = await f.read()
            text = _read_text_from_upload(f.filename, raw)
            if not text:
                continue

            # Chunk the text so we don't exceed the embedding model's context
            for part in chunk_text(text):
                if not part.strip():
                    continue

                vec = embed([part])[0]
                row = {
                    "id": str(uuid.uuid4()),
                    "text": part,
                    "project": proj,
                    "source": f.filename or "(upload)",
                    "tags": tag_list,
                    "vector": vec,
                }
                t.add([row])
                saved.append({"id": row["id"], "source": row["source"]})

        return {"ok": True, "count": len(saved), "saved": saved}
    except Exception as e:
        print("UPLOAD ERROR:", repr(e))
        return {"error": str(e)}


# ---------- /memory/upload-folder (ZIP of a customer folder) ----------
@app.post("/memory/upload-folder")
async def memory_upload_folder(
    zip_file: UploadFile = File(
        ..., description="A .zip containing .txt/.md/.pdf files (customer folder export)"
    ),
    project: Optional[str] = Form(
        None, description='Project name, e.g., "Casterline David" (defaults to zip name)'
    ),
    tags: Optional[str] = Form(
        None, description='Comma-separated tags, e.g., "2026, customers"'
    ),
):
    """
    Upload a ZIP of customer files, unzip server-side, and ingest each file.
    Large files are chunked so we never exceed OpenAI token limits.
    """
    import io

    saved = []
    raw_zip = await zip_file.read()

    # Base project name: explicit project override, else zip filename without .zip
    proj = project or (zip_file.filename or "General").replace(".zip", "")
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] or []

    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
            for info in z.infolist():
                # Skip directories
                if info.is_dir():
                    continue

                fname = info.filename
                # Optional: ignore hidden/system files
                if fname.startswith("__MACOSX") or fname.endswith(".DS_Store"):
                    continue

                data = z.read(info)
                text = _read_text_from_upload(fname, data)
                if not text:
                    continue

                # Chunk the text so we don't hit the token limit
                for part in chunk_text(text):
                    if not part.strip():
                        continue

                    vec = embed([part])[0]
                    row = {
                        "id": str(uuid.uuid4()),
                        "text": part,
                        "project": proj,
                        "source": fname,
                        "tags": tag_list,
                        "vector": vec,
                    }
                    t.add([row])
                    saved.append({"id": row["id"], "source": row["source"]})

        return {"ok": True, "count": len(saved), "saved": saved}

    except Exception as e:
        print("UPLOAD-FOLDER ERROR:", repr(e))
        return {"error": str(e)}


# ---------- /memory/reset-project (delete all rows for a project) ----------
@app.delete("/memory/reset-project")
def reset_project(
    project: str = Query(
        ...,
        description="Exact project name to wipe, e.g., 'Casterline David'",
    )
):
    """
    Delete all memory rows for a single project.

    Call:
      DELETE /memory/reset-project?project=Casterline%20David
    """
    try:
        # Escape single-quotes in the project name for the where clause
        safe_project = project.replace("'", "''")
        where_expr = f"project = '{safe_project}'"

        # LanceDB delete by filter expression
        t.delete(where=where_expr)

        return {"ok": True, "project": project}
    except Exception as e:
        print("RESET-PROJECT ERROR:", repr(e))
        return {"error": str(e), "project": project}
