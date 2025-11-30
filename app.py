import os
from dotenv import load_dotenv

# ---------- Load .env ----------
load_dotenv(dotenv_path=".env", override=True)

# app.py — Stately local “AI memory” with LanceDB + OpenAI
import uuid
from typing import List, Optional

import pyarrow as pa
import lancedb
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI

# ---------- OpenAI Client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env file")

oa = OpenAI(api_key=OPENAI_API_KEY)


def embed(texts: List[str]) -> List[List[float]]:
    """Return 1536-dim embeddings for a list of texts."""
    resp = oa.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


# ---------- LanceDB (explicit schema) ----------
DB_PATH = "stately_lancedb"
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

# ---------- FastAPI app + CORS ----------
app = FastAPI()

# ONLY ONE CORS BLOCK
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
                    "citations": []
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

Answer the user's question as a short, direct sentence using the numbers in the text 
(e.g. “The toilet allowance is $350.00.”). 
If you are not sure, say you are not sure.
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
        # Pull the whole table into a DataFrame
        df = t.to_pandas()

        if df.empty or "project" not in df.columns:
            return {"projects": []}

        # Clean and deduplicate project names
        raw_projects = df["project"].fillna("General").tolist()
        projects = sorted({p.strip() for p in raw_projects if p and p.strip()})

        return {"projects": projects}
    except Exception as e:
        print("PROJECTS ERROR:", repr(e))
        return {"error": str(e), "projects": []}
# ---------- /memory/upload ----------
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
            vec = embed([text])[0]
            row = {
                "id": str(uuid.uuid4()),
                "text": text,
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
        import zipfile
import tempfile
import shutil
import os

@app.post("/memory/upload-folder")
async def memory_upload_folder(
    zip_file: UploadFile = File(..., description="A ZIP file containing a full customer folder"),
    project: Optional[str] = Form(None, description="Project name override. Defaults to ZIP top folder name."),
    tags: Optional[str] = Form(None, description="Comma-separated tags")
):
    try:
        # Create temp folder
        temp_dir = tempfile.mkdtemp()

        # Save uploaded zip temporarily
        zip_path = os.path.join(temp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(await zip_file.read())

        # Extract ZIP content
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Determine project name
        root_items = [p for p in os.listdir(temp_dir) if p != "upload.zip"]
        top_folder = root_items[0] if root_items else "General"
        project_name = project or top_folder
        tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] or []

        saved = []

        # Walk all files
        for root, dirs, files in os.walk(temp_dir):
            for fname in files:
                full_path = os.path.join(root, fname)

                # Skip internal temp ZIP
                if fname == "upload.zip":
                    continue

                # Load file
                try:
                    with open(full_path, "rb") as f:
                        data = f.read()
                except:
                    continue

                # Extract text
                text = _read_text_from_upload(fname, data)
                if not text:
                    continue

                # Embed and save
                vec = embed([text])[0]
                row = {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "project": project_name,
                    "source": fname,
                    "tags": tag_list,
                    "vector": vec,
                }
                t.add([row])
                saved.append({"id": row["id"], "source": fname})

        # Clean up temp folder
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {"ok": True, "count": len(saved), "saved": saved, "project": project_name}

    except Exception as e:
        print("UPLOAD-FOLDER ERROR:", repr(e))
        return {"error": str(e)}