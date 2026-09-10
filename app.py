```python
import os
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
DB_FILE = BASE_DIR / "memory.db"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("NVIDIA_API_KEY")

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

if not API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY was not found.\n"
        f"Make sure your .env file exists at:\n{ENV_FILE}"
    )

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=180.0,
)

app = FastAPI(title="UPI Fraud Help Chatbot")


# ============================================================
# INKROACH PROFILE
# ============================================================

INKROACH_PROFILE = """
Inkroach is an art streamer and illustrator.

Inkroach is a published illustrator with Qissa Comics.

Inkroach has participated in major art and manga-related contests,
including the Silent Manga Audition and KADOKAWA contests.

Inkroach was also shortlisted by Shueisha for one of his contest entries.

Inkroach creates art content and streams art. He also helps other
artists by giving art tips, demonstrations, explanations, and
educational resources.

Official social accounts:

YouTube:
www.youtube.com/@Inkroach

Instagram:
www.instagram.com/Inkroach/
"""


def is_inkroach_question(message: str):
    """
    Detect questions about Inkroach.

    This is handled locally instead of relying on the AI model
    to remember Inkroach's profile.
    """

    t = message.lower().strip()

    # Direct mentions of Inkroach.
    if "inkroach" not in t:
        return False

    question_patterns = [
        "who is",
        "who's",
        "tell me about",
        "what is",
        "what's",
        "what does",
        "about",
        "artist",
        "illustrator",
        "streamer",
        "art streamer",
        "qissa comics",
        "shueisha",
        "kadokawa",
        "silent manga",
        "youtube",
        "instagram",
        "social media",
    ]

    return any(pattern in t for pattern in question_patterns)


# ============================================================
# STATIC FILES
# ============================================================

STATIC_DIR = BASE_DIR / "static"

if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def now():
    return datetime.now(timezone.utc).isoformat()


def init_database():
    conn = get_connection()
    cur = conn.cursor()

    # Old memory table — kept so existing installations don't break.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL
        )
    """)

    # Long-term memory
    cur.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL DEFAULT 'general',
            memory TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 5,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_used TEXT
        )
    """)

    # Conversation history
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id, id)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_memory_importance
        ON long_term_memories(importance DESC)
    """)

    conn.commit()
    conn.close()

    migrate_old_memories()


def migrate_old_memories():
    """
    Copies memories from the original 'memories' table
    into the new long-term memory system.
    """

    conn = get_connection()
    cur = conn.cursor()

    old_memories = cur.execute(
        "SELECT memory FROM memories"
    ).fetchall()

    for row in old_memories:
        text = normalize_memory(row["memory"])

        if text:
            add_long_term_memory(text, conn=conn)

    conn.commit()
    conn.close()


# ============================================================
# MEMORY SYSTEM
# ============================================================

def normalize_memory(text: str):
    text = text.strip()

    while text.startswith(("'", '"')):
        text = text[1:].strip()

    while text.endswith(("'", '"')):
        text = text[:-1].strip()

    return text


def detect_category(text: str):
    t = text.lower()

    personal_words = [
        "my name",
        "i am",
        "i'm",
        "my age",
        "i live",
        "i'm from",
        "my birthday",
    ]

    preference_words = [
        "i like",
        "i love",
        "i hate",
        "my favorite",
        "i prefer",
        "i don't like",
        "i dislike",
    ]

    project_words = [
        "project",
        "building",
        "coding",
        "programming",
        "app",
        "website",
        "chatbot",
        "bot",
        "software",
    ]

    school_words = [
        "school",
        "class",
        "exam",
        "subject",
        "homework",
        "teacher",
        "study",
        "studying",
    ]

    art_words = [
        "art",
        "drawing",
        "anime",
        "manga",
        "artist",
        "sketch",
        "painting",
    ]

    if any(x in t for x in personal_words):
        return "personal"

    if any(x in t for x in preference_words):
        return "preference"

    if any(x in t for x in project_words):
        return "project"

    if any(x in t for x in school_words):
        return "school"

    if any(x in t for x in art_words):
        return "art"

    return "general"


def detect_importance(text: str):
    t = text.lower()

    if any(x in t for x in [
        "my name",
        "my age",
        "my birthday",
        "i live",
        "i'm from",
    ]):
        return 10

    if any(x in t for x in [
        "my favorite",
        "i love",
        "i hate",
        "i prefer",
        "i don't like",
    ]):
        return 8

    if any(x in t for x in [
        "project",
        "building",
        "coding",
        "programming",
        "learning",
        "studying",
    ]):
        return 7

    return 5


def add_long_term_memory(text: str, conn=None):
    text = normalize_memory(text)

    if not text:
        return False

    own_connection = conn is None

    if own_connection:
        conn = get_connection()

    cur = conn.cursor()

    category = detect_category(text)
    importance = detect_importance(text)
    timestamp = now()

    # Case-insensitive duplicate check
    existing = cur.execute(
        """
        SELECT id
        FROM long_term_memories
        WHERE LOWER(memory) = LOWER(?)
        LIMIT 1
        """,
        (text,)
    ).fetchone()

    if existing:
        cur.execute(
            """
            UPDATE long_term_memories
            SET updated_at = ?,
                importance = MAX(importance, ?)
            WHERE id = ?
            """,
            (timestamp, importance, existing["id"])
        )
    else:
        cur.execute(
            """
            INSERT INTO long_term_memories
            (
                category,
                memory,
                importance,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                category,
                text,
                importance,
                timestamp,
                timestamp,
            )
        )

    if own_connection:
        conn.commit()
        conn.close()

    return True


def get_all_memories():
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            id,
            category,
            memory,
            importance,
            created_at,
            updated_at,
            last_used
        FROM long_term_memories
        ORDER BY importance DESC, updated_at DESC
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_relevant_memories(query: str, limit=8):
    """
    Simple relevance search.

    It does not use embeddings yet.
    It scores memories based on shared words + importance.
    """

    memories = get_all_memories()

    if not memories:
        return []

    query_words = {
        word.lower().strip(".,!?;:\"'()[]{}")
        for word in query.split()
        if len(word) >= 3
    }

    scored = []

    for memory in memories:
        memory_words = {
            word.lower().strip(".,!?;:\"'()[]{}")
            for word in memory["memory"].split()
            if len(word) >= 3
        }

        overlap = len(query_words & memory_words)

        score = overlap * 10

        # Importance gives important memories a slight advantage.
        score += memory["importance"] * 0.5

        # If the query contains the category name, boost it.
        if memory["category"].lower() in query.lower():
            score += 5

        scored.append((score, memory))

    scored.sort(
        key=lambda x: (
            x[0],
            x[1]["importance"],
            x[1]["updated_at"],
        ),
        reverse=True
    )

    selected = [item[1] for item in scored[:limit]]

    # Mark memories as recently used.
    if selected:
        conn = get_connection()
        timestamp = now()

        for memory in selected:
            conn.execute(
                """
                UPDATE long_term_memories
                SET last_used = ?
                WHERE id = ?
                """,
                (timestamp, memory["id"])
            )

        conn.commit()
        conn.close()

    return selected


def build_memory_text(query: str):
    memories = get_relevant_memories(query)

    if not memories:
        return "No relevant long-term memories were found."

    lines = []

    for memory in memories:
        lines.append(
            f"- [{memory['category']}] {memory['memory']}"
        )

    return "\n".join(lines)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def save_message(session_id: str, role: str, content: str):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO messages
        (
            session_id,
            role,
            content,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            now(),
        )
    )

    conn.commit()
    conn.close()


def get_recent_messages(session_id: str, limit=20):
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            session_id,
            limit
        )
    ).fetchall()

    conn.close()

    # We selected newest first, so reverse it.
    rows = list(reversed(rows))

    return [
        {
            "role": row["role"],
            "content": row["content"],
        }
        for row in rows
    ]


# ============================================================
# SESSION
# ============================================================

def get_session_id(request: Request):
    session_id = request.headers.get("X-Session-ID")

    if not session_id:
        session_id = str(uuid.uuid4())

    return session_id


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(memory_text: str):
    return f"""
You are the UPI Fraud Help Chatbot.

You were created by Grinchy.

You are a helpful, friendly AI assistant focused primarily on
UPI scams, digital-payment fraud awareness, online safety,
and general assistance.

You have two types of memory:

1. Conversation history
   - This is the recent conversation in the current session.

2. Long-term memory
   - These are persistent facts/preferences/projects that may be useful later.

IMPORTANT MEMORY RULES:
- Use memories only when relevant.
- Never invent a memory.
- Never claim to remember something that is not present.
- Do not mention the database or internal memory system unless the user asks about it.
- If a memory conflicts with what the user currently says, prefer the user's current statement.
- Do not unnecessarily repeat memories.
- Answer naturally.

IMPORTANT CREATOR INFORMATION:
- Grinchy is the creator of this specific chatbot.
- If asked who created, made, built, or developed this chatbot, say that
  Grinchy created it.
- This chatbot uses NVIDIA's model/API underneath, but NVIDIA did not
  create this specific chatbot.
- Do not incorrectly claim that NVIDIA created this chatbot.

IMPORTANT INKROACH INFORMATION:
- Inkroach is an art streamer and illustrator.
- Inkroach is a published illustrator with Qissa Comics.
- Inkroach has participated in the Silent Manga Audition and KADOKAWA
  contests.
- Inkroach was shortlisted by Shueisha for one of his contest entries.
- Inkroach creates art content and streams art.
- Inkroach also helps other artists through art tips, demonstrations,
  explanations, and educational resources.
- If asked about Inkroach, use this information and do not invent
  additional achievements or credentials.

LONG-TERM MEMORIES:
{memory_text}

You can help with:
- UPI and digital-payment fraud awareness
- scam identification
- online safety
- coding
- school
- science
- technology
- art
- creativity
- general questions

Be clear and practical.
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        return HTMLResponse(
            """
            <h1>UPI Fraud Help Chatbot</h1>
            <p>static/index.html was not found.</p>
            """,
            status_code=500,
        )

    return HTMLResponse(
        index_file.read_text(encoding="utf-8")
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "database": str(DB_FILE),
        "api_key_loaded": bool(API_KEY),
    }


# ============================================================
# VIEW MEMORIES
# ============================================================

@app.get("/memories")
async def memories():
    data = get_all_memories()

    return {
        "count": len(data),
        "memories": data,
    }


# ============================================================
# TEST NVIDIA CONNECTION
# ============================================================

@app.get("/test-api")
async def test_api():
    """
    Simple non-streaming NVIDIA API test.

    This is useful because it separates:
    NVIDIA connectivity problems
    from
    FastAPI/frontend/streaming problems.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: NVIDIA API connection works."
                }
            ],
            temperature=0.2,
            max_tokens=30,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )

        text = response.choices[0].message.content

        return {
            "success": True,
            "model": MODEL,
            "response": text,
        }

    except Exception as error:
        print("\n================ NVIDIA API TEST ERROR ================")
        print(repr(error))
        print("=========================================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
async def chat(request: Request):

    # --------------------------------------------------------
    # Read request
    # --------------------------------------------------------

    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON request."}
        )

    message = str(data.get("message", "")).strip()

    if not message:
        return JSONResponse(
            status_code=400,
            content={"error": "Please type a message."}
        )

    session_id = get_session_id(request)

    print("\n================================================")
    print("NEW MESSAGE")
    print("Session:", session_id)
    print("User:", message)
    print("================================================")


    # --------------------------------------------------------
    # MANUAL MEMORY COMMAND
    # --------------------------------------------------------

    if message.lower().startswith("remember "):

        memory = message[9:].strip()

        if not memory:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Tell me what you want me to remember."
                }
            )

        try:
            add_long_term_memory(memory)

            save_message(
                session_id,
                "user",
                message
            )

            reply = f"🧠 Memory saved!\n\n{memory}"

            save_message(
                session_id,
                "assistant",
                reply
            )

            async def memory_stream():
                yield reply

            response = StreamingResponse(
                memory_stream(),
                media_type="text/plain"
            )

            response.headers["X-Session-ID"] = session_id

            return response

        except Exception as error:
            print("\n================ MEMORY ERROR ================")
            print(repr(error))
            print("==============================================\n")

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Could not save memory.",
                    "details": str(error),
                }
            )


    # --------------------------------------------------------
    # INKROACH PROFILE
    # --------------------------------------------------------
    #
    # This happens BEFORE NVIDIA.
    #
    # Therefore the model cannot hallucinate or forget
    # Inkroach's profile when directly asked about him.
    # --------------------------------------------------------

    if is_inkroach_question(message):

        try:
            save_message(
                session_id,
                "user",
                message
            )

            save_message(
                session_id,
                "assistant",
                INKROACH_PROFILE
            )

            async def inkroach_stream():
                yield INKROACH_PROFILE

            response = StreamingResponse(
                inkroach_stream(),
                media_type="text/plain"
            )

            response.headers["X-Session-ID"] = session_id

            return response

        except Exception as error:

            print("\n================ INKROACH PROFILE ERROR ================")
            print(repr(error))
            print("==========================================================\n")

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Could not load Inkroach profile.",
                    "details": str(error),
                }
            )


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    try:
        save_message(
            session_id,
            "user",
            message
        )
    except Exception as error:
        print("\n================ DATABASE ERROR ================")
        print(repr(error))
        print("=================================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not save conversation.",
                "details": str(error),
            }
        )


    # --------------------------------------------------------
    # LOAD MEMORY + HISTORY
    # --------------------------------------------------------

    try:
        memory_text = build_memory_text(message)

        history = get_recent_messages(
            session_id,
            limit=20
        )

    except Exception as error:
        print("\n================ MEMORY/HISTORY ERROR ================")
        print(repr(error))
        print("=======================================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not load memory/history.",
                "details": str(error),
            }
        )


    # --------------------------------------------------------
    # BUILD MODEL MESSAGES
    # --------------------------------------------------------

    system_prompt = build_system_prompt(
        memory_text
    )

    model_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # History already contains the current user message.
    model_messages.extend(history)


    # --------------------------------------------------------
    # STREAM NVIDIA RESPONSE
    # --------------------------------------------------------

    async def generate():

        full_response = ""

        try:

            print("\nSending request to NVIDIA...")
            print("Model:", MODEL)
            print("History messages:", len(history))
            print("Relevant memory:")
            print(memory_text)

            stream = client.chat.completions.create(
                model=MODEL,
                messages=model_messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=2048,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
            )

            print("NVIDIA connection established.")

            for chunk in stream:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                content = getattr(
                    delta,
                    "content",
                    None
                )

                if content:
                    full_response += content
                    yield content

            print("\nNVIDIA response completed.")

            # Save complete assistant response.
            if full_response.strip():
                save_message(
                    session_id,
                    "assistant",
                    full_response
                )

        except Exception as error:

            print("\n")
            print("======================================================")
            print("❌ NVIDIA API ERROR")
            print("Error type:", type(error).__name__)
            print("Error:", repr(error))
            print("======================================================")
            print("\n")

            error_message = (
                "\n\n⚠️ NVIDIA API error:\n"
                f"{type(error).__name__}: {error}"
            )

            yield error_message


    response = StreamingResponse(
        generate(),
        media_type="text/plain"
    )

    response.headers["X-Session-ID"] = session_id

    return response


# ============================================================
# START SERVER
# ============================================================

init_database()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
```
