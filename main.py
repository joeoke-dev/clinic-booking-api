"""Clinic booking API - Levels 1-4 (Postgres storage).

Endpoints (see koyaacademy.com/assessment-api):
  POST   /reset          replace all slots, delete all bookings
  GET    /slots          list slots (optional ?provider=...)
  POST   /bookings       book a slot
  GET    /bookings       list a person's bookings (?email=...&limit=...&cursor=...)
  GET    /bookings/{id}  look up a booking
  DELETE /bookings/{id}  cancel a booking

Environment variables (set both on Render):
  API_KEY       the key clients must send in the X-API-Key header
  DATABASE_URL  the Neon Postgres connection string
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from psycopg.errors import ForeignKeyViolation, UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("clinic")

# ---------------------------------------------------------------- auth (Level 2)
# Set API_KEY as an environment variable on Render. The default is for local testing only.
API_KEY = os.environ.get("API_KEY", "dev-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(request: Request, key: str = Depends(api_key_header)):
    """Reject any request without the right X-API-Key (the web page itself is exempt)."""
    if request.url.path == "/":
        return
    if not key or not hmac.compare_digest(key.encode(), API_KEY.encode()):
        raise StarletteHTTPException(401, "missing or invalid API key")


# ---------------------------------------------------------------- database
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Paste your Neon connection string into it.")

pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=10,
    open=False,
    check=ConnectionPool.check_connection,  # drop connections Neon closed while idle
    kwargs={"row_factory": dict_row, "prepare_threshold": None},
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS slots (
    id          text PRIMARY KEY,
    provider    text,
    start_time  text,
    end_time    text,
    start_at    timestamptz
);
CREATE TABLE IF NOT EXISTS bookings (
    seq         bigserial,
    id          text PRIMARY KEY,
    slot_id     text NOT NULL REFERENCES slots(id) ON DELETE CASCADE,
    name        text NOT NULL,
    email       text NOT NULL,
    status      text NOT NULL,
    created_at  text NOT NULL
);
-- The key to Level 3: a slot can have at most ONE confirmed booking.
-- The database enforces this, so it holds even for simultaneous requests.
CREATE UNIQUE INDEX IF NOT EXISTS one_confirmed_booking_per_slot
    ON bookings (slot_id) WHERE status = 'confirmed';
CREATE INDEX IF NOT EXISTS bookings_email_idx ON bookings (email, seq);
-- Level 4: remembers the response for each Idempotency-Key.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key          text PRIMARY KEY,
    body_hash    text NOT NULL,
    status_code  integer,
    response     text
);
"""

BOOKING_COLS = "id, slot_id, name, email, status, created_at"


@asynccontextmanager
async def lifespan(app):
    pool.open(wait=True)
    with pool.connection() as conn:
        conn.execute(SCHEMA)
    yield
    pool.close()


app = FastAPI(
    title="Clinic booking API",
    dependencies=[Depends(require_key)],
    lifespan=lifespan,
)

# Lets a page on another URL call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


# ---------------------------------------------------------------- helpers
def error(status, message, headers=None):
    return JSONResponse({"error": message}, status_code=status, headers=headers)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_dt(value):
    """Parse an ISO timestamp to an aware UTC datetime. Unreadable values sort last."""
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return datetime.max.replace(tzinfo=timezone.utc)


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


async def read_json(request: Request):
    """Return the parsed JSON body, or None if it isn't a JSON object."""
    try:
        body = await request.json()
    except Exception:
        return None
    return body if isinstance(body, dict) else None


# ---------------------------------------------------------------- database functions
# These are plain (blocking) functions; the routes run them in a thread pool.
def db_reset(slot_rows):
    with pool.connection() as conn:
        conn.execute("DELETE FROM idempotency_keys")
        conn.execute("DELETE FROM bookings")
        conn.execute("DELETE FROM slots")
        if slot_rows:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO slots (id, provider, start_time, end_time, start_at)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    slot_rows,
                )


def db_list_slots(provider):
    sql = (
        'SELECT s.id, s.provider, s.start_time AS "start", s.end_time AS "end",'
        " NOT EXISTS (SELECT 1 FROM bookings b"
        "             WHERE b.slot_id = s.id AND b.status = 'confirmed') AS available"
        " FROM slots s"
    )
    params = ()
    if provider is not None:
        sql += " WHERE s.provider = %s"
        params = (provider,)
    sql += " ORDER BY s.start_at, s.id"
    with pool.connection() as conn:
        return conn.execute(sql, params).fetchall()


def _try_book(conn, slot_id, name, email):
    """Attempt the booking. Returns (status_code, payload)."""
    slot = conn.execute("SELECT start_at FROM slots WHERE id = %s", (slot_id,)).fetchone()
    if slot is None:
        return 404, {"error": "slot not found"}
    if slot["start_at"] <= datetime.now(timezone.utc):
        return 400, {"error": "slot has already started"}
    try:
        with conn.transaction():  # a savepoint: a failed insert doesn't abort the outer transaction
            booking = conn.execute(
                "INSERT INTO bookings (id, slot_id, name, email, status, created_at)"
                f" VALUES (%s, %s, %s, %s, 'confirmed', %s) RETURNING {BOOKING_COLS}",
                (str(uuid.uuid4()), slot_id, name, email, now_iso()),
            ).fetchone()
    except UniqueViolation:  # someone else already holds this slot
        return 409, {"error": "slot is already booked"}
    except ForeignKeyViolation:  # slot was removed by a /reset in the meantime
        return 404, {"error": "slot not found"}
    return 201, booking


def db_create_booking(slot_id, name, email, idem_key):
    """Returns (status_code, payload).

    With an idempotency key, the first request claims the key and saves its response.
    A repeat of the same request gets that saved response back and creates nothing new.
    """
    body_hash = hashlib.sha256(json.dumps([slot_id, name, email]).encode()).hexdigest()
    with pool.connection() as conn:
        if idem_key:
            # If another request holds this key right now, this INSERT waits for it to finish.
            claimed = conn.execute(
                "INSERT INTO idempotency_keys (key, body_hash) VALUES (%s, %s)"
                " ON CONFLICT (key) DO NOTHING RETURNING key",
                (idem_key, body_hash),
            ).fetchone()
            if claimed is None:
                saved = conn.execute(
                    "SELECT body_hash, status_code, response FROM idempotency_keys WHERE key = %s",
                    (idem_key,),
                ).fetchone()
                if saved is None or saved["status_code"] is None:
                    return 409, {"error": "a request with this Idempotency-Key is still in progress"}
                if saved["body_hash"] != body_hash:
                    return 422, {"error": "Idempotency-Key was already used with a different request body"}
                return saved["status_code"], json.loads(saved["response"])

        status, payload = _try_book(conn, slot_id, name, email)

        if idem_key:
            conn.execute(
                "UPDATE idempotency_keys SET status_code = %s, response = %s WHERE key = %s",
                (status, json.dumps(payload), idem_key),
            )
    return status, payload


def encode_cursor(seq):
    return base64.urlsafe_b64encode(str(seq).encode()).decode().rstrip("=")


def decode_cursor(text):
    """Return the sequence number inside a cursor, or None if the cursor is invalid."""
    try:
        value = int(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode())
    except Exception:
        return None
    return value if 0 <= value < 2**62 else None


def db_list_bookings(email, limit, after_seq):
    with pool.connection() as conn:
        return conn.execute(
            f"SELECT seq, {BOOKING_COLS} FROM bookings"
            " WHERE email = %s AND seq > %s ORDER BY seq LIMIT %s",
            (email, after_seq, limit + 1),
        ).fetchall()


def db_get_booking(booking_id):
    with pool.connection() as conn:
        return conn.execute(
            f"SELECT {BOOKING_COLS} FROM bookings WHERE id = %s", (booking_id,)
        ).fetchone()


def db_cancel_booking(booking_id):
    """Returns (outcome, booking). outcome: ok | not_found | already."""
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT status FROM bookings WHERE id = %s FOR UPDATE", (booking_id,)
        ).fetchone()
        if row is None:
            return "not_found", None
        if row["status"] == "cancelled":
            return "already", None
        booking = conn.execute(
            f"UPDATE bookings SET status = 'cancelled' WHERE id = %s RETURNING {BOOKING_COLS}",
            (booking_id,),
        ).fetchone()
    return "ok", booking


# ---------------------------------------------------------------- errors
# Every error must look like {"error": "<message>"}.
@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    return error(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return error(400, "invalid request")


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    log.exception("unexpected error")
    return error(500, "internal server error")


# ---------------------------------------------------------------- routes
# We read request bodies by hand (so we control the error messages), which means
# FastAPI can't see them. These tell the /docs test page to show a body box.
def body_doc(example):
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"type": "object"}, "example": example}},
        }
    }


RESET_DOC = body_doc({"slots": [
    {"id": "s1", "provider": "Dr Ade", "start": "2026-12-10T09:00:00Z", "end": "2026-12-10T09:30:00Z"},
    {"id": "s2", "provider": "Dr Bisi", "start": "2026-12-10T08:00:00Z", "end": "2026-12-10T08:30:00Z"},
]})
BOOKING_DOC = body_doc({"slot_id": "s1", "name": "Ada Obi", "email": "ada@example.com"})


@app.post("/reset", openapi_extra=RESET_DOC)
async def reset(request: Request):
    body = await read_json(request)
    if body is None or not isinstance(body.get("slots"), list):
        return error(400, "body must be an object with a 'slots' list")

    rows = {}
    for s in body["slots"]:
        if not isinstance(s, dict) or not s.get("id"):
            return error(400, "each slot needs an id")
        sid = str(s["id"])
        rows[sid] = (
            sid,
            s.get("provider"),
            None if s.get("start") is None else str(s["start"]),
            None if s.get("end") is None else str(s["end"]),
            parse_dt(s.get("start")),
        )

    await run_in_threadpool(db_reset, list(rows.values()))
    return {"status": "ok"}


@app.get("/slots")
async def list_slots(provider: str = None):
    slots = await run_in_threadpool(db_list_slots, provider)
    return {"slots": slots}


@app.post("/bookings", openapi_extra=BOOKING_DOC)
async def create_booking(
    request: Request,
    idempotency_key: str = Header(None, alias="Idempotency-Key"),
):
    body = await read_json(request)
    if body is None:
        return error(400, "body must be a JSON object")

    slot_id = body.get("slot_id")
    name = body.get("name")
    email = body.get("email")
    if not all(isinstance(v, str) and v.strip() for v in (slot_id, name, email)):
        return error(400, "slot_id, name and email are required")
    name = name.strip()
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return error(400, "email is not valid")

    key = idempotency_key.strip() if idempotency_key else None
    if key and len(key) > 255:
        return error(400, "Idempotency-Key is too long")

    status, payload = await run_in_threadpool(db_create_booking, slot_id, name, email, key or None)
    return JSONResponse(payload, status_code=status)


@app.get("/bookings")
async def list_bookings(email: str = None, limit: str = None, cursor: str = None):
    if email is None or not email.strip():
        return error(400, "email is required")
    email = email.strip().lower()

    if limit is None or limit.strip() == "":
        n = 10
    else:
        if not re.fullmatch(r"-?[0-9]+", limit.strip()):
            return error(400, "limit must be a whole number from 1 to 50")
        n = int(limit.strip())
        if not 1 <= n <= 50:
            return error(400, "limit must be a whole number from 1 to 50")

    after = 0
    if cursor is not None and cursor.strip() != "":
        after = decode_cursor(cursor.strip())
        if after is None:
            return error(400, "invalid cursor")

    rows = await run_in_threadpool(db_list_bookings, email, n, after)
    next_cursor = None
    if len(rows) > n:  # we fetched one extra row to learn there is another page
        rows = rows[:n]
        next_cursor = encode_cursor(rows[-1]["seq"])
    bookings = [{k: v for k, v in r.items() if k != "seq"} for r in rows]
    return {"bookings": bookings, "next_cursor": next_cursor}


@app.get("/", include_in_schema=False)
async def home():
    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace('"__API_KEY__"', json.dumps(API_KEY)))


@app.get("/bookings/{booking_id}")
async def get_booking(booking_id: str):
    booking = await run_in_threadpool(db_get_booking, booking_id)
    if booking is None:
        return error(404, "booking not found")
    return booking


@app.delete("/bookings/{booking_id}")
async def cancel_booking(booking_id: str):
    outcome, booking = await run_in_threadpool(db_cancel_booking, booking_id)
    if outcome == "not_found":
        return error(404, "booking not found")
    if outcome == "already":
        return error(409, "booking is already cancelled")
    return booking
