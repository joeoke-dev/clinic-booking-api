"""Clinic booking API - Level 1.

Endpoints (see koyaacademy.com/assessment-api):
  POST /reset          replace all slots, delete all bookings
  GET  /slots          list slots (optional ?provider=...)
  POST /bookings       book a slot
  GET  /bookings/{id}  look up a booking
"""
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI(title="Clinic booking API")

# The web page lives on a different URL, so the browser needs CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)

# ---------------------------------------------------------------- storage
# In-memory for now. We move this to a database at Level 3.
LOCK = threading.Lock()
SLOTS = {}     # slot id -> {"id", "provider", "start", "end"}
BOOKINGS = {}  # booking id -> booking dict


# ---------------------------------------------------------------- helpers
def error(status, message, headers=None):
    return JSONResponse({"error": message}, status_code=status, headers=headers)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_dt(value):
    """Parse an ISO timestamp to an aware UTC datetime (for sorting)."""
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return datetime.max.replace(tzinfo=timezone.utc)


def is_booked(slot_id):
    return any(
        b["slot_id"] == slot_id and b["status"] == "confirmed"
        for b in BOOKINGS.values()
    )


async def read_json(request: Request):
    """Return the parsed JSON body, or None if it isn't a JSON object."""
    try:
        body = await request.json()
    except Exception:
        return None
    return body if isinstance(body, dict) else None


# ---------------------------------------------------------------- errors
# Every error must look like {"error": "<message>"}.
@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    return error(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return error(400, "invalid request")


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

    new_slots = {}
    for s in body["slots"]:
        if not isinstance(s, dict) or not s.get("id"):
            return error(400, "each slot needs an id")
        new_slots[str(s["id"])] = {
            "id": str(s["id"]),
            "provider": s.get("provider"),
            "start": s.get("start"),
            "end": s.get("end"),
        }

    with LOCK:
        SLOTS.clear()
        SLOTS.update(new_slots)
        BOOKINGS.clear()
    return {"status": "ok"}


@app.get("/slots")
async def list_slots(provider: str = None):
    with LOCK:
        slots = [
            {**s, "available": not is_booked(s["id"])}
            for s in SLOTS.values()
            if provider is None or s["provider"] == provider
        ]
    slots.sort(key=lambda s: parse_dt(s["start"]))
    return {"slots": slots}


@app.post("/bookings", openapi_extra=BOOKING_DOC)
async def create_booking(request: Request):
    body = await read_json(request)
    if body is None:
        return error(400, "body must be a JSON object")

    slot_id = body.get("slot_id")
    name = body.get("name")
    email = body.get("email")
    if not all(isinstance(v, str) and v.strip() for v in (slot_id, name, email)):
        return error(400, "slot_id, name and email are required")

    with LOCK:
        if slot_id not in SLOTS:
            return error(404, "slot not found")
        booking = {
            "id": str(uuid.uuid4()),
            "slot_id": slot_id,
            "name": name.strip(),
            "email": email.strip().lower(),
            "status": "confirmed",
            "created_at": now_iso(),
        }
        BOOKINGS[booking["id"]] = booking
    return JSONResponse(booking, status_code=201)


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/bookings/{booking_id}")
async def get_booking(booking_id: str):
    with LOCK:
        booking = BOOKINGS.get(booking_id)
    if booking is None:
        return error(404, "booking not found")
    return booking
