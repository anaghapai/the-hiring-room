from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.auth import router as auth_router
from app.resume import router as resume_router
from app.panel import router as panel_router
from app.voice import router as voice_router
from app.dashboard import router as dashboard_router
from app.dsa import router as dsa_router

app = FastAPI(title="The Hiring Room API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before you ship anywhere real
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ok", "service": "the-hiring-room-api"}


app.include_router(auth_router)
app.include_router(resume_router)
app.include_router(panel_router)
app.include_router(voice_router)
app.include_router(dashboard_router)
app.include_router(dsa_router)
