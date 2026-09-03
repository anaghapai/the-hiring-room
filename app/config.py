import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# gpt-oss-20b runs at ~1000 tokens/sec on Groq's hardware vs ~500 for the
# 120b version — for short outputs like a single interview question, that's
# the difference between a near-instant response and a multi-second wait,
# with negligible quality loss for this use case.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hiring_room.db")
# Google Cloud Console OAuth Client ID (Web application type). Get one at
# console.google.com/apis/credentials — leave blank to disable Google sign-in.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
