import json
import io
import re
from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlmodel import Session, select

from app.database import get_session
from app.auth import get_current_user
from app.models import User, ResumeAnalysis, ResumeAnalyzeResponse

router = APIRouter(prefix="/resume", tags=["resume"])


class ResumeSummary(dict):
    pass


def _to_response(record: ResumeAnalysis) -> ResumeAnalyzeResponse:
    return ResumeAnalyzeResponse(
        id=record.id,
        filename=record.filename,
        match_score=record.match_score,
        matched_points=json.loads(record.matched_points),
        gap_points=json.loads(record.gap_points),
    )


@router.get("/latest", response_model=Optional[ResumeAnalyzeResponse])
def latest_resume(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    # Most recent resume this user has analyzed — used by the frontend to
    # offer "continue with the same resume" instead of forcing a re-upload.
    record = session.exec(
        select(ResumeAnalysis)
        .where(ResumeAnalysis.user_id == user.id)
        .order_by(ResumeAnalysis.created_at.desc())
    ).first()
    if not record:
        return None
    return _to_response(record)


@router.get("/{resume_id}", response_model=ResumeAnalyzeResponse)
def get_resume(
    resume_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    record = session.get(ResumeAnalysis, resume_id)
    if not record or record.user_id != user.id:
        raise HTTPException(status_code=404, detail="Resume analysis not found")
    return _to_response(record)


@lru_cache(maxsize=1)
def get_embedder():
    # Lazy import + lazy load: the model only downloads the first time this
    # is actually called, not at server startup.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(filename: str, raw: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if lower.endswith(".docx"):
        import docx
        doc = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    # fallback: treat as plain text
    return raw.decode("utf-8", errors="ignore")


def naive_requirement_split(jd_text: str) -> list:
    # Very simple heuristic: JD lines/bullets become "requirements" we can
    # check the resume against individually. Good enough for a v1 gap list —
    # AS LONG AS the JD actually has line breaks. If it was pasted as one
    # continuous paragraph (very common — most job sites strip bullet
    # formatting on copy), splitlines() returns exactly ONE giant
    # "requirement," which collapses the whole gap analysis into a single
    # all-or-nothing comparison: the match % becomes a coarse whole-resume-
    # vs-whole-JD score, that score itself is what decides "matched" or
    # "gap," and match=score>=0.35 for a single item can never coexist with
    # "gaps: none flagged" being meaningful — that's exactly the confusing
    # "44% match, zero gaps" result this was producing.
    lines = [l.strip("-•* \t") for l in jd_text.splitlines()]
    lines = [l for l in lines if len(l) > 8]
    if len(lines) >= 3:
        return lines[:12]

    sentences = re.split(r"(?<=[.!?])\s+", jd_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
    return (sentences or lines)[:12]


@router.post("/analyze", response_model=ResumeAnalyzeResponse)
def analyze_resume(
    jd_text: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    raw = file.file.read()
    resume_text = extract_text(file.filename, raw)
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from that file")

    embedder = get_embedder()
    requirements = naive_requirement_split(jd_text)
    if not requirements:
        requirements = [jd_text[:200]]

    resume_emb = embedder.encode([resume_text])[0]
    req_embs = embedder.encode(requirements)

    import numpy as np
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    scored = [(req, cosine(resume_emb, req_emb)) for req, req_emb in zip(requirements, req_embs)]
    scored.sort(key=lambda x: x[1], reverse=True)

    matched = [r for r, s in scored if s >= 0.35]
    gaps = [r for r, s in scored if s < 0.35]
    overall = sum(s for _, s in scored) / len(scored) if scored else 0.0

    record = ResumeAnalysis(
        user_id=user.id,
        filename=file.filename,
        match_score=round(overall * 100, 1),
        matched_points=json.dumps(matched),
        gap_points=json.dumps(gaps),
    )
    session.add(record)
    session.commit()
    session.refresh(record)

    return _to_response(record)
