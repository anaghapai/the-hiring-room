import json
from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.database import get_session
from app.auth import get_current_user
from app.models import User, InterviewSession, ResumeAnalysis, Verdict, InterviewTurn

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
def get_stats(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    sessions = session.exec(
        select(InterviewSession).where(InterviewSession.user_id == user.id)
    ).all()
    resumes = session.exec(
        select(ResumeAnalysis).where(ResumeAnalysis.user_id == user.id)
    ).all()

    history = []
    for s in sessions:
        verdicts = session.exec(select(Verdict).where(Verdict.session_id == s.id)).all()
        overall = "discuss"
        if verdicts:
            counts = {"strong": 0, "lean": 0, "discuss": 0}
            for v in verdicts:
                counts[v.verdict] = counts.get(v.verdict, 0) + 1
            overall = max(counts, key=counts.get)
        history.append({
            "session_id": s.id,
            "role_title": s.role_title,
            "status": s.status,
            "overall_verdict": overall,
            "created_at": s.created_at.isoformat(),
        })

    return {
        "sessions_completed": len([s for s in sessions if s.status == "completed"]),
        "resumes_analyzed": len(resumes),
        "history": sorted(history, key=lambda h: h["created_at"], reverse=True),
    }


@router.get("/session/{session_id}")
def get_session_detail(session_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    interview = session.get(InterviewSession, session_id)
    if not interview or interview.user_id != user.id:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")

    verdicts = session.exec(select(Verdict).where(Verdict.session_id == session_id)).all()
    return {
        "session_id": interview.id,
        "role_title": interview.role_title,
        "status": interview.status,
        "created_at": interview.created_at.isoformat(),
        "transcript": interview.transcript,
        "verdicts": [
            {
                "persona": v.persona,
                "verdict": v.verdict,
                "reasoning": v.reasoning,
                "quote": v.quote,
                "quote_verified": v.quote_verified,
            }
            for v in verdicts
        ],
    }


@router.get("/overview")
def get_overview(session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    """Aggregated, rule-based (not AI-generated) summary of strengths and
    improvement areas across every completed session — the frontend calls
    this for the Dashboard's 'general performance' panel, but this endpoint
    didn't exist before, so that panel always silently failed to load."""
    completed_sessions = session.exec(
        select(InterviewSession)
        .where(InterviewSession.user_id == user.id)
        .where(InterviewSession.status == "completed")
    ).all()

    if not completed_sessions:
        return {"summary": "", "strengths": [], "improve": [], "sessions_counted": 0}

    session_ids = [s.id for s in completed_sessions]

    turns = session.exec(
        select(InterviewTurn)
        .where(InterviewTurn.session_id.in_(session_ids))
        .where(InterviewTurn.answer.is_not(None))
    ).all()

    verdicts = session.exec(
        select(Verdict).where(Verdict.session_id.in_(session_ids))
    ).all()
    # "Not evaluated" placeholders (persona never got a question that
    # session) shouldn't count as a real "discuss" verdict when judging
    # overall pattern — they carry no actual opinion either way.
    real_verdicts = [v for v in verdicts if not v.reasoning.startswith("Not evaluated")]

    strengths, improve = [], []

    wpm_values = [t.voice_wpm for t in turns if t.voice_wpm]
    if wpm_values:
        avg_wpm = sum(wpm_values) / len(wpm_values)
        if 110 <= avg_wpm <= 160:
            strengths.append(f"Speaking pace averages {avg_wpm:.0f} wpm — a comfortable, easy-to-follow range.")
        elif avg_wpm < 110:
            improve.append(f"Speaking pace averages {avg_wpm:.0f} wpm, on the slower side — practising answers out loud can help build fluency.")
        else:
            improve.append(f"Speaking pace averages {avg_wpm:.0f} wpm, quite fast — slowing down slightly gives the panel more time to follow your reasoning.")

    filler_values = [t.voice_filler_count for t in turns if t.voice_filler_count is not None]
    if filler_values:
        avg_fillers = sum(filler_values) / len(filler_values)
        if avg_fillers < 1:
            strengths.append(f"Very few filler words per answer (avg {avg_fillers:.1f}).")
        else:
            improve.append(f"Averaging {avg_fillers:.1f} filler word(s) per answer — a silent pause instead of 'um'/'like' reads as more confident.")

    pause_values = [t.voice_pause_count for t in turns if t.voice_pause_count is not None]
    if pause_values:
        avg_pauses = sum(pause_values) / len(pause_values)
        if avg_pauses <= 2:
            strengths.append(f"Few long pauses per answer (avg {avg_pauses:.1f}).")
        else:
            improve.append(f"Averaging {avg_pauses:.1f} long pause(s) per answer — structuring answers (e.g. STAR: Situation/Task/Action/Result) before speaking can reduce these.")

    eye_values = [t.eye_contact_pct for t in turns if t.eye_contact_pct is not None]
    if eye_values:
        avg_eye = sum(eye_values) / len(eye_values)
        if avg_eye >= 70:
            strengths.append(f"Strong self-reported eye contact (avg {avg_eye:.0f}%).")
        else:
            improve.append(f"Self-reported eye contact averages {avg_eye:.0f}% — try positioning your camera at eye level and glancing at the lens, not the screen.")

    if real_verdicts:
        counts = {"strong": 0, "lean": 0, "discuss": 0}
        for v in real_verdicts:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
        total = sum(counts.values())
        strong_pct = round(counts["strong"] / total * 100)
        discuss_pct = round(counts["discuss"] / total * 100)
        if strong_pct >= 40:
            strengths.append(f"The panel has rated you 'Strong Hire' in {strong_pct}% of evaluations so far.")
        if discuss_pct >= 50:
            improve.append(f"The panel most often lands on 'Discuss' ({discuss_pct}% of evaluations) — specific, quantified examples tend to move this toward a clearer verdict.")

    return {
        "summary": f"Across {len(completed_sessions)} completed session(s), here's what's working and what to sharpen.",
        "strengths": strengths,
        "improve": improve,
        "sessions_counted": len(completed_sessions),
    }


@router.get("/session/{session_id}/performance")
def get_session_performance(session_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    """Per-section performance metrics: delivery (voice), body language, and verdict."""
    interview = session.get(InterviewSession, session_id)
    if not interview or interview.user_id != user.id:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")

    turns = session.exec(
        select(InterviewTurn).where(InterviewTurn.session_id == session_id)
    ).all()

    # Aggregate by persona (hr, tech, hm)
    persona_metrics = {}
    for persona_key in ["hr", "tech", "hm"]:
        persona_label = {"hr": "HR", "tech": "Tech Lead", "hm": "Hiring Manager"}[persona_key]
        turns_for_persona = [t for t in turns if t.persona == persona_key and t.answer is not None]
        
        if not turns_for_persona:
            continue

        wpm_values = [t.voice_wpm for t in turns_for_persona if t.voice_wpm]
        filler_values = [t.voice_filler_count for t in turns_for_persona if t.voice_filler_count is not None]
        pause_values = [t.voice_pause_count for t in turns_for_persona if t.voice_pause_count is not None]
        eye_contact_values = [t.eye_contact_pct for t in turns_for_persona if t.eye_contact_pct is not None]

        persona_metrics[persona_key] = {
            "label": persona_label,
            "turns_answered": len(turns_for_persona),
            "avg_wpm": round(sum(wpm_values) / len(wpm_values), 1) if wpm_values else None,
            "total_fillers": sum(filler_values) if filler_values else 0,
            "avg_pauses": round(sum(pause_values) / len(pause_values), 1) if pause_values else None,
            "avg_eye_contact_pct": round(sum(eye_contact_values) / len(eye_contact_values), 1) if eye_contact_values else None,
        }

    verdicts = session.exec(select(Verdict).where(Verdict.session_id == session_id)).all()
    section_verdicts = {v.persona: v.verdict for v in verdicts}

    return {
        "session_id": interview.id,
        "role_title": interview.role_title,
        "sections": {
            persona_key: {
                **persona_metrics[persona_key],
                "verdict": section_verdicts.get(persona_key, "discuss"),
            }
            for persona_key in persona_metrics
        },
    }
