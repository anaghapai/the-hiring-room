from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field
from pydantic import BaseModel


# ---------- DB TABLES ----------

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str = Field(index=True, unique=True)
    # Optional: accounts created via "Sign in with Google" have no password
    # at all — is_google_user marks them so /auth/login can reject a
    # password attempt cleanly instead of erroring on a missing hash.
    hashed_password: Optional[str] = None
    is_google_user: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ResumeAnalysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    filename: str
    match_score: float
    matched_points: str   # JSON-encoded list[str]
    gap_points: str        # JSON-encoded list[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InterviewSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    role_title: str = "Untitled Role"
    resume_analysis_id: Optional[int] = Field(default=None, index=True)  # which resume this session used, if any
    transcript: str = ""     # accumulated Q&A as plain text, used for quote verification
    status: str = "in_progress"  # in_progress | completed
    # NEW: which difficulty the candidate picked (easy | moderate | difficult) and which
    # UI language responses should come back in (en | hi | kn). Both were previously
    # silently dropped because this table/schema never declared them.
    difficulty: str = "moderate"
    language: str = "en"
    company: str = "general"
    domain: str = "general"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InterviewTurn(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(index=True)
    persona: str          # hr | tech | hm
    question: str
    answer: Optional[str] = None
    # Voice signal, filled in from /voice/analyze when the candidate answers by speaking.
    # All optional: text-only answers simply leave these unset.
    voice_wpm: Optional[float] = None
    voice_filler_count: Optional[int] = None
    voice_pause_count: Optional[int] = None
    # Browser-side face-tracking signal, self-reported by the candidate's own
    # device — not verified server-side. Optional: only present if they
    # enabled the camera self-check for that answer.
    eye_contact_pct: Optional[float] = None
    posture_note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Verdict(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(index=True)
    persona: str
    verdict: str            # strong | lean | discuss
    reasoning: str
    quote: str
    quote_verified: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InterviewFlag(SQLModel, table=True):
    # Integrity signals during an interview (left the tab, tried to paste an
    # answer). Previously the frontend sent these to a backend route that
    # didn't exist, so nothing was ever actually saved — the warning banner
    # on Results only reflected an in-memory counter that reset on reload.
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(index=True)
    kind: str  # "context_switch" | "paste_attempt"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------- REQUEST / RESPONSE SCHEMAS ----------

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ResumeAnalyzeResponse(BaseModel):
    id: int
    filename: str
    match_score: float
    matched_points: List[str]
    gap_points: List[str]


class StartInterviewRequest(BaseModel):
    role_title: str = "Untitled Role"
    resume_analysis_id: Optional[int] = None
    # These two existed on the frontend already but were never declared here,
    # so FastAPI silently dropped them before they ever reached panel.py.
    difficulty: str = "moderate"      # easy | moderate | difficult
    language: str = "en"              # en | hi | kn
    company: str = "general"          # general | google | amazon | meta | apple | microsoft | netflix
    domain: str = "general"           # general | backend | frontend | data_science | data_analyst | ml_engineer


class NextQuestionResponse(BaseModel):
    session_id: int
    turn_id: int
    persona: str
    question: str


class ActiveInterviewResponse(NextQuestionResponse):
    # Same shape the frontend already knows how to render (applyQuestion),
    # plus the role title so a "resume where you left off" banner can show
    # something more useful than just a session id.
    role_title: str


class SubmitAnswerRequest(BaseModel):
    session_id: int
    turn_id: int
    answer: str
    # Optional voice signal for this answer, from a prior POST /voice/analyze call.
    voice_wpm: Optional[float] = None
    voice_filler_count: Optional[int] = None
    voice_pause_count: Optional[int] = None
    eye_contact_pct: Optional[float] = None
    posture_note: Optional[str] = None


class FlagRequest(BaseModel):
    kind: str


class VerdictItem(BaseModel):
    persona: str
    verdict: str
    reasoning: str
    quote: str
    quote_verified: bool


class StarDetected(BaseModel):
    s: bool = False
    t: bool = False
    a: bool = False
    r: bool = False


class CoachingItem(BaseModel):
    question: str
    your_answer: str
    stronger_answer: str
    why_its_stronger: str
    # NEW: score (0-100) of the candidate's OWN answer against the ideal/"star"
    # answer, plus which individual STAR components (Situation/Task/Action/
    # Result) were actually detected in what the candidate said. For technical
    # (non-behavioral) questions the model is instructed to leave star_detected
    # all-false and score completeness/correctness instead.
    score: int = 0
    star_detected: StarDetected = StarDetected()
    # NEW (round 7): structured what-went-well / what-to-improve bullets,
    # missing key terms the answer should have used, a domain-knowledge score
    # judged by the model, and a communication score computed server-side
    # from REAL captured voice signal (filler words, pace) for that turn —
    # not guessed by the LLM. communication_score is None when the answer was
    # typed rather than spoken (no voice signal exists to compute from).
    what_went_well: List[str] = []
    what_to_improve: List[str] = []
    missing_terminologies: List[str] = []
    domain_knowledge_score: int = 0
    communication_score: Optional[int] = None


class DeliberationResponse(BaseModel):
    session_id: int
    verdicts: List[VerdictItem]
    transcript: str
    coaching: List[CoachingItem] = []
    context_switches: int = 0
    paste_attempts: int = 0


# ---------- DSA PRACTICE (new) ----------

class DsaQuestionOut(BaseModel):
    id: str
    title: str
    difficulty: str          # easy | medium | hard
    topic: str
    prompt: str               # problem statement, in the requested language
    examples: List[str] = []  # sample input/output lines, in the requested language


class DsaSubmitRequest(BaseModel):
    question_id: str
    prog_language: str = "python"   # python | java | cpp | javascript | c
    code: str
    debug_mode: bool = False
    language: str = "en"            # en | hi | kn — UI/response language


class DsaSubmitResponse(BaseModel):
    question_id: str
    debug_mode: bool
    score: int = 0                  # 0-100, only meaningful when debug_mode is False
    verdict: str = ""               # short verdict line, in the requested language
    feedback: str = ""              # fuller explanation, in the requested language
    hint: Optional[str] = None      # debug-mode only: a nudge, not the answer
    reference_approach: Optional[str] = None  # short description of an ideal approach
