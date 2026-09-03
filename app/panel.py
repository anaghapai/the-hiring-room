import json
import random
import re
from typing import List, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.database import get_session
from app.auth import get_current_user
from app.config import GROQ_API_KEY, GROQ_MODEL
from app.voice import summarize_signal
from app.models import (
    User, InterviewSession, InterviewTurn, Verdict, InterviewFlag,
    StartInterviewRequest, NextQuestionResponse, ActiveInterviewResponse,
    SubmitAnswerRequest, DeliberationResponse, VerdictItem, FlagRequest,
)

router = APIRouter(prefix="/interview", tags=["interview"])

PERSONAS = ["hr", "tech", "hm"]
PERSONA_LABEL = {"hr": "HR", "tech": "Tech Lead", "hm": "Hiring Manager"}

# Each persona gets its own rubric. This is the "few-shot instead of
# fine-tuning" approach: no training needed, the rubric + examples steer
# the model at request time.
PERSONA_RUBRIC = {
    "hr": """You are the HR evaluator on a hiring panel. You weigh: clear communication,
culture fit signals, and red flags (dishonesty, blame-shifting, disrespect).
Ask ONE question at a time. Keep questions short and natural, like a real interviewer.
Do not repeat a question that has already been asked in this session's history.""",
    "tech": """You are the Tech Lead evaluator on a hiring panel. You weigh: technical depth,
correctness of reasoning, and how the candidate handles tradeoffs and failure modes.
At least some of your questions should be genuinely technical for the role_title given —
a real problem-solving, system-design, or debugging scenario (not just "tell me about a
time..."). Scale difficulty to the role (e.g. a data-structure/algorithm question for a
Backend Engineer role, an architecture tradeoff question for a senior role). You may also
build on something specific the candidate already said (interviewer memory) rather than
asking generically. Ask ONE question at a time.""",
    "hm": """You are the Hiring Manager evaluator on a hiring panel. You weigh: ownership,
impact, and team fit. Ask ONE question at a time about a time they drove something
or disagreed with a decision.""",
}

# A curated bank of real, commonly-asked interview questions, used as few-shot
# reference material only — grounds question *quality, tone, and role-relevance*
# in real examples without training anything or repeating a fixed script.
# HR and Hiring Manager questions are largely role-agnostic (they're about
# communication, ownership, conflict — same regardless of what you build), so
# those stay as flat lists. Tech Lead questions genuinely differ by role, so
# that one is split into categories and matched against the role title.
QUESTION_BANK = {
    "hr": [
        "Tell me about a time you disagreed with a teammate. How did you handle it?",
        "What part of your last project or internship are you proudest of, and why?",
        "Describe a situation where you had to adapt to a sudden change in plans.",
        "How do you prioritize when you have multiple deadlines at once?",
        "Tell me about a time you received tough feedback. What did you do with it?",
        "What kind of team environment brings out your best work?",
        "Tell me about a time you had to learn something completely new, fast.",
        "How do you handle it when you realize you've made a mistake at work?",
        "Tell me about yourself and why you're interested in this role.",
        "Describe a time you had to work with someone whose working style was very different from yours.",
        "How do you handle stress or pressure when things aren't going to plan?",
        "Tell me about a time you had to say no to a request. How did you frame it?",
        "What motivates you day-to-day, outside of deadlines?",
        "Describe a time you went out of your way to help a teammate.",
        "How do you handle ambiguity when you're not given clear instructions?",
        "Where do you see yourself growing in the next couple of years?",
    ],
    "hm": [
        "Tell me about a project you owned end-to-end. What was the outcome?",
        "Describe a time you had to push back on a decision from someone more senior.",
        "How do you handle a teammate who isn't pulling their weight?",
        "What's a mistake you made that changed how you work now?",
        "Tell me about a time you had to influence people without formal authority.",
        "Describe a time your priorities changed midway through a project.",
        "What does 'ownership' mean to you in practice, with a real example?",
        "Tell me about a decision you made that you'd make differently today.",
        "Tell me about a time you had to deliver bad news to a team or stakeholder.",
        "Describe a time you had limited information but still had to make a call.",
        "How do you decide what to cut when a deadline is at risk?",
        "Tell me about a time you mentored or unblocked someone less experienced than you.",
        "What's an example of you setting a goal for yourself beyond what was asked?",
        "Describe a conflict between two priorities you had to resolve.",
    ],
    "tech": {
        "frontend": [
            "How would you optimize a React app that's re-rendering too often?",
            "Walk me through how you'd make a page accessible to a screen reader.",
            "What's the tradeoff between client-side and server-side rendering?",
            "How would you debug a CSS layout that breaks only on mobile?",
            "Explain how you'd structure state in a form with many interdependent fields.",
            "How do you handle a slow-loading image-heavy page?",
            "What's your approach to keeping a component library consistent across a team?",
            "How would you prevent a memory leak in a single-page app that stays open for hours?",
            "Walk me through how you'd design a debounced search input from scratch.",
            "What's the difference between controlled and uncontrolled components, and when would you use each?",
            "How would you handle state that needs to be shared across many unrelated components?",
            "Explain how you'd test a component that depends on an external API call.",
        ],
        "backend": [
            "Given a list of transactions, how would you detect duplicate entries efficiently?",
            "Walk me through how you'd design a rate limiter for an API.",
            "You have a function that's suddenly running slow in production — how do you debug it?",
            "How would you decide between a SQL and NoSQL database for a new feature?",
            "Explain a tradeoff between consistency and availability you've encountered or would expect.",
            "How would you design a system to handle retrying failed background jobs?",
            "Walk me through how you'd add caching to a slow endpoint, and what could go wrong.",
            "How would you design a URL shortener, at a high level?",
            "Walk me through how you'd paginate an API that returns millions of rows.",
            "How would you prevent two requests from double-processing the same payment?",
            "Explain how you'd version an API without breaking existing clients.",
            "How would you design a notification system that has to scale to millions of users?",
        ],
        "data": [
            "How would you detect if a dataset has a class imbalance problem, and what would you do about it?",
            "Walk me through how you'd validate that a model isn't overfitting.",
            "How would you design an A/B test for a new feature, and what could bias the results?",
            "Explain a tradeoff between model accuracy and inference latency.",
            "How would you handle missing data in a pipeline feeding a production model?",
            "Walk me through how you'd debug a sudden drop in a model's live performance.",
            "How would you explain precision vs. recall to a non-technical stakeholder?",
            "Walk me through how you'd choose an evaluation metric for an imbalanced classification problem.",
            "How would you detect data drift in a model that's been running for months?",
            "Explain how you'd decide between a simple model and a more complex one for a given problem.",
        ],
        "mobile": [
            "How would you reduce a mobile app's cold-start time?",
            "Walk me through handling a network request that might fail on a poor connection.",
            "What's your approach to keeping UI smooth while doing heavy background work?",
            "How would you structure local storage for data that needs to sync later?",
            "Explain a tradeoff between native and cross-platform development.",
            "How would you handle a feature that needs to work fully offline?",
            "Walk me through how you'd debug battery drain reported by users.",
        ],
        "generic": [
            "Walk me through how you'd approach debugging a bug you can't reproduce locally.",
            "How do you decide when a piece of code needs a test versus when it doesn't?",
            "Explain a technical decision you made and a tradeoff it involved.",
            "How would you explain a technical concept from your work to a non-technical teammate?",
            "Tell me about a time your first solution to a problem turned out to be wrong.",
            "How do you approach reviewing someone else's code?",
            "Walk me through how you'd estimate how long a feature will take to build.",
            "Tell me about a technical constraint you had to design around.",
        ],
        "dsa": [
            "Given an array, find two numbers that add up to a target value. Walk me through your approach and its time complexity.",
            "How would you reverse a linked list, and what's the space complexity of your approach?",
            "Given a string, how would you check if it's a valid palindrome, ignoring case and punctuation?",
            "How would you find the first non-repeating character in a string efficiently?",
            "Walk me through how you'd detect a cycle in a linked list.",
            "Given a binary tree, how would you find its maximum depth?",
            "How would you merge two sorted arrays into one sorted array in place?",
            "Explain how you'd find the longest substring without repeating characters.",
            "Given a matrix, how would you rotate it 90 degrees in place?",
            "How would you implement a LRU cache, and what data structures would you use?",
            "Walk me through how you'd check if two strings are anagrams of each other.",
            "Given a sorted rotated array, how would you search for a target value in O(log n)?",
        ],
        "devops": [
            "How would you design a zero-downtime deployment for a service with a live database migration?",
            "Walk me through how you'd set up alerting so you find out about an outage before your users do.",
            "How would you decide what to put in a CI pipeline versus a CD pipeline?",
            "Explain how you'd roll back a bad deployment quickly and safely.",
            "How would you approach reducing a Docker image's build time and size?",
            "Walk me through your approach to managing secrets across multiple environments.",
        ],
        "qa": [
            "How would you decide what to automate versus what to keep as manual testing?",
            "Walk me through how you'd design a test plan for a feature with no existing tests.",
            "How would you approach testing a flaky feature that only fails intermittently?",
            "Explain how you'd prioritize which bugs to fix first when there are more than you can handle.",
            "How would you write a regression suite for a checkout flow?",
        ],
    },
}

_TECH_CATEGORY_KEYWORDS = {
    "frontend": ["frontend", "front-end", "front end", "react", "ui developer", "web developer"],
    "backend": ["backend", "back-end", "back end", "api", "server", "database"],
    "data": ["data scientist", "data science", "machine learning", "ml engineer", "ai engineer", "data analyst"],
    "mobile": ["mobile", "android", "ios", "flutter", "react native"],
    "dsa": ["sde", "swe", "software engineer", "software developer", "coding", "algorithm"],
    "devops": ["devops", "sre", "site reliability", "infrastructure", "platform engineer"],
    "qa": ["qa", "quality assurance", "test engineer", "sdet"],
}


def _tech_question_pool(role_title: str) -> list:
    role_lower = (role_title or "").lower()
    for category, keywords in _TECH_CATEGORY_KEYWORDS.items():
        if any(kw in role_lower for kw in keywords):
            return QUESTION_BANK["tech"][category]
    # Fall back to a full-stack blend if we can't confidently categorize the role.
    return (
        QUESTION_BANK["tech"]["generic"]
        + QUESTION_BANK["tech"]["backend"][:2]
        + QUESTION_BANK["tech"]["frontend"][:2]
        + QUESTION_BANK["tech"]["dsa"][:2]
    )


DELIBERATION_INSTRUCTIONS = """You are simulating all three evaluators on a hiring panel — HR,
Tech Lead, and Hiring Manager — reviewing one finished interview transcript together. Give each
evaluator's independent verdict on the candidate, in their own voice and rubric.

Rules:
1. Each verdict must be one of exactly: "strong", "lean", or "discuss".
2. Each evaluator MUST quote a short exact phrase (under 20 words) copied verbatim from the
   transcript below that supports their verdict. If nothing in the transcript supports a
   confident verdict, say so honestly and use "discuss".
3. Do not invent anything the candidate did not say.
4. Evaluators may disagree with each other — that's expected and fine.
5. Respond ONLY as a JSON array of exactly three objects, one per evaluator, in this order:
   hr, tech, hm. Each object: {"persona": "hr"|"tech"|"hm", "verdict": "...", "reasoning": "...", "quote": "..."}
"""


def _get_client():
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server. Add it to your .env file.",
        )
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def _generate(prompt: str, max_tokens: int = 400) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


@router.get("/active", response_model=Optional[ActiveInterviewResponse])
def get_active_interview(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    # Lets the frontend resume an in-progress interview after a page reload,
    # tab close, or a fresh login — instead of forcing the candidate back
    # through Upload every time browser state is lost. The DB is the source
    # of truth here, not the browser's in-memory JS state.
    interview = session.exec(
        select(InterviewSession)
        .where(InterviewSession.user_id == user.id)
        .where(InterviewSession.status == "in_progress")
        .order_by(InterviewSession.created_at.desc())
    ).first()
    if not interview:
        return None

    turn = session.exec(
        select(InterviewTurn)
        .where(InterviewTurn.session_id == interview.id)
        .where(InterviewTurn.answer.is_(None))
        .order_by(InterviewTurn.created_at.desc())
    ).first()
    if not turn:
        # Session exists but has no pending question (shouldn't normally
        # happen) — treat as nothing to resume rather than erroring.
        return None

    return ActiveInterviewResponse(
        session_id=interview.id,
        turn_id=turn.id,
        persona=turn.persona,
        question=turn.question,
        role_title=interview.role_title,
    )


@router.post("/start", response_model=NextQuestionResponse)
def start_interview(
    body: StartInterviewRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    # Any earlier interview(s) for this user that never got a verdict (crashed
    # tab, abandoned mid-way during testing, etc.) stay stuck as "in_progress"
    # forever otherwise — and /interview/active would resurrect the wrong one
    # after you'd already finished a newer session. Close them out first so
    # there's only ever one resumable session per user.
    stale = session.exec(
        select(InterviewSession)
        .where(InterviewSession.user_id == user.id)
        .where(InterviewSession.status == "in_progress")
    ).all()
    for s in stale:
        s.status = "abandoned"
        session.add(s)

    interview = InterviewSession(
        user_id=user.id,
        role_title=body.role_title,
        resume_analysis_id=body.resume_analysis_id,
    )
    session.add(interview)
    session.commit()
    session.refresh(interview)
    return _ask_next(session, interview, persona="hr")


def _ask_next(session: Session, interview: InterviewSession, persona: str) -> NextQuestionResponse:
    history_turns = session.exec(
        select(InterviewTurn).where(InterviewTurn.session_id == interview.id)
    ).all()
    def _turn_block(t: InterviewTurn) -> str:
        block = f"[{PERSONA_LABEL[t.persona]}] Q: {t.question}\nA: {t.answer or '(not yet answered)'}"
        signal = summarize_signal(t.voice_wpm, t.voice_filler_count, t.voice_pause_count)
        if signal:
            block += f"\n(delivery: {signal})"
        if t.eye_contact_pct is not None:
            block += f"\n(candidate's own device reported {t.eye_contact_pct:.0f}% eye contact with camera during this answer"
            if t.posture_note:
                block += f", {t.posture_note}"
            block += "; self-reported, unverified — weigh accordingly)"
        return block

    history_text = "\n".join(_turn_block(t) for t in history_turns)
    question_pool = _tech_question_pool(interview.role_title) if persona == "tech" else QUESTION_BANK[persona]
    examples = "\n".join(f"- {q}" for q in random.sample(question_pool, k=min(3, len(question_pool))))
    prompt = f"""{PERSONA_RUBRIC[persona]}

For reference, here are examples of the style/quality of question a real interviewer
in this role would ask (don't copy them verbatim — use them as a calibration for tone
and specificity):
{examples}

Candidate is interviewing for: {interview.role_title}

Interview so far:
{history_text or '(this is the first question)'}

Ask your next question now. Reply with ONLY the question text, nothing else."""

    question_text = _generate(prompt)
    if not question_text or not question_text.strip():
        # Groq occasionally returns an empty completion. Retry once before
        # falling back, so a real interview question always reaches the user.
        question_text = _generate(prompt)
    if not question_text or not question_text.strip():
        fallback_pool = [q for q in question_pool if q not in [t.question for t in history_turns]] or question_pool
        question_text = random.choice(fallback_pool)
    turn = InterviewTurn(session_id=interview.id, persona=persona, question=question_text)
    session.add(turn)
    session.commit()
    session.refresh(turn)

    return NextQuestionResponse(session_id=interview.id, turn_id=turn.id, persona=persona, question=question_text)


@router.post("/answer", response_model=NextQuestionResponse)
def submit_answer(
    body: SubmitAnswerRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    interview = session.get(InterviewSession, body.session_id)
    if not interview or interview.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    if interview.status != "in_progress":
        # This session already got a verdict (or was superseded by a newer
        # one) — refuse to silently bolt more answers onto it. This is the
        # server-side backstop for "the old interview was still answerable";
        # the /start cleanup above stops it from happening in the first
        # place, this stops it from ever mattering even if a stale session
        # reference slips through on the frontend somehow.
        raise HTTPException(status_code=409, detail="This interview has already ended. Start a new one from Upload.")

    turn = session.get(InterviewTurn, body.turn_id)
    if not turn or turn.session_id != interview.id:
        raise HTTPException(status_code=404, detail="Question not found")

    turn.answer = body.answer
    turn.voice_wpm = body.voice_wpm
    turn.voice_filler_count = body.voice_filler_count
    turn.voice_pause_count = body.voice_pause_count
    turn.eye_contact_pct = body.eye_contact_pct
    turn.posture_note = body.posture_note
    session.add(turn)

    interview.transcript += f"\n[{PERSONA_LABEL[turn.persona]}] Q: {turn.question}\nA: {body.answer}"
    signal_line = summarize_signal(body.voice_wpm, body.voice_filler_count, body.voice_pause_count)
    if signal_line:
        interview.transcript += f"\n(delivery: {signal_line})"
    if body.eye_contact_pct is not None:
        interview.transcript += (
            f"\n(self-reported eye contact: {body.eye_contact_pct:.0f}%"
            + (f", {body.posture_note}" if body.posture_note else "")
            + ")"
        )
    session.add(interview)
    session.commit()

    all_turns = session.exec(
        select(InterviewTurn).where(InterviewTurn.session_id == interview.id)
    ).all()
    asked_count = len(all_turns)
    next_persona = PERSONAS[asked_count % len(PERSONAS)]
    return _ask_next(session, interview, persona=next_persona)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _build_coaching_prompt(turns: List[InterviewTurn]) -> Optional[str]:
    answered = [t for t in turns if t.answer is not None]
    if not answered:
        return None
    qa_block = "\n\n".join(
        f"Q{i+1} ({PERSONA_LABEL[t.persona]}): {t.question}\nCandidate's answer: {t.answer}"
        for i, t in enumerate(answered)
    )
    return f"""For each Q&A below, show how the candidate could strengthen their own answer.
Start with what they said (acknowledge it), then show the improved version building on that core idea.
Use STAR structure (Situation/Task/Action/Result) for behavioral questions. For technical questions,
add more depth/specificity. Keep the improved answer under 120 words. Be coaching, not harsh.

Reply ONLY as a JSON array in this exact format, same order as questions:
{{"question": "...", "your_answer": "...", "stronger_answer": "...", "why_its_stronger": "..."}}

{qa_block}"""


def _parse_coaching(raw: str) -> List["CoachingItem"]:
    from app.models import CoachingItem
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON array")
        return [CoachingItem(**item) for item in parsed]
    except Exception:
        # Coaching is a bonus, not core to the verdict — fail quietly rather
        # than break the whole deliberation response.
        return []


@router.post("/{session_id}/flag")
def flag_interview(
    session_id: int,
    body: FlagRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    # This route didn't exist before — the frontend was already calling it
    # (fire-and-forget) on tab-switches and paste attempts, so every one of
    # those signals was silently lost. Now it's actually saved.
    interview = session.get(InterviewSession, session_id)
    if not interview or interview.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    session.add(InterviewFlag(session_id=session_id, kind=body.kind))
    session.commit()
    return {"ok": True}


def _flag_counts(session: Session, session_id: int) -> Dict[str, int]:
    flags = session.exec(select(InterviewFlag).where(InterviewFlag.session_id == session_id)).all()
    return {
        "context_switches": sum(1 for f in flags if f.kind == "context_switch"),
        "paste_attempts": sum(1 for f in flags if f.kind == "paste_attempt"),
    }


@router.post("/{session_id}/deliberate", response_model=DeliberationResponse)
def deliberate(
    session_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    interview = session.get(InterviewSession, session_id)
    if not interview or interview.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    interview.status = "completed"
    session.add(interview)

    normalized_transcript = _normalize(interview.transcript)

    all_turns = session.exec(
        select(InterviewTurn).where(InterviewTurn.session_id == interview.id)
    ).all()
    # Only personas with at least one ANSWERED turn get a real verdict.
    # Without this, ending the interview after only the HR question would
    # still have the model invent Tech/HM verdicts from an empty transcript.
    evaluated_personas = [p for p in PERSONAS if any(t.persona == p and t.answer for t in all_turns)]
    skipped_personas = [p for p in PERSONAS if p not in evaluated_personas]

    rubric_block = "\n\n".join(
        f"{PERSONA_LABEL[p]} rubric:\n{PERSONA_RUBRIC[p]}" for p in evaluated_personas
    )
    verdict_prompt = f"""{DELIBERATION_INSTRUCTIONS}

{rubric_block}

Transcript:
{interview.transcript}

Only these evaluators actually got to ask/hear an answer this session: {', '.join(evaluated_personas) or 'none'}.
Give a verdict ONLY for those evaluators, as a JSON array with exactly that many objects."""

    coaching_prompt = _build_coaching_prompt(all_turns) if evaluated_personas else None

    # These two Groq calls are fully independent (coaching only needs the
    # raw Q&A, not the verdicts) — running them one after another used to
    # roughly double how long "the panel is deliberating…" took. Firing
    # them off in parallel threads cuts that wait to whichever call is
    # slower, not the sum of both.
    from concurrent.futures import ThreadPoolExecutor

    by_persona: Dict[str, dict] = {}
    coaching_raw = None
    if evaluated_personas:
        with ThreadPoolExecutor(max_workers=2) as pool:
            verdict_future = pool.submit(_generate, verdict_prompt, 900)
            coaching_future = pool.submit(_generate, coaching_prompt, 1600) if coaching_prompt else None

            raw = verdict_future.result()
            coaching_raw = coaching_future.result() if coaching_future else None

        try:
            cleaned = raw.strip().strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array")
            by_persona = {item.get("persona"): item for item in parsed}
        except Exception:
            # Fall back to "discuss" for the evaluated personas rather than fail
            # the whole request — a malformed model response shouldn't crash a
            # completed interview.
            by_persona = {}

    results: List[VerdictItem] = []
    for persona in PERSONAS:
        if persona in skipped_personas:
            item = Verdict(
                session_id=interview.id,
                persona=persona,
                verdict="discuss",
                reasoning="Not evaluated — you didn't answer any question from this evaluator this session.",
                quote="",
                quote_verified=False,
            )
        else:
            data = by_persona.get(persona, {})
            quote = data.get("quote", "")
            verified = bool(quote) and _normalize(quote) in normalized_transcript
            item = Verdict(
                session_id=interview.id,
                persona=persona,
                verdict=data.get("verdict", "discuss"),
                reasoning=data.get("reasoning", "Could not parse a verdict from the panel this time."),
                quote=quote,
                quote_verified=verified,
            )
        session.add(item)
        results.append(VerdictItem(
            persona=persona,
            verdict=item.verdict,
            reasoning=item.reasoning,
            quote=item.quote,
            quote_verified=item.quote_verified,
        ))

    session.commit()

    coaching = _parse_coaching(coaching_raw) if coaching_raw else []

    flag_counts = _flag_counts(session, interview.id)
    return DeliberationResponse(
        session_id=interview.id,
        verdicts=results,
        transcript=interview.transcript,
        coaching=coaching,
        context_switches=flag_counts["context_switches"],
        paste_attempts=flag_counts["paste_attempts"],
    )
