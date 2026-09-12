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

# Shared by panel.py and dsa.py so every Groq-generated response (questions,
# verdicts, coaching, DSA feedback) can be asked for in the candidate's
# chosen UI language, instead of only ever coming back in English.
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}


def language_instruction(language: str) -> str:
    name = LANGUAGE_NAMES.get((language or "en").lower(), "English")
    if name == "English":
        return ""
    return f"\n\nIMPORTANT: Respond ONLY in {name}. Every word of your reply must be in {name}, not English."


DIFFICULTY_INSTRUCTION = {
    "easy": "Keep questions approachable — fundamentals, no multi-step tradeoffs. Ease the candidate in.",
    "moderate": "Keep questions at a normal, realistic interview difficulty — some depth expected, not entry-level.",
    "difficult": "Push harder than a normal interview — expect the candidate to reason through edge cases, "
                 "tradeoffs, and follow-up pressure. Don't go easy.",
}


def _extract_json(raw: str):
    """Best-effort JSON extraction from a Groq completion.

    Small models (this app uses gpt-oss-20b for speed) sometimes wrap JSON in
    prose, markdown fences, or partial commentary even when told not to —
    especially once a prompt asks for several things at once (language,
    STAR flags, score, verbatim quotes). A plain json.loads() on the raw
    string was failing outright in exactly that situation, which showed up
    as "Could not parse a verdict" for every persona at once. This tries,
    in order: the raw string as-is, the string with code fences stripped,
    then the first {...}/[...] block found anywhere in the text.
    """
    if not raw:
        return None
    candidates = [raw.strip()]
    fenced = raw.strip().strip("`")
    if fenced.lower().startswith("json"):
        fenced = fenced[4:]
    candidates.append(fenced.strip())
    # Grab the first balanced-looking [...] or {...} block as a last resort.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def loose_extract_field(chunk: str, key: str, next_keys=()) -> str:
    """Pull a string field's value out of JSON-ish text WITHOUT requiring the
    surrounding text to be valid JSON.

    This is the fallback of last resort for when a model asked to copy
    something "verbatim" embeds a raw quote/newline/backslash that breaks
    strict parsing, or the response got truncated mid-string by a token
    limit. Rather than needing the whole blob to parse, this just finds
    `"key": "` and reads forward to wherever the next expected key starts
    (or the end of the chunk) — good enough to recover a field even out of
    genuinely malformed JSON.
    """
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', chunk)
    if not m:
        return ""
    start = m.end()
    end = len(chunk)
    for nk in next_keys:
        m2 = re.search(r'"' + re.escape(nk) + r'"\s*:', chunk[start:])
        if m2:
            end = min(end, start + m2.start())
    value = chunk[start:end]
    value = re.sub(r'["\',}\s]+$', '', value)
    # Unescape the common JSON escapes a truncated/malformed blob still uses.
    value = value.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t')
    return value.strip()


def _loose_extract_verdicts(raw: str, personas: list) -> Dict[str, dict]:
    """Tertiary fallback for the verdict array when even _extract_json()
    fails on both the first try and the retry. Anchors on each "persona"
    occurrence and reads the fields around it with loose_extract_field()."""
    result: Dict[str, dict] = {}
    if not raw:
        return result
    matches = list(re.finditer(r'"persona"\s*:\s*"(\w+)"', raw))
    for i, m in enumerate(matches):
        persona = m.group(1)
        if persona not in personas:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[start:end]
        verdict = loose_extract_field(chunk, "verdict", ["reasoning", "quote"]) or "discuss"
        reasoning = loose_extract_field(chunk, "reasoning", ["quote"])
        quote = loose_extract_field(chunk, "quote", ["persona"])
        result[persona] = {"persona": persona, "verdict": verdict or "discuss", "reasoning": reasoning, "quote": quote}
    return result

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

# Structured domain -> question-bank category. This is the primary signal now
# that the frontend always exposes a Domain picker (previously it was hidden
# unless a company was chosen, so this path only mattered in the company
# branch; the free-text role_title keyword match below was the sole signal
# for the no-company case, which meant picking a domain there did nothing).
_DOMAIN_CATEGORY_MAP = {
    "backend": "backend",
    "frontend": "frontend",
    "data_science": "data",
    "data_analyst": "data",
    "ml_engineer": "data",
}


def _tech_question_pool(role_title: str, domain: str = None) -> list:
    # 1. Explicit domain selection wins — it's a deliberate structured choice,
    #    more reliable than guessing from freeform role-title text.
    if domain and domain in _DOMAIN_CATEGORY_MAP:
        return QUESTION_BANK["tech"][_DOMAIN_CATEGORY_MAP[domain]]
    # 2. Fall back to keyword matching against role_title (covers categories
    #    with no dedicated domain option yet, e.g. mobile/devops/qa, and any
    #    caller that still only supplies role_title).
    role_lower = (role_title or "").lower()
    for category, keywords in _TECH_CATEGORY_KEYWORDS.items():
        if any(kw in role_lower for kw in keywords):
            return QUESTION_BANK["tech"][category]
    # 3. Fall back to a full-stack blend if we can't confidently categorize the role.
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
        difficulty=body.difficulty or "moderate",
        language=body.language or "en",
        company=body.company or "general",
        domain=body.domain or "general",
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
    question_pool = _tech_question_pool(interview.role_title, interview.domain) if persona == "tech" else QUESTION_BANK[persona]

    company_note = ""
    if interview.company and interview.company != "general":
        from app.company_bank import get_behavioral_pool, get_technical_pool, COMPANIES, DOMAINS
        company_label = COMPANIES.get(interview.company, interview.company)
        if persona == "tech":
            bank_pool = get_technical_pool(interview.company, interview.domain)
        else:
            bank_pool = get_behavioral_pool(interview.company)
        if bank_pool:
            # Real, company-flavored examples used as calibration for EVERY
            # turn (not just the first question) — a large pool so the same
            # handful don't repeat across turns or across sessions. The
            # candidate's actual answers still drive adaptation; this only
            # steers style/topic to stay in the chosen company+domain lane.
            question_pool = bank_pool
            domain_label = DOMAINS.get(interview.domain, "the candidate's field") if persona == "tech" else None
            company_note = (
                f"\n\nThis interview is styled after {company_label}'s real, publicly documented interview "
                f"approach{f' for {domain_label} roles' if domain_label else ''}. Every question you ask — "
                f"not just this one — should stay grounded in that company's genuine interview style and "
                f"topic focus. Do not claim a question is a verified real transcript; it should simply be "
                f"authentically in that style."
            )

    examples = "\n".join(f"- {q}" for q in random.sample(question_pool, k=min(5, len(question_pool))))
    prompt = f"""{PERSONA_RUBRIC[persona]}

For reference, here are examples of the style/quality of question a real interviewer
in this role would ask (don't copy them verbatim — use them as a calibration for tone
and specificity):
{examples}{company_note}

Difficulty for this session: {interview.difficulty}. {DIFFICULTY_INSTRUCTION.get(interview.difficulty, DIFFICULTY_INSTRUCTION['moderate'])}

Candidate is interviewing for: {interview.role_title}

Interview so far:
{history_text or '(this is the first question)'}

Ask your next question now. Reply with ONLY the question text, nothing else.{language_instruction(interview.language)}"""

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


def _compute_communication_score(answer: str, wpm: Optional[float], filler_count: Optional[int]) -> Optional[int]:
    """Communication score computed from REAL captured voice signal (pace,
    filler-word count) — deliberately NOT an LLM guess from the text alone,
    since that would just be duplicating "score" under a different name.
    Returns None when no voice signal was captured for this turn (e.g. the
    candidate typed the answer instead of speaking it) rather than faking a
    number with nothing real behind it.
    """
    if wpm is None and filler_count is None:
        return None
    word_count = max(len((answer or "").split()), 1)
    score = 100.0
    if filler_count is not None and word_count > 0:
        filler_ratio = filler_count / word_count
        score -= min(50.0, filler_ratio * 300.0)
    if wpm is not None:
        if wpm < 100:
            score -= min(25.0, (100 - wpm) * 0.5)
        elif wpm > 180:
            score -= min(25.0, (wpm - 180) * 0.5)
    return max(0, round(score))


def _build_coaching_prompt(turns: List[InterviewTurn], language: str = "en") -> Optional[str]:
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
add more depth/specificity. Aim for roughly 100-160 words — long enough to actually demonstrate a
full Situation/Task/Action/Result arc (or real technical depth) instead of a thin one-liner, but stop
once the point is made rather than padding. Be coaching, not harsh.

Also evaluate the CANDIDATE'S OWN answer as literally written above (NOT your rewrite in
"stronger_answer" — read only the "Candidate's answer:" text for this part). Be an honest, calibrated
grader — this cuts BOTH ways, and getting either direction wrong is equally a failure:
- "score": an integer 0-100 for how strong/complete their own answer already was, compared to the
  ideal ("star") answer you wrote as stronger_answer.
  - An answer that is incoherent, trails off, asks for more time, or doesn't actually address the
    question should score under 20 — do not be generous just because the topic is relevant.
  - An answer that is genuinely clear, specific, well-structured, and actually answers what was asked
    should score high (80-100) — do not manufacture nitpicks or dock points just to seem rigorous.
    If it already covers Situation/Task/Action/Result concretely (or, for a technical question, is
    correct and reasonably complete), that IS a strong answer and the score must reflect that.
  - Judge the answer that's actually there, not how it compares to a hypothetical perfect candidate —
    "good but could add one more metric" is still a high score, not a middling one.
- "domain_knowledge_score": an integer 0-100 judging specifically the technical/domain accuracy and
  depth shown — separate from how well-structured or articulate the answer was. A technically correct
  but plainly-worded answer should still score high here; a fluent answer with wrong or vague technical
  content should score low here even if "score" above is more forgiving of its structure.
- "star_detected": for behavioral questions, re-read the candidate's own answer sentence by sentence
  and individually flag whether IT (not your rewrite) already contained a Situation, a Task, an
  Action, and a Result, as booleans {{"s":bool,"t":bool,"a":bool,"r":bool}}. Be strict and literal:
  a flag is true only if that specific element is actually present in their words — a vague mention
  of a topic is not a Situation, wanting to explain something is not a Task, and "I'll tell you in a
  second" is not any of the four. If the candidate's answer never mentions a piece (e.g. never states
  a measurable Result), that flag must be false — do not mark something true just because your
  improved version added it, and never default all four to true. Equally, if the candidate's answer
  DOES clearly state all four, all four must be true — don't withhold a flag that's genuinely earned.
  An answer that is rambling, unfinished, or asks to come back to the question later should have MOST
  or ALL of these false. For a purely technical question, set all four to false and base "score" on
  correctness/completeness instead.
- "what_went_well": 1-3 short bullet strings (under 15 words each) on specifically what was good about
  THIS answer. If the answer was genuinely weak with nothing to credit, this can be a single honest
  bullet like "Attempted to engage with the question" rather than an invented compliment — but don't
  leave it empty just because the answer was weak; find the one true thing if there is one.
- "what_to_improve": 1-3 short bullet strings (under 15 words each) on specifically what to fix, tied
  to what's actually missing or wrong in THIS answer — not generic advice that could apply to any answer.
- "missing_terminologies": 0-4 short key terms/concepts (a few words each, e.g. "Time complexity",
  "A/B testing", "Load balancing") that a strong answer to this specific question would have used but
  this candidate's answer did not mention. Leave this empty if the answer already used the relevant
  terms, or if the question doesn't really have expected terminology (e.g. a pure opinion question).

Reply ONLY as a JSON array in this exact format, same order as questions:
{{"question": "...", "your_answer": "...", "stronger_answer": "...", "why_its_stronger": "...",
"score": 0, "domain_knowledge_score": 0, "star_detected": {{"s": false, "t": false, "a": false, "r": false}},
"what_went_well": ["..."], "what_to_improve": ["..."], "missing_terminologies": ["..."]}}

{qa_block}{language_instruction(language)}
Exception: "your_answer" must stay copied verbatim from what the candidate actually said, not translated."""


def _loose_extract_string_array(chunk: str, key: str) -> list:
    """Pull a JSON array of short strings out of possibly-malformed text —
    used for what_went_well/what_to_improve/missing_terminologies when the
    surrounding JSON doesn't parse cleanly."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[(.*?)\]', chunk, re.S)
    if not m:
        return []
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return [i.replace('\\"', '"').strip() for i in items if i.strip()]


def _loose_extract_coaching(raw: str) -> list:
    """Same idea as _loose_extract_verdicts, for coaching items. Anchors on
    each "question" occurrence since that's the first key in every object."""
    items = []
    if not raw:
        return items
    matches = list(re.finditer(r'"question"\s*:\s*"', raw))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[start:end]
        question = loose_extract_field(chunk, "question", ["your_answer"])
        your_answer = loose_extract_field(chunk, "your_answer", ["stronger_answer"])
        stronger_answer = loose_extract_field(chunk, "stronger_answer", ["why_its_stronger"])
        why = loose_extract_field(chunk, "why_its_stronger", ["score", "domain_knowledge_score", "star_detected"])
        score_m = re.search(r'"score"\s*:\s*(\d+)', chunk)
        score = int(score_m.group(1)) if score_m else 0
        dk_m = re.search(r'"domain_knowledge_score"\s*:\s*(\d+)', chunk)
        domain_knowledge_score = int(dk_m.group(1)) if dk_m else 0
        star_chunk_m = re.search(r'"star_detected"\s*:\s*\{([^}]*)\}', chunk)
        star_chunk = star_chunk_m.group(1) if star_chunk_m else ""
        def flag(key):
            fm = re.search(r'"' + key + r'"\s*:\s*(true|false)', star_chunk)
            return fm.group(1) == "true" if fm else False
        items.append({
            "question": question, "your_answer": your_answer,
            "stronger_answer": stronger_answer, "why_its_stronger": why,
            "score": score, "domain_knowledge_score": domain_knowledge_score,
            "star_detected": {"s": flag("s"), "t": flag("t"), "a": flag("a"), "r": flag("r")},
            "what_went_well": _loose_extract_string_array(chunk, "what_went_well"),
            "what_to_improve": _loose_extract_string_array(chunk, "what_to_improve"),
            "missing_terminologies": _loose_extract_string_array(chunk, "missing_terminologies"),
        })
    return items


def _parse_coaching(raw: str) -> List["CoachingItem"]:
    from app.models import CoachingItem, StarDetected
    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        parsed = _loose_extract_coaching(raw)
    if not isinstance(parsed, list):
        return []
    items = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        star_raw = item.get("star_detected") or {}
        if not isinstance(star_raw, dict):
            star_raw = {}
        def _str_list(v):
            if not isinstance(v, list):
                return []
            return [str(x).strip() for x in v if str(x).strip()][:4]
        try:
            items.append(CoachingItem(
                question=item.get("question", ""),
                your_answer=item.get("your_answer", ""),
                stronger_answer=item.get("stronger_answer", ""),
                why_its_stronger=item.get("why_its_stronger", ""),
                score=int(item.get("score", 0) or 0),
                domain_knowledge_score=int(item.get("domain_knowledge_score", 0) or 0),
                star_detected=StarDetected(
                    s=bool(star_raw.get("s")), t=bool(star_raw.get("t")),
                    a=bool(star_raw.get("a")), r=bool(star_raw.get("r")),
                ),
                what_went_well=_str_list(item.get("what_went_well")),
                what_to_improve=_str_list(item.get("what_to_improve")),
                missing_terminologies=_str_list(item.get("missing_terminologies")),
            ))
        except Exception:
            # One malformed item shouldn't drop every other coaching item.
            continue
    # Drop items that came out with an empty stronger_answer — that's the
    # signature of the response getting truncated mid-array (usually the
    # last item). Showing a blank "Stronger version:" is more confusing than
    # just omitting that one Q&A from coaching entirely.
    items = [i for i in items if i.stronger_answer.strip()]
    return items


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
Give a verdict ONLY for those evaluators, as a JSON array with exactly that many objects.{language_instruction(interview.language)}
Exception: the "quote" field must stay copied VERBATIM from the transcript above, in whatever language
the transcript is actually written in — do not translate the quote itself, only "reasoning"."""

    coaching_prompt = _build_coaching_prompt(all_turns, interview.language) if evaluated_personas else None

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
            verdict_future = pool.submit(_generate, verdict_prompt, 1200)
            coaching_future = pool.submit(_generate, coaching_prompt, 5500) if coaching_prompt else None

            raw = verdict_future.result()
            coaching_raw = coaching_future.result() if coaching_future else None

        parsed = _extract_json(raw)
        if isinstance(parsed, list):
            by_persona = {item.get("persona"): item for item in parsed if isinstance(item, dict)}
        else:
            # Try the cheap regex-based recovery BEFORE paying for another full
            # model call — this is the common case now (truncation/unescaped
            # quotes, not the model ignoring the format), and skipping the
            # retry here is most of the latency win in this round.
            by_persona = _loose_extract_verdicts(raw, evaluated_personas)
            if not by_persona:
                # Genuinely nothing usable in the first response — now it's
                # worth paying for one retry with a blunter reminder.
                raw = _generate(verdict_prompt + "\n\nReminder: reply with ONLY the JSON array, no commentary, no markdown.", 1200)
                parsed = _extract_json(raw)
                if isinstance(parsed, list):
                    by_persona = {item.get("persona"): item for item in parsed if isinstance(item, dict)}
                else:
                    by_persona = _loose_extract_verdicts(raw, evaluated_personas)

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
    if coaching_raw and not coaching and coaching_prompt:
        # Same one-retry treatment as the verdict parse above — a blank
        # coaching list from a malformed response used to just silently
        # hide the whole "Stronger Answers" section.
        retry_raw = _generate(coaching_prompt + "\n\nReminder: reply with ONLY the JSON array, no commentary, no markdown.", 5500)
        coaching = _parse_coaching(retry_raw)

    # Attach a REAL, server-computed communication score per item from the
    # matching turn's actual captured voice signal — same order as the
    # "answered" list _build_coaching_prompt used, so index i lines up.
    answered_turns = [t for t in all_turns if t.answer is not None]
    for i, item in enumerate(coaching):
        if i < len(answered_turns):
            turn = answered_turns[i]
            item.communication_score = _compute_communication_score(
                item.your_answer or turn.answer, turn.voice_wpm, turn.voice_filler_count
            )

    flag_counts = _flag_counts(session, interview.id)
    return DeliberationResponse(
        session_id=interview.id,
        verdicts=results,
        transcript=interview.transcript,
        coaching=coaching,
        context_switches=flag_counts["context_switches"],
        paste_attempts=flag_counts["paste_attempts"],
    )
