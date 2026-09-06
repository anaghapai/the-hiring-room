import json
import random
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import get_current_user
from app.models import User, DsaQuestionOut, DsaSubmitRequest, DsaSubmitResponse
from app.panel import _generate, LANGUAGE_NAMES, language_instruction, _extract_json, loose_extract_field
from app.dsa_bank import by_difficulty, by_id

router = APIRouter(prefix="/dsa", tags=["dsa"])

# In-memory cache: (question_id, language) -> translated {"prompt":..., "examples":[...]}.
# Resets on server restart, which is fine — it just means the first request
# for a given question+language pays for one extra Groq call.
_TRANSLATION_CACHE: dict = {}


def _translated_question(q: dict, language: str) -> dict:
    language = (language or "en").lower()
    if language == "en" or language not in LANGUAGE_NAMES:
        return {"prompt": q["prompt"], "examples": q["examples"]}

    key = (q["id"], language)
    if key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[key]

    prompt_text = (
        f"Translate the following DSA problem statement and its example lines into "
        f"{LANGUAGE_NAMES[language]}. Keep code-like tokens, numbers, and array/bracket "
        f"notation exactly as-is (only translate the surrounding natural language). "
        f"Reply ONLY as JSON: {{\"prompt\": \"...\", \"examples\": [\"...\"]}}\n\n"
        f"Title: {q['title']}\nProblem: {q['prompt']}\nExamples: {json.dumps(q['examples'])}"
    )
    try:
        raw = _generate(prompt_text, 500)
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        parsed = json.loads(cleaned)
        result = {
            "prompt": parsed.get("prompt") or q["prompt"],
            "examples": parsed.get("examples") or q["examples"],
        }
    except Exception:
        # Translation is a nice-to-have here — fall back to English rather
        # than failing the whole question fetch.
        result = {"prompt": q["prompt"], "examples": q["examples"]}

    _TRANSLATION_CACHE[key] = result
    return result


@router.get("/question", response_model=DsaQuestionOut)
def get_question(
    difficulty: str = Query("medium"),
    language: str = Query("en"),
    exclude: Optional[str] = Query(None, description="Comma-separated question ids already seen this session"),
    user: User = Depends(get_current_user),
):
    pool = by_difficulty(difficulty)
    seen = set((exclude or "").split(",")) if exclude else set()
    candidates = [q for q in pool if q["id"] not in seen] or pool
    q = random.choice(candidates)
    translated = _translated_question(q, language)
    return DsaQuestionOut(
        id=q["id"], title=q["title"], difficulty=q["difficulty"], topic=q["topic"],
        prompt=translated["prompt"], examples=translated["examples"],
    )


@router.post("/submit", response_model=DsaSubmitResponse)
def submit(body: DsaSubmitRequest, user: User = Depends(get_current_user)):
    q = by_id(body.question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    if not body.code or not body.code.strip():
        raise HTTPException(status_code=400, detail="No code was submitted")

    if body.debug_mode:
        prompt = f"""You are a patient debugging assistant helping a candidate practice DSA.
Problem: {q['title']} — {q['prompt']}

Candidate's {body.prog_language} code so far:
```
{body.code}
```

Do NOT give away a full working solution. Instead:
1. Walk through, in words, what the code's key variables/state would look like as it runs on a
   small example (a step-by-step trace) — enough to help them see where behavior diverges from
   what they expect. Keep this under 150 words — a short trace, not an exhaustive one.
2. Point at the specific likely bug(s) or edge case(s) being missed, as a nudge/hint — not the fix
   itself spelled out verbatim. Keep this under 40 words.

Reply ONLY as JSON, no markdown fences, no commentary before or after:
{{"feedback": "the step-through trace", "hint": "the nudge, without the full fix"}}
{language_instruction(body.language)}"""
        raw = _generate(prompt, 1400)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            feedback = parsed.get("feedback", "") or ""
            hint = parsed.get("hint")
        else:
            # Try the cheap regex recovery before paying for a second full
            # model call — usually recovers it without needing the retry.
            feedback = loose_extract_field(raw, "feedback", ["hint"])
            hint = loose_extract_field(raw, "hint", []) or None
            if not feedback:
                raw = _generate(prompt + "\n\nReminder: reply with ONLY that JSON object, nothing else.", 1400)
                parsed = _extract_json(raw)
                if isinstance(parsed, dict):
                    feedback = parsed.get("feedback", "") or ""
                    hint = parsed.get("hint")
                else:
                    feedback = loose_extract_field(raw, "feedback", ["hint"]) or raw
                    hint = loose_extract_field(raw, "hint", []) or None
        return DsaSubmitResponse(
            question_id=q["id"], debug_mode=True, score=0,
            verdict="", feedback=feedback, hint=hint, reference_approach=None,
        )

    prompt = f"""You are grading a candidate's solution to a DSA practice problem.
Problem: {q['title']} — {q['prompt']}
Examples: {json.dumps(q['examples'])}

Candidate's {body.prog_language} code:
```
{body.code}
```

Grade it 0-100 for: correctness on the given examples and obvious edge cases, time/space
complexity, and code clarity. Be an honest, calibrated grader — this cuts both ways:
- If the code is correct, handles the obvious edge cases, and uses a reasonable approach, score it
  80-100. Don't invent nitpicks or dock points just to seem rigorous — minor style preferences are
  not a reason to lower a genuinely correct solution's score.
- If the code is actually wrong, missing edge cases, or would error/infinite-loop, score it low and
  say specifically why — don't soften a real problem into a passing grade.
Then give a short verdict line (under 10 words) and a fuller
explanation (under 120 words), and a short description (2-3 sentences, not full code) of an
ideal/reference approach for comparison.

Reply ONLY as JSON, no markdown fences, no commentary before or after:
{{"score": 0, "verdict": "short line", "feedback": "fuller explanation", "reference_approach": "short description"}}
{language_instruction(body.language)}"""
    raw = _generate(prompt, 1200)
    parsed = _extract_json(raw)
    if isinstance(parsed, dict):
        return DsaSubmitResponse(
            question_id=q["id"], debug_mode=False,
            score=int(parsed.get("score", 0) or 0),
            verdict=parsed.get("verdict", ""),
            feedback=parsed.get("feedback", ""),
            reference_approach=parsed.get("reference_approach"),
        )
    # Try the cheap regex recovery first — skip the retry call if it works.
    score_m = re.search(r'"score"\s*:\s*(\d+)', raw)
    feedback = loose_extract_field(raw, "feedback", ["reference_approach"])
    if not feedback:
        raw = _generate(prompt + "\n\nReminder: reply with ONLY that JSON object, nothing else.", 1200)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return DsaSubmitResponse(
                question_id=q["id"], debug_mode=False,
                score=int(parsed.get("score", 0) or 0),
                verdict=parsed.get("verdict", ""),
                feedback=parsed.get("feedback", ""),
                reference_approach=parsed.get("reference_approach"),
            )
        score_m = re.search(r'"score"\s*:\s*(\d+)', raw)
        feedback = loose_extract_field(raw, "feedback", ["reference_approach"]) or raw
    return DsaSubmitResponse(
        question_id=q["id"], debug_mode=False,
        score=int(score_m.group(1)) if score_m else 0,
        verdict=loose_extract_field(raw, "verdict", ["feedback"]) or "Could not parse a clean verdict this time.",
        feedback=feedback,
        reference_approach=loose_extract_field(raw, "reference_approach", []) or None,
    )
