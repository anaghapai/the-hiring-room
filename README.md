# The Hiring Room — Backend

A real, runnable FastAPI backend for the app: auth, resume/JD matching,
the three-persona interview panel (Gemini), a deliberation step that
verifies each verdict quotes something that actually exists in the
transcript, and a voice/prosody analysis endpoint.

No model training required to get this running — see "About training /
datasets" at the bottom.

## 0. Prerequisites

- Python 3.10+
- A Gemini API key (console.cloud.google.com or aistudio.google.com — search
  "Gemini API key" if you don't have one yet)
- `ffmpeg` installed on your machine (needed by librosa to read audio files):
  - Mac: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: install via https://ffmpeg.org and add it to PATH

## 1. Set up the project

```bash
cd hiring-room-backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This step will take a few minutes the first time — sentence-transformers
and librosa pull in some large dependencies (torch, etc.).

## 2. Configure your environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
GEMINI_API_KEY=your_real_key_here
JWT_SECRET=any_long_random_string
```

## 3. Run it

```bash
uvicorn app.main:app --reload
```

You should see it running at http://127.0.0.1:8000

FastAPI auto-generates interactive docs — open this in your browser and
you can test every endpoint by hand, no frontend needed yet:

```
http://127.0.0.1:8000/docs
```

## 4. Try it end-to-end with curl

Register:
```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com","password":"pass1234"}'
```
Copy the `access_token` from the response, then set it as a variable:
```bash
export TOKEN="paste_the_token_here"
```

Analyze a resume against a JD:
```bash
curl -X POST http://127.0.0.1:8000/resume/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "jd_text=Looking for a backend engineer with Python and system design experience." \
  -F "file=@/path/to/your/resume.pdf"
```

Start an interview:
```bash
curl -X POST http://127.0.0.1:8000/interview/start \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"role_title":"Backend Engineer"}'
```
This returns a `session_id`, a `turn` question, and which persona asked it.
Submit an answer with the `session_id` and `turn_id` you got back:
```bash
curl -X POST http://127.0.0.1:8000/interview/answer \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id":1,"turn_id":1,"answer":"I led a migration to a queue-based system..."}'
```
Repeat a few times (each call returns the next question), then get the
panel's verdict:
```bash
curl -X POST http://127.0.0.1:8000/interview/1/deliberate \
  -H "Authorization: Bearer $TOKEN"
```
Each verdict comes back with a `quote` and a `quote_verified` boolean —
that's the hallucination check: it's `false` if the model's quote doesn't
actually appear in the transcript, so you can flag it in the UI instead of
silently trusting it.

Dashboard stats:
```bash
curl http://127.0.0.1:8000/dashboard/stats -H "Authorization: Bearer $TOKEN"
```

Voice analysis (needs a real .wav/.mp3 file):
```bash
curl -X POST http://127.0.0.1:8000/voice/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/answer.wav" \
  -F "transcript=I um led the migration and uh it went well"
```

## 5. If it works, wire it to the frontend

The `hiring-room-app.html` file's mocked JS functions (`submitAuth`,
`handleFile`, `toggleMic`/interview flow, `endInterview`) are the exact
places to swap in real `fetch()` calls to these endpoints instead of the
fake setTimeout logic. Happy to do that wiring next once you've confirmed
this runs on your machine.

## About training / datasets — should you train a model first?

No. Nothing here needs training to work. The panel uses Gemini directly
with a rubric written into each persona's prompt (see `app/panel.py`,
`PERSONA_RUBRIC` and `DELIBERATION_INSTRUCTIONS`) — this is "few-shot
steering," not fine-tuning, and it's enough for a working demo.

The only place a dataset would actually help is *proving your accuracy
number* to judges: build a small set of 15-20 mock interviews with a
verdict you and your team agree is correct, run them through
`/interview/.../deliberate`, and see how often the panel agrees with you.
That's a real, honest evaluation — do it after this is running, not before.
