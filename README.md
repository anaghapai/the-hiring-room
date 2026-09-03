# The Hiring Room

An AI-powered mock interview platform. Instead of a single opaque score, a
simulated 3-person hiring panel (HR, Tech Lead, Hiring Manager) each ask you
real interview questions, then each give an independent verdict — and every
verdict's supporting quote is checked against your actual transcript, so it's
not just trusted blindly.

## What you need before starting

- Python 3.10+
- A free Groq API key — get one at https://console.groq.com/keys
- (Optional) A Google OAuth Client ID, only if you want "Sign in with Google"
  — get one at https://console.cloud.google.com/apis/credentials
- `ffmpeg` installed and on your PATH (needed to analyze recorded voice answers):
  - Windows: download from https://ffmpeg.org, add the `bin` folder to PATH
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## 1. Backend setup

```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure environment variables

```bash
copy .env.example .env       # Mac/Linux: cp .env.example .env
```

Open `.env` and fill in:
GROQ_API_KEY=your_real_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b
JWT_SECRET=any_long_random_string
DATABASE_URL=sqlite:///./hiring_room.db
GOOGLE_CLIENT_ID= # optional — leave blank to disable Google Sign-In

## 3. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

Leave this running. Interactive API docs are available at:
`http://127.0.0.1:8000/docs`

## 4. Run the frontend

The frontend is a single static HTML file (`hiring-room-v7.html`) — no build
step. In a **separate** terminal, from the folder containing that file:

```bash
python -m http.server 5500
```

Then open: `http://localhost:5500/hiring-room-v7.html`

The frontend expects the backend at `http://localhost:8000` — if you run the
backend on a different port, update the `API_BASE` constant near the top of
the `<script>` section in the HTML file to match.

## 5. If you enabled Google Sign-In

In the HTML file, find this line near the top of the `<script>` section and
replace the placeholder with your real Client ID:

```js
const GOOGLE_CLIENT_ID = 'YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com';
```

When creating the OAuth Client ID in Google Cloud Console, set the Authorized
JavaScript origin to whatever address you're serving the frontend from (e.g.
`http://localhost:5500`).

## Project structure
app/
main.py — FastAPI app setup, mounts all routers
auth.py — register/login/Google sign-in, JWT tokens
resume.py — resume vs. job-description semantic match
panel.py — the 3-persona interview + verdict deliberation
voice.py — speaking pace / filler words / pause detection from audio
dashboard.py — session history, stats, per-evaluator performance
models.py — database tables & request/response schemas
database.py — SQLite setup
config.py — reads .env


## Notes

- No model training required — the panel uses Groq's LLM directly, steered
  by a rubric written into each persona's prompt (see `PERSONA_RUBRIC` in
  `app/panel.py`), not fine-tuning.
- The database (`hiring_room.db`) is created automatically on first run —
  delete it any time to start fresh with no history.
- Every verdict's `quote` field comes with a `quote_verified` boolean — this
  is checked against the real interview transcript in plain Python, not
  trusted from the AI's output.


