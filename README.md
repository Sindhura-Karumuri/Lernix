# LERNIX — AI Curriculum Planner

> An AI-powered academic curriculum generation and student learning management platform built with Flask, Groq API (IBM Granite 3.3), RAG, and a ReAct agent.

**Author:** [Sindhura Karumuri](https://github.com/Sindhura-Karumuri)
**Repository:** https://github.com/Sindhura-Karumuri/Lernix

---

## Overview

LERNIX enables educators to generate fully structured, multi-semester academic curricula in seconds using AI. Students can enrol into published curricula, receive personalized weekly study plans, take AI-generated quizzes, track task-level progress, and get course-specific interview preparation — all within a single platform.

The AI layer goes beyond simple prompt-response: curriculum content is indexed into a TF-IDF vector store for Retrieval-Augmented Generation (RAG), and a ReAct-style agent with tool calling autonomously selects and executes tools to answer student questions from actual curriculum data.

Three distinct user roles drive the experience:
- **Educator** — generates and manages AI curricula, monitors student engagement via a tracker
- **Student** — browses and enrols in curricula, follows study plans, takes quizzes, tracks progress
- **Admin** — full system visibility: users, curricula, feedback, download analytics

---

## Features

- AI curriculum generation (title, field, semesters, audience → full structured curriculum)
- Semester-wise course breakdown with codes, credits, outcomes, prerequisites, assessments, and resources
- Visual curriculum roadmap
- Student opt-in / opt-out enrolment
- AI-generated personalized weekly study plans per course
- Task-level checkbox progress tracking (persisted per student)
- AI quiz generation — course-level and semester-level — with score logging and wrong-answer explanations
- AI assistant for course-specific Q&A and interview preparation
- **RAG (Retrieval-Augmented Generation)** — curriculum content indexed into a TF-IDF vector store; top-k relevant course chunks retrieved and injected into LLM prompts
- **Vector store** — in-memory TF-IDF document store with cosine similarity search over curriculum chunks (no external DB dependency)
- **Tool calling** — LLM selects from 4 tools (`get_course_info`, `search_curriculum`, `get_career_paths`, `get_quiz_hint`) and invokes them with structured inputs
- **ReAct agent** — Thought / Action / Observation loop (max 3 iterations) that answers student questions by reasoning over real curriculum data
- Multi-format export: PDF, DOCX, JSON, CSV
- AI-powered standalone academic report generator
- Admin dashboard: system stats, user management, curriculum list, feedback, download analytics
- Subscription tiers: Free (5 generations/month), Monthly, Annual
- Role-based access control with session authentication
- Password reset and profile management

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11 / Flask 3.x |
| Database | SQLite (default), upgradeable to PostgreSQL |
| AI / LLM | Groq API — IBM Granite 3.3 (`ibm-granite/granite-3.3-8b-instruct`) |
| RAG / Vector Store | TF-IDF cosine similarity — Python stdlib (`math`, `re`, `collections`) |
| Agent | ReAct loop with tool calling — `agent.py` |
| PDF Export | ReportLab 4.x |
| DOCX Export | python-docx 1.x |
| Frontend | Jinja2 templates, vanilla JavaScript, CSS |
| Server | Gunicorn (production) |

---

## Project Structure

```
lernix/
├── app.py                  # All Flask routes and API endpoints
├── database.py             # SQLite CRUD operations and schema migration
├── llm_service.py          # Groq API LLM prompt construction and calls
├── rag_engine.py           # TF-IDF vector store and RAG retrieval engine
├── agent.py                # ReAct agent with tool calling loop
├── pdf_generator.py        # PDF, DOCX, and TXT report generation
├── schema.sql              # Full database schema (8 tables)
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn start command for cloud deployment
├── runtime.txt             # Python version pin (3.11.9)
├── .env.example            # Environment variable template
├── static/
│   ├── css/
│   │   └── style.css       # Global stylesheet
│   └── js/
│       ├── main.js         # Core interactions: auth, quiz, assistant, opt-in
│       ├── dashboard.js    # Dashboard dynamic elements
│       └── roadmap.js      # Visual curriculum roadmap renderer
└── templates/
    ├── base.html           # Shared layout (nav, flash messages, footer)
    ├── login.html          # Login and registration
    ├── dashboard.html      # Student / Educator dashboard
    ├── admin_dashboard.html# Admin panel
    ├── generate.html       # Curriculum generation form
    ├── curriculum.html     # Curriculum detail view
    ├── roadmap.html        # Visual roadmap
    ├── assistant.html      # AI study plan + interview prep
    ├── assistant_plan.html # Weekly task checklist
    ├── history.html        # Curriculum history + downloads
    ├── tracker.html        # Educator student tracker
    ├── profile.html        # Profile management
    ├── subscription.html   # Subscription tier management
    ├── feedback.html       # Platform feedback
    ├── faq.html            # FAQ page
    ├── contact.html        # Contact page
    └── forgot_password.html# Password reset
```

---

## AI Architecture

### RAG — Retrieval-Augmented Generation (`rag_engine.py`)

When a student asks a question, the system does not rely solely on the LLM's general knowledge. Instead:

1. All courses in the curriculum (descriptions, outcomes, skills, career paths) are chunked and indexed into a TF-IDF vector store.
2. The student's question is vectorised and compared against all chunks using cosine similarity.
3. The top-3 most relevant course chunks are retrieved and prepended to the LLM prompt as grounded context.

This ensures answers are based on the actual curriculum content rather than generic responses.

### Vector Store (`rag_engine.VectorStore`)

- Built from SQLite curriculum data at query time — no external vector database required.
- Uses TF-IDF weighting and cosine similarity for semantic retrieval.
- `build_store_from_curriculum(curriculum_id)` indexes all courses for a given curriculum.
- `retrieve_context(question, curriculum_id)` returns a formatted context string ready for prompt injection.

### Tool Calling & ReAct Agent (`agent.py`)

The agent exposes 4 tools to the LLM:

| Tool | Description |
|---|---|
| `get_course_info` | Look up a specific course by name — returns description, credits, outcomes, skills, prerequisites |
| `search_curriculum` | Semantic RAG search over all curriculum chunks for a topic or question |
| `get_career_paths` | List career opportunities and key skills for a course or skill area |
| `get_quiz_hint` | Return a sample quiz question and answer for interview preparation |

On each turn the LLM outputs:
```
Thought: <reasoning>
Action: <tool_name>
Action Input: {"param": "value"}
```
The agent executes the tool, feeds the `Observation` back, and repeats up to 3 iterations until the LLM emits a `Final Answer`.

---

## Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com) — sign up and create a key under **API Keys**

> Any Groq-hosted model works. Update `GROQ_MODEL` in your `.env` file if using a different one.

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Sindhura-Karumuri/Lernix.git
cd Lernix

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and fill in SECRET_KEY and GROQ_API_KEY

# 5. Run the application
flask run
```

Open `http://localhost:5000` in your browser.

---

## Default Accounts

| Role | Email | Password |
|---|---|---|
| Admin | admin@lernix.edu | admin123 |
| Educator | educator@lernix.edu | educator123 |
| Student | student@lernix.edu | student123 |

> Change all default passwords before any public or production deployment.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SECRET_KEY` | Flask session signing key | auto-generated (insecure) |
| `FLASK_DEBUG` | Enable debug mode (`1` / `0`) | `0` |
| `DATABASE_PATH` | Path to the SQLite database file | `lernix.db` |
| `GROQ_API_KEY` | Groq API key — get free at [console.groq.com](https://console.groq.com) | required |
| `GROQ_MODEL` | Groq model identifier | `ibm-granite/granite-3.3-8b-instruct` |

Copy `.env.example` to `.env` and fill in values before running.

---

## API Endpoints

| Method | Endpoint | Role | Description |
|---|---|---|---|
| POST | `/api/auth/login` | Public | Authenticate and start session |
| POST | `/api/auth/register` | Public | Create new account |
| POST | `/api/auth/forgot_password` | Public | Reset password by email |
| POST | `/api/generate` | Educator / Admin | Generate AI curriculum |
| GET | `/api/curriculum/<id>` | Any | Fetch curriculum JSON |
| GET | `/api/curriculum/<id>/courses` | Any | List course names |
| POST | `/api/curriculum/optin` | Student | Enrol in curriculum |
| POST | `/api/curriculum/optout` | Student | Remove enrolment |
| POST | `/api/curriculum/<id>/rate` | Any | Rate a curriculum (1–5) |
| POST | `/api/curriculum/delete/<id>` | Educator / Admin | Delete curriculum |
| POST | `/api/assistant/generate` | Any | Generate AI study plan |
| POST | `/api/assistant/task/toggle` | Student | Toggle task completion |
| POST | `/api/assistant/interview/ask` | Any | RAG-augmented AI interview answer |
| POST | `/api/agent/ask` | Any | ReAct agent — tool-calling Q&A over curriculum |
| GET | `/api/quiz/get` | Any | Fetch AI-generated quiz questions |
| POST | `/api/quiz/submit` | Student | Submit quiz score |
| POST | `/api/feedback` | Any | Submit platform feedback |
| POST | `/api/profile/update` | Any | Update profile details |
| POST | `/api/subscription/upgrade` | Any | Change subscription tier |
| POST | `/api/report/generate` | Any | Generate full AI report (PDF/DOCX/TXT) |
| GET | `/api/curriculum/<id>/capstone` | Any | Generate capstone project guidelines |

### Agent endpoint request body

```json
POST /api/agent/ask
{
  "question": "What career paths does the Machine Learning course lead to?",
  "curriculum_id": 1
}
```

---

## Export Formats

| Route | Format | Description |
|---|---|---|
| `/curriculum/<id>/export/pdf` | PDF | Formatted curriculum document |
| `/curriculum/<id>/export/json` | JSON | Machine-readable curriculum data |
| `/curriculum/<id>/export/csv` | CSV | Spreadsheet-compatible course list |
| `/api/report/generate` | PDF / DOCX / TXT / CSV / MD | Full AI-generated academic report |

---

## Deployment (Render / Railway / Heroku)

1. Push to GitHub
2. Create a new Web Service on your chosen platform
3. Set Build Command: `pip install -r requirements.txt`
4. Set Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Add environment variables:
   - `SECRET_KEY` — a long random string (use `python -c "import secrets; print(secrets.token_hex(32))"`)
   - `FLASK_DEBUG` — `0`
   - `DATABASE_PATH` — `lernix.db` or a persistent volume path
   - `GROQ_API_KEY` — your Groq API key from [console.groq.com](https://console.groq.com)
   - `GROQ_MODEL` — `ibm-granite/granite-3.3-8b-instruct`

> SQLite is suitable for demos and single-instance deployments. For production traffic with concurrent users, migrate to PostgreSQL — no application-level changes are required beyond updating the database connection in `database.py`.

---

## Database

The schema is defined in `schema.sql` and auto-applied on first run via `database.init_db()`. Subsequent runs apply migration patches without dropping existing data.

**Tables:**

| Table | Purpose |
|---|---|
| `users` | All users — students, educators, admins |
| `curricula` | AI-generated curriculum records |
| `student_curricula` | Student enrolment (opt-in) linkage |
| `learning_plans` | AI-generated weekly study plans |
| `student_tasks` | Per-task checkbox progress |
| `student_quizzes` | Quiz attempt scores and explanations |
| `download_history` | Export action logs |
| `feedbacks` | Platform feedback and ratings |

---

## License

This project is for academic and educational use.
