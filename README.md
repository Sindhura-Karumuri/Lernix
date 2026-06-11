# LERNIX — AI Curriculum Planner

An AI-powered curriculum and study plan generator built with Flask + local LLM (Ollama/IBM Granite).

## Stack
- **Backend:** Python / Flask
- **Database:** SQLite (local), upgradeable to PostgreSQL
- **AI:** Ollama (local LLM) — IBM Granite model
- **PDF Export:** Reportlab
- **Frontend:** Jinja2 templates, vanilla CSS

---

## Local Setup

```bash
# 1. Clone
git clone [https://github.com/Sindhura-Karumuri/Lernix.git]
cd Lernix

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set SECRET_KEY to a strong random value

# 5. Run
flask run
```

Open `http://localhost:5000`

### Default Accounts
| Role | Email | Password |
|---|---|---|
| Admin | admin@lernix.edu | admin123 |
| Educator | educator@lernix.edu | educator123 |
| Student | student@lernix.edu | student123 |

> **Change all default passwords before deploying publicly.**

---

## Deployment (Render / Railway / Heroku)

1. Push to GitHub
2. Create a new Web Service on your platform
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Add environment variables:
   - `SECRET_KEY` — a long random string
   - `FLASK_DEBUG` — `0`
   - `DATABASE_PATH` — `lernix.db` (or a persistent volume path)

> **Note:** SQLite works for demos. For production traffic, migrate to PostgreSQL.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SECRET_KEY` | Flask session encryption key | auto-generated |
| `FLASK_DEBUG` | Enable debug mode (`1`/`0`) | `0` |
| `DATABASE_PATH` | Path to SQLite database file | `lernix.db` |
| `OLLAMA_BASE_URL` | Ollama LLM server URL | `http://localhost:11434` |

---

## Project Structure

```
lernix/
├── app.py              # Flask routes
├── database.py         # SQLite operations
├── llm_service.py      # Ollama LLM calls
├── pdf_generator.py    # PDF export
├── schema.sql          # DB schema
├── requirements.txt
├── Procfile            # Gunicorn start command
├── runtime.txt         # Python version
├── .env.example        # Environment variable template
├── static/
│   ├── css/style.css
│   └── js/
└── templates/          # Jinja2 HTML templates
```
