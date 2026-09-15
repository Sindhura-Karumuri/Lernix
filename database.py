import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

db_name = os.environ.get("DATABASE_PATH", "lernix.db")
if not os.path.isabs(db_name):
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), db_name)
else:
    DATABASE_PATH = db_name

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(force=False):
    """Initializes/Migrates the database using schema.sql and migration patches."""
    db_exists = os.path.exists(DATABASE_PATH)
    if force or not db_exists:
        conn = get_db_connection()
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())
        
        # Populate defaults
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            admin_pwd = generate_password_hash("admin123")
            cursor.execute(
                "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
                ("admin@lernix.edu", admin_pwd, "System Admin", "admin")
            )
            edu_pwd = generate_password_hash("educator123")
            cursor.execute(
                "INSERT INTO users (email, password_hash, full_name, role, subscription_tier) VALUES (?, ?, ?, ?, ?)",
                ("educator@lernix.edu", edu_pwd, "Prof. Sarah Jenkins", "educator", "monthly")
            )
            stud_pwd = generate_password_hash("student123")
            cursor.execute(
                "INSERT INTO users (email, password_hash, full_name, role, year, branch, roll_no) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("student@lernix.edu", stud_pwd, "Alex Carter", "student", "3rd Year", "Computer Science", "CS-2026-102")
            )
            conn.commit()
        conn.close()
        print("Database initialized successfully.")
    else:
        # DB already exists, apply migration patches for new fields/tables
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Alter user columns if missing
        columns_to_add = [
            ("users", "year", "TEXT"),
            ("users", "branch", "TEXT"),
            ("users", "roll_no", "TEXT"),
            ("users", "subscription_tier", "TEXT CHECK(subscription_tier IN ('free', 'monthly', 'annual')) NOT NULL DEFAULT 'free'"),
            ("users", "generation_count", "INTEGER DEFAULT 0"),
            ("curricula", "satisfaction_score", "INTEGER CHECK(satisfaction_score BETWEEN 1 AND 5) DEFAULT NULL")
        ]
        
        for table, col, col_def in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                conn.commit()
            except sqlite3.OperationalError:
                pass # Already exists
        
        # Create student tracking tables if missing
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_curricula (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            curriculum_id INTEGER NOT NULL,
            opted_in_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(curriculum_id) REFERENCES curricula(id) ON DELETE CASCADE,
            UNIQUE(student_id, curriculum_id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            plan_id INTEGER NOT NULL,
            week_number INTEGER NOT NULL,
            task_index INTEGER NOT NULL,
            completed INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(plan_id) REFERENCES learning_plans(id) ON DELETE CASCADE,
            UNIQUE(student_id, plan_id, week_number, task_index)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            curriculum_id INTEGER NOT NULL,
            course_code TEXT NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            wrong_answers_explanation TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(student_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(curriculum_id) REFERENCES curricula(id) ON DELETE CASCADE
        )
        """)
        conn.commit()
        conn.close()
        print("Database migrated and validated successfully.")

# --- User Auth Operations ---

def create_user(email, password, full_name, role, year=None, branch=None, roll_no=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        pwd_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (email, password_hash, full_name, role, year, branch, roll_no, subscription_tier) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (email, pwd_hash, full_name, role, year, branch, roll_no, 'free')
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None

def authenticate_user(email, password):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None

def update_user_password(email, new_password):
    conn = get_db_connection()
    pwd_hash = generate_password_hash(new_password)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password_hash = ? WHERE email = ?", (pwd_hash, email))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def get_all_users():
    conn = get_db_connection()
    rows = conn.execute('SELECT id, email, full_name, role, subscription_tier, generation_count, created_at FROM users ORDER BY id ASC').fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

def update_student_profile(user_id, full_name, year, branch, roll_no):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET full_name = ?, year = ?, branch = ?, roll_no = ? WHERE id = ?",
        (full_name, year, branch, roll_no, user_id)
    )
    conn.commit()
    conn.close()

# --- Subscription Logic ---

def update_subscription_tier(user_id, tier):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET subscription_tier = ? WHERE id = ?", (tier, user_id))
    conn.commit()
    conn.close()

def increment_generation_count(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET generation_count = generation_count + 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

# --- Curriculum Operations ---

def save_curriculum(user_id, title, field_of_study, duration_semesters, raw_json):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO curricula (user_id, title, field_of_study, duration_semesters, raw_json) VALUES (?, ?, ?, ?, ?)",
        (user_id, title, field_of_study, duration_semesters, raw_json)
    )
    conn.commit()
    curriculum_id = cursor.lastrowid
    conn.close()
    return curriculum_id

def get_curriculum(curriculum_id):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT c.*, u.full_name as author_name FROM curricula c JOIN users u ON c.user_id = u.id WHERE c.id = ?", 
        (curriculum_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_curriculum_history(user_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, title, field_of_study, duration_semesters, satisfaction_score, created_at FROM curricula WHERE user_id = ? ORDER BY created_at DESC", 
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_curricula():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT c.id, c.title, c.field_of_study, c.duration_semesters, c.satisfaction_score, c.created_at, u.full_name as author_name "
        "FROM curricula c JOIN users u ON c.user_id = u.id ORDER BY c.created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def rate_curriculum(curriculum_id, score):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE curricula SET satisfaction_score = ? WHERE id = ?", (score, curriculum_id))
    conn.commit()
    conn.close()

def delete_curriculum(curriculum_id, user_id, user_role):
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_role == 'admin':
        cursor.execute("DELETE FROM curricula WHERE id = ?", (curriculum_id,))
    else:
        # Educators can delete their own generated courses
        cursor.execute("DELETE FROM curricula WHERE id = ? AND user_id = ?", (curriculum_id, user_id))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

# --- Student Opt-In Operations ---

def opt_in_curriculum(student_id, curriculum_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO student_curricula (student_id, curriculum_id) VALUES (?, ?)",
            (student_id, curriculum_id)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def opt_out_curriculum(student_id, curriculum_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM student_curricula WHERE student_id = ? AND curriculum_id = ?",
        (student_id, curriculum_id)
    )
    conn.commit()
    conn.close()

def get_student_opted_curricula(student_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT c.id, c.title, c.field_of_study, c.duration_semesters, sc.opted_in_at "
        "FROM student_curricula sc JOIN curricula c ON sc.curriculum_id = c.id "
        "WHERE sc.student_id = ? ORDER BY sc.opted_in_at DESC",
        (student_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def is_student_opted_in(student_id, curriculum_id):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id FROM student_curricula WHERE student_id = ? AND curriculum_id = ?",
        (student_id, curriculum_id)
    ).fetchone()
    conn.close()
    return row is not None

# --- Student Task Progress Tracking ---

def toggle_student_task(student_id, plan_id, week_number, task_index, completed):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO student_tasks (student_id, plan_id, week_number, task_index, completed) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(student_id, plan_id, week_number, task_index) "
        "DO UPDATE SET completed = excluded.completed, updated_at = CURRENT_TIMESTAMP",
        (student_id, plan_id, week_number, task_index, 1 if completed else 0)
    )
    conn.commit()
    conn.close()

def get_completed_tasks(student_id, plan_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT week_number, task_index FROM student_tasks WHERE student_id = ? AND plan_id = ? AND completed = 1",
        (student_id, plan_id)
    ).fetchall()
    conn.close()
    return [(r['week_number'], r['task_index']) for r in rows]

def get_opted_students_by_curriculum(curriculum_id):
    """Returns student list, details, and task checkbox progress for this curriculum."""
    conn = get_db_connection()
    
    # First get students who opted in
    students = conn.execute(
        "SELECT u.id, u.full_name, u.email, u.year, u.branch, u.roll_no "
        "FROM student_curricula sc JOIN users u ON sc.student_id = u.id "
        "WHERE sc.curriculum_id = ?", (curriculum_id,)
    ).fetchall()
    
    student_list = []
    for s in students:
        s_data = dict(s)
        
        # Calculate checkbox progress across all learning plans under this curriculum
        # Total tasks available
        plans = conn.execute(
            "SELECT id, plan_json FROM learning_plans WHERE curriculum_id = ? AND user_id = ?",
            (curriculum_id, s['id'])
        ).fetchall()
        
        total_tasks = 0
        completed_tasks = 0
        
        for p in plans:
            try:
                plan_data = json.loads(p['plan_json'])
                for week in plan_data.get('weekly_schedule', []):
                    total_tasks += len(week.get('tasks', []))
            except Exception:
                pass
                
            # Count completed
            completed_count = conn.execute(
                "SELECT COUNT(*) FROM student_tasks WHERE student_id = ? AND plan_id = ? AND completed = 1",
                (s['id'], p['id'])
            ).fetchone()[0]
            completed_tasks += completed_count
            
        s_data['total_tasks'] = total_tasks
        s_data['completed_tasks'] = completed_tasks
        s_data['progress_percent'] = round((completed_tasks / total_tasks * 100), 1) if total_tasks > 0 else 0
        
        student_list.append(s_data)
        
    conn.close()
    return student_list

# --- Student Quiz Results ---

def save_quiz_score(student_id, curriculum_id, course_code, score, total_questions, explanation):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO student_quizzes (student_id, curriculum_id, course_code, score, total_questions, wrong_answers_explanation) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (student_id, curriculum_id, course_code, score, total_questions, explanation)
    )
    conn.commit()
    conn.close()

def get_quiz_scores_by_curriculum(curriculum_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT sq.*, u.full_name, u.email, u.roll_no "
        "FROM student_quizzes sq JOIN users u ON sq.student_id = u.id "
        "WHERE sq.curriculum_id = ? ORDER BY sq.timestamp DESC",
        (curriculum_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_quiz_scores_by_student(student_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT sq.*, c.title as curriculum_title "
        "FROM student_quizzes sq JOIN curricula c ON sq.curriculum_id = c.id "
        "WHERE sq.student_id = ? ORDER BY sq.timestamp DESC",
        (student_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Student Learning Assistant Plans ---

def save_learning_plan(curriculum_id, user_id, title, plan_json):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO learning_plans (curriculum_id, user_id, title, plan_json) VALUES (?, ?, ?, ?)",
        (curriculum_id, user_id, title, plan_json)
    )
    conn.commit()
    plan_id = cursor.lastrowid
    conn.close()
    return plan_id

def get_learning_plans_by_user(user_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT lp.id, lp.curriculum_id, lp.title, lp.created_at, c.title as curriculum_title "
        "FROM learning_plans lp JOIN curricula c ON lp.curriculum_id = c.id "
        "WHERE lp.user_id = ? ORDER BY lp.created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_learning_plan(plan_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM learning_plans WHERE id = ?", (plan_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_learning_plan_by_course(student_id, curriculum_id, course_name):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM learning_plans WHERE user_id = ? AND curriculum_id = ? AND title LIKE ?",
        (student_id, curriculum_id, f"%{course_name}%")
    ).fetchone()
    conn.close()
    return dict(row) if row else None

# --- Log / Download Logs & Feedback ---

def log_download(user_id, curriculum_id, export_format):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO download_history (user_id, curriculum_id, export_format) VALUES (?, ?, ?)",
        (user_id, curriculum_id, export_format)
    )
    conn.commit()
    conn.close()

def get_download_history_by_user(user_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT dh.export_format, dh.timestamp, c.title as curriculum_title "
        "FROM download_history dh JOIN curricula c ON dh.curriculum_id = c.id "
        "WHERE dh.user_id = ? ORDER BY dh.timestamp DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_download_stats():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT export_format, COUNT(*) as count FROM download_history GROUP BY export_format"
    ).fetchall()
    conn.close()
    return {r['export_format']: r['count'] for r in rows}

def save_feedback(user_id, rating, comments):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedbacks (user_id, rating, comments) VALUES (?, ?, ?)",
        (user_id, rating, comments)
    )
    conn.commit()
    conn.close()

def get_all_feedbacks():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT f.rating, f.comments, f.timestamp, u.full_name "
        "FROM feedbacks f JOIN users u ON f.user_id = u.id ORDER BY f.timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Admin General Statistics ---

def get_system_stats():
    conn = get_db_connection()
    stats = {}
    stats['total_users'] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    stats['total_curricula'] = conn.execute("SELECT COUNT(*) FROM curricula").fetchone()[0]
    stats['total_plans'] = conn.execute("SELECT COUNT(*) FROM learning_plans").fetchone()[0]
    stats['total_downloads'] = conn.execute("SELECT COUNT(*) FROM download_history").fetchone()[0]
    
    # Calculate avg feedback
    avg_rating = conn.execute("SELECT AVG(rating) FROM feedbacks").fetchone()[0]
    stats['avg_rating'] = round(avg_rating, 1) if avg_rating else 5.0
    
    # User roles split
    roles_rows = conn.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role").fetchall()
    stats['role_distribution'] = {r['role']: r['count'] for r in roles_rows}
    
    conn.close()
    return stats

def get_user_dashboard_stats(user_id, role):
    conn = get_db_connection()
    stats = {}
    
    # 1. Programs Logged
    if role == 'student':
        stats['total_curricula'] = conn.execute(
            "SELECT COUNT(*) FROM student_curricula WHERE student_id = ?", (user_id,)
        ).fetchone()[0]
    elif role == 'educator':
        stats['total_curricula'] = conn.execute(
            "SELECT COUNT(*) FROM curricula WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
    else:
        stats['total_curricula'] = conn.execute("SELECT COUNT(*) FROM curricula").fetchone()[0]
        
    # 2. Study Plans Generated
    stats['total_plans'] = conn.execute(
        "SELECT COUNT(*) FROM learning_plans WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    
    # 3. Exports Executed
    stats['total_downloads'] = conn.execute(
        "SELECT COUNT(*) FROM download_history WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    
    # 4. Satisfaction Score (average rating of individual courses)
    if role == 'student':
        avg_rating = conn.execute(
            "SELECT AVG(c.satisfaction_score) FROM student_curricula sc "
            "JOIN curricula c ON sc.curriculum_id = c.id WHERE sc.student_id = ? AND c.satisfaction_score IS NOT NULL", 
            (user_id,)
        ).fetchone()[0]
        stats['avg_rating'] = round(avg_rating, 1) if avg_rating else 5.0
    elif role == 'educator':
        avg_rating = conn.execute(
            "SELECT AVG(satisfaction_score) FROM curricula WHERE user_id = ? AND satisfaction_score IS NOT NULL",
            (user_id,)
        ).fetchone()[0]
        stats['avg_rating'] = round(avg_rating, 1) if avg_rating else 5.0
    else:
        avg_rating = conn.execute("SELECT AVG(satisfaction_score) FROM curricula WHERE satisfaction_score IS NOT NULL").fetchone()[0]
        stats['avg_rating'] = round(avg_rating, 1) if avg_rating else 5.0
        
    roles_rows = conn.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role").fetchall()
    stats['role_distribution'] = {r['role']: r['count'] for r in roles_rows}
    
    conn.close()
    return stats
