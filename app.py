from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, flash
import os
import json
import csv
import io
import secrets
import logging
import time
from functools import wraps
from datetime import datetime
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

load_dotenv()

# Import database and helper services
import database
import llm_service
import pdf_generator
import agent as lernix_agent

app = Flask(__name__)

_secret = os.environ.get("SECRET_KEY", "")
if not _secret:
    _secret = secrets.token_hex(32)
    if os.environ.get("FLASK_DEBUG", "0") != "1":
        logging.warning("SECRET_KEY not set — using a random key. Sessions will not persist across restarts.")
app.secret_key = _secret

database.init_db()

# --- Rate limiting (simple in-memory per-IP) ---
_login_attempts: dict = {}

def _check_rate_limit(ip: str, max_attempts: int = 10, window: int = 60) -> bool:
    now = time.time()
    attempts = [t for t in _login_attempts.get(ip, []) if now - t < window]
    _login_attempts[ip] = attempts
    if len(attempts) >= max_attempts:
        return False
    _login_attempts[ip].append(now)
    return True

# --- Middleware & Helper Decorators ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def educator_or_admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        if session.get('role') not in ['educator', 'admin']:
            flash("Unauthorized. Educator or Admin privileges required.", "danger")
            return redirect(url_for('dashboard_page'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        if session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        return f(*args, **kwargs)
    return decorated_function

# --- Error Handlers ---

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "message": "Endpoint not found."}), 404
    return render_template('base.html'), 404

@app.errorhandler(500)
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({"success": False, "message": "Internal server error."}), 500
    return render_template('base.html'), 500

# --- Page Controllers ---

@app.route('/')
@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard_page'))
    return render_template('login.html')

@app.route('/forgot_password')
def forgot_password_page():
    return render_template('forgot_password.html')

@app.route('/logout')
def logout_action():
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for('login_page'))

@app.route('/dashboard')
@login_required
def dashboard_page():
    user_id = session['user_id']
    user_role = session['role']

    # Admins get dedicated admin panel
    if user_role == 'admin':
        sys_stats = database.get_system_stats()
        all_users = database.get_all_users()
        all_curricula = database.get_all_curricula()
        feedbacks = database.get_all_feedbacks()
        recent_curricula = all_curricula[:8]
        download_stats = database.get_download_stats()
        debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'
        return render_template('admin_dashboard.html',
            sys_stats=sys_stats, all_users=all_users,
            all_curricula=all_curricula, feedbacks=feedbacks,
            recent_curricula=recent_curricula,
            download_stats=download_stats, debug_mode=debug_mode)

    stats = database.get_user_dashboard_stats(user_id, user_role)
    
    # Students see only opted-in curricula. Educators/Admins see generated ones.
    saved_plans = []
    quizzes = []
    if user_role == 'student':
        user_history = database.get_student_opted_curricula(user_id)
        saved_plans = database.get_learning_plans_by_user(user_id)
        quizzes = database.get_quiz_scores_by_student(user_id)
    else:
        user_history = database.get_user_curriculum_history(user_id)
        
    all_curricula = database.get_all_curricula() if user_role == 'admin' else []
    feedbacks = database.get_all_feedbacks() if user_role == 'admin' else []
    
    user_details = database.get_user_by_id(user_id)
    
    return render_template(
        'dashboard.html', 
        stats=stats, 
        user_history=user_history[:5], 
        all_curricula=all_curricula,
        feedbacks=feedbacks,
        user_details=user_details,
        saved_plans=saved_plans,
        quizzes=quizzes
    )

@app.route('/generate')
@login_required
@educator_or_admin_required
def generate_page():
    user = database.get_user_by_id(session['user_id'])
    return render_template('generate.html', user_details=user)

@app.route('/curriculum/<int:id>')
@login_required
def curriculum_page(id):
    curr = database.get_curriculum(id)
    if not curr:
        flash("Curriculum not found.", "danger")
        return redirect(url_for('dashboard_page'))
    
    parsed_json = json.loads(curr['raw_json'])
    sum_courses = sum(len(sem.get('courses', [])) for sem in parsed_json.get('semesters', []))
    total_credits = sum(sum(int(c.get('credits', 0)) for c in sem.get('courses', [])) for sem in parsed_json.get('semesters', []))
    
    # Check if student is opted in
    opted_in = False
    if session['role'] == 'student':
        opted_in = database.is_student_opted_in(session['user_id'], id)
        
    return render_template(
        'curriculum.html', 
        curriculum=curr, 
        data=parsed_json, 
        sum_courses=sum_courses, 
        total_credits=total_credits,
        opted_in=opted_in
    )

@app.route('/roadmap/<int:id>')
@login_required
def roadmap_page(id):
    curr = database.get_curriculum(id)
    if not curr:
        flash("Curriculum not found.", "danger")
        return redirect(url_for('dashboard_page'))
        
    parsed_json = json.loads(curr['raw_json'])
    return render_template('roadmap.html', curriculum=curr, data=parsed_json)

@app.route('/assistant')
@login_required
def assistant_page():
    # Students see curricula they opted into. Educators see what they generated.
    user_id = session['user_id']
    if session['role'] == 'student':
        # List all educator-generated curricula in the system so students can choose courses
        history = database.get_all_curricula()
    else:
        history = database.get_user_curriculum_history(user_id)
        
    saved_plans = database.get_learning_plans_by_user(user_id)
    return render_template('assistant.html', curricula=history, saved_plans=saved_plans)

@app.route('/assistant/plan/<int:id>')
@login_required
def assistant_plan_page(id):
    plan = database.get_learning_plan(id)
    if not plan or (plan['user_id'] != session['user_id'] and session['role'] != 'admin'):
        flash("Learning plan not found or access restricted.", "danger")
        return redirect(url_for('assistant_page'))
        
    parsed_json = json.loads(plan['plan_json'])
    
    # Get completed checkboxes
    completed_tuples = database.get_completed_tasks(session['user_id'], id)
    # Convert list of tuples (week, idx) to structure matching template checks
    completed_tasks = {}
    for week_num, task_idx in completed_tuples:
        if week_num not in completed_tasks:
            completed_tasks[week_num] = []
        completed_tasks[week_num].append(task_idx)
        
    return render_template(
        'assistant_plan.html', 
        plan=plan, 
        data=parsed_json, 
        completed_tasks=completed_tasks
    )

@app.route('/history')
@login_required
def history_page():
    user_id = session['user_id']
    if session['role'] == 'admin':
        history = database.get_all_curricula()
        opted_ids = set()
    elif session['role'] == 'student':
        history = database.get_all_curricula()  # show all, with opt-in status
        opted = database.get_student_opted_curricula(user_id)
        opted_ids = {c['id'] for c in opted}
    else:
        history = database.get_user_curriculum_history(user_id)
        opted_ids = set()

    download_history = database.get_download_history_by_user(user_id)
    saved_plans = database.get_learning_plans_by_user(user_id) if session['role'] == 'student' else []
    return render_template('history.html', history=history, downloads=download_history,
                           saved_plans=saved_plans, opted_ids=opted_ids)

@app.route('/profile')
@login_required
def profile_page():
    user = database.get_user_by_id(session['user_id'])
    return render_template('profile.html', user=user)

@app.route('/faq')
def faq_page():
    return render_template('faq.html')

@app.route('/feedback')
@login_required
def feedback_page():
    return render_template('feedback.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/subscription')
@login_required
def subscription_page():
    user = database.get_user_by_id(session['user_id'])
    return render_template('subscription.html', user_details=user)

@app.route('/tracker')
@login_required
@educator_or_admin_required
def tracker_page():
    # Fetch curricula created by this educator or all if admin
    user_id = session['user_id']
    if session['role'] == 'admin':
        curricula_list = database.get_all_curricula()
    else:
        curricula_list = database.get_user_curriculum_history(user_id)
        
    tracker_data = []
    for curr in curricula_list:
        opted_students = database.get_opted_students_by_curriculum(curr['id'])
        quiz_scores = database.get_quiz_scores_by_curriculum(curr['id'])
        tracker_data.append({
            "curriculum": curr,
            "students": opted_students,
            "quizzes": quiz_scores
        })
        
    return render_template('tracker.html', tracker_data=tracker_data)

# --- API endpoints ---

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    if not _check_rate_limit(request.remote_addr):
        return jsonify({"success": False, "message": "Too many attempts. Please wait and try again."}), 429
    data = request.json or {}
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required."}), 400
    user = database.authenticate_user(email, password)
    if user:
        session['user_id'] = user['id']
        session['email'] = user['email']
        session['full_name'] = user['full_name']
        session['role'] = user['role']
        return jsonify({"success": True, "message": "Logged in successfully."})
    return jsonify({"success": False, "message": "Invalid email or password."}), 401

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    if not _check_rate_limit(request.remote_addr):
        return jsonify({"success": False, "message": "Too many attempts. Please wait and try again."}), 429
    data = request.json or {}
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name')
    role = data.get('role', 'student')
    
    # Registration details
    year = data.get('year')
    branch = data.get('branch')
    roll_no = data.get('roll_no')
    
    if not email or not password or not full_name:
        return jsonify({"success": False, "message": "All fields are required."}), 400
        
    user_id = database.create_user(email, password, full_name, role, year, branch, roll_no)
    if user_id:
        session['user_id'] = user_id
        session['email'] = email
        session['full_name'] = full_name
        session['role'] = role
        return jsonify({"success": True, "message": "Account created successfully."})
        
    return jsonify({"success": False, "message": "Email address already registered."}), 409

@app.route('/api/auth/forgot_password', methods=['POST'])
def api_forgot_password():
    data = request.json or {}
    email = data.get('email')
    new_password = data.get('new_password')
    
    if not email or not new_password:
        return jsonify({"success": False, "message": "Email and new password are required."}), 400
        
    success = database.update_user_password(email, new_password)
    if success:
        return jsonify({"success": True, "message": "Password updated successfully. You can now login."})
    return jsonify({"success": False, "message": "Email address not found in system."}), 404

@app.route('/api/generate', methods=['POST'])
@login_required
@educator_or_admin_required
def api_generate_curriculum():
    data = request.json or {}
    title = data.get('title')
    field = data.get('field_of_study')
    duration = data.get('duration_semesters')
    audience = data.get('target_audience', 'Undergraduate')
    
    if not title or not field or not duration:
        return jsonify({"success": False, "message": "Title, field of study, and duration semesters are required."}), 400
        
    try:
        duration_int = int(duration)
        if duration_int < 1 or duration_int > 8:
            return jsonify({"success": False, "message": "Duration semesters must be between 1 and 8."}), 400
    except ValueError:
        return jsonify({"success": False, "message": "Invalid duration value."}), 400
        
    # --- Check Subscription generation limits ---
    user = database.get_user_by_id(session['user_id'])
    if user['role'] != 'admin' and user['subscription_tier'] == 'free' and user.get('generation_count', 0) >= 5:
        return jsonify({
            "success": False, 
            "message": "Monthly generation limit reached (Max 5 for Free Tier). Please upgrade to a premium plan!"
        }), 403
        
    # Query AI/Granite client
    curriculum_json = llm_service.generate_curriculum_from_llm(title, field, duration_int, audience)
    
    # Save to database
    curr_id = database.save_curriculum(
        user_id=session['user_id'],
        title=title,
        field_of_study=field,
        duration_semesters=duration_int,
        raw_json=json.dumps(curriculum_json)
    )
    
    # Increment generation count
    database.increment_generation_count(session['user_id'])
    
    return jsonify({
        "success": True,
        "curriculum_id": curr_id,
        "message": "Curriculum generated and logged successfully."
    })

@app.route('/api/curriculum/<int:id>')
@login_required
def api_curriculum_detail(id):
    curr = database.get_curriculum(id)
    if not curr:
        return jsonify({"success": False, "message": "Curriculum not found."}), 404
    return jsonify(json.loads(curr['raw_json']))

@app.route('/api/curriculum/<int:id>/courses')
@login_required
def api_curriculum_courses(id):
    curr = database.get_curriculum(id)
    if not curr:
        return jsonify([]), 404
    parsed_json = json.loads(curr['raw_json'])
    courses = []
    for sem in parsed_json.get('semesters', []):
        for course in sem.get('courses', []):
            courses.append(course.get('name'))
    return jsonify(courses)

@app.route('/api/curriculum/optin', methods=['POST'])
@login_required
def api_curriculum_optin():
    data = request.json or {}
    curriculum_id = data.get('curriculum_id')
    if not curriculum_id:
        return jsonify({"success": False, "message": "Curriculum ID is required."}), 400
        
    success = database.opt_in_curriculum(session['user_id'], curriculum_id)
    if success:
        return jsonify({"success": True, "message": "Opted-in to curriculum successfully."})
    return jsonify({"success": False, "message": "Already opted-in to this curriculum."})

@app.route('/api/curriculum/optout', methods=['POST'])
@login_required
def api_curriculum_optout():
    data = request.json or {}
    curriculum_id = data.get('curriculum_id')
    if not curriculum_id:
        return jsonify({"success": False, "message": "Curriculum ID is required."}), 400
        
    database.opt_out_curriculum(session['user_id'], curriculum_id)
    return jsonify({"success": True, "message": "Opted-out of curriculum successfully."})

@app.route('/api/curriculum/<int:id>/rate', methods=['POST'])
@login_required
def api_curriculum_rate(id):
    data = request.json or {}
    rating = data.get('rating')
    if not rating:
        return jsonify({"success": False, "message": "Rating value required."}), 400
    try:
        rating_int = int(rating)
        database.rate_curriculum(id, rating_int)
        return jsonify({"success": True, "message": "Rating updated successfully."})
    except ValueError:
        return jsonify({"success": False, "message": "Invalid rating score."}), 400

@app.route('/api/curriculum/delete/<int:id>', methods=['POST', 'DELETE'])
@login_required
def api_curriculum_delete(id):
    # If student, delete their opt-in link, else delete curriculum
    if session['role'] == 'student':
        database.opt_out_curriculum(session['user_id'], id)
        return jsonify({"success": True, "message": "Opt-in reference deleted."})
        
    success = database.delete_curriculum(id, session['user_id'], session['role'])
    if success:
        return jsonify({"success": True, "message": "Curriculum deleted successfully."})
    return jsonify({"success": False, "message": "Permission denied or curriculum not found."}), 403

# --- Student assistant updates ---

@app.route('/api/assistant/generate', methods=['POST'])
@login_required
def api_generate_study_plan():
    data = request.json or {}
    curriculum_id = data.get('curriculum_id')
    course_name = data.get('course_name')
    weekly_hours = data.get('weekly_hours', 10)
    
    if not curriculum_id:
        return jsonify({"success": False, "message": "Curriculum ID is required."}), 400
        
    curr = database.get_curriculum(curriculum_id)
    if not curr:
        return jsonify({"success": False, "message": "Associated curriculum not found."}), 404
        
    if not course_name:
        course_name = curr['title']
        
    plan_data = llm_service.generate_study_plan_from_llm(curr['title'], course_name, weekly_hours)
    
    # Save plan to database
    plan_id = database.save_learning_plan(
        curriculum_id=curriculum_id,
        user_id=session['user_id'],
        title=f"Study Plan: {course_name}",
        plan_json=json.dumps(plan_data)
    )
    
    return jsonify({
        "success": True,
        "plan_id": plan_id,
        "message": "Custom student study plan generated successfully."
    })

@app.route('/api/assistant/task/toggle', methods=['POST'])
@login_required
def api_assistant_task_toggle():
    data = request.json or {}
    plan_id = data.get('plan_id')
    week_number = data.get('week_number')
    task_index = data.get('task_index')
    completed = data.get('completed')
    
    if plan_id is None or week_number is None or task_index is None or completed is None:
        return jsonify({"success": False, "message": "Missing arguments."}), 400
        
    database.toggle_student_task(
        student_id=session['user_id'],
        plan_id=int(plan_id),
        week_number=int(week_number),
        task_index=int(task_index),
        completed=bool(completed)
    )
    return jsonify({"success": True, "message": "Task progress updated."})

@app.route('/api/assistant/interview/ask', methods=['POST'])
@login_required
def api_assistant_interview_ask():
    data = request.json or {}
    course_name = data.get('course_name')
    question = data.get('question')
    
    if not course_name or not question:
        return jsonify({"success": False, "message": "Course name and question are required."}), 400
        
    answer = llm_service.generate_custom_interview_answer(
        course_name, question,
        curriculum_id=data.get('curriculum_id')
    )
    return jsonify({"success": True, "answer": answer})

# --- Quiz APIs ---

@app.route('/api/quiz/get')
@login_required
def api_quiz_get():
    course_name = request.args.get('course_name')
    course_names = request.args.get('course_names')  # comma-separated for semester quiz
    week_topic  = request.args.get('week_topic', course_name)
    if not course_name and not course_names:
        return jsonify([]), 400
    if course_names:
        names_list = [n.strip() for n in course_names.split('|') if n.strip()]
        quiz = llm_service.generate_semester_quiz(names_list)
    else:
        quiz = llm_service.generate_course_quiz(week_topic or course_name)
    return jsonify(quiz)

@app.route('/api/quiz/submit', methods=['POST'])
@login_required
def api_quiz_submit():
    data = request.json or {}
    curriculum_id = data.get('curriculum_id')
    course_code = data.get('course_code')
    score = data.get('score')
    total_questions = data.get('total_questions')
    explanation = data.get('wrong_explanations', '')
    
    if not curriculum_id or not course_code or score is None or total_questions is None:
        return jsonify({"success": False, "message": "Missing arguments."}), 400
        
    database.save_quiz_score(
        student_id=session['user_id'],
        curriculum_id=int(curriculum_id),
        course_code=course_code,
        score=int(score),
        total_questions=int(total_questions),
        explanation=explanation
    )
    return jsonify({"success": True, "message": "Quiz attempt logged successfully."})

# --- Feedback & Subscriptions ---

@app.route('/api/feedback', methods=['POST'])
@login_required
def api_feedback():
    data = request.json or {}
    rating = data.get('rating')
    comments = data.get('comments', '')
    
    if not rating:
        return jsonify({"success": False, "message": "Rating score is required."}), 400
        
    try:
        rating_int = int(rating)
        if rating_int < 1 or rating_int > 5:
            return jsonify({"success": False, "message": "Rating score must be between 1 and 5."}), 400
    except ValueError:
        return jsonify({"success": False, "message": "Invalid rating score."}), 400
        
    database.save_feedback(session['user_id'], rating_int, comments)
    return jsonify({"success": True, "message": "Feedback submitted successfully. Thank you!"})

@app.route('/api/profile/update', methods=['POST'])
@login_required
def api_profile_update():
    data = request.json or {}
    full_name = data.get('full_name')
    password = data.get('password')
    
    # Roll, Year, Branch details
    year = data.get('year')
    branch = data.get('branch')
    roll_no = data.get('roll_no')
    
    if session['role'] == 'student':
        database.update_student_profile(session['user_id'], full_name, year, branch, roll_no)
        session['full_name'] = full_name
    else:
        conn = database.get_db_connection()
        cursor = conn.cursor()
        if full_name:
            cursor.execute("UPDATE users SET full_name = ? WHERE id = ?", (full_name, session['user_id']))
            session['full_name'] = full_name
        conn.commit()
        conn.close()
        
    if password:
        pwd_hash = generate_password_hash(password)
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (pwd_hash, session['user_id']))
        conn.commit()
        conn.close()
        
    return jsonify({"success": True, "message": "Profile updated successfully."})

@app.route('/api/subscription/upgrade', methods=['POST'])
@login_required
def api_subscription_upgrade():
    data = request.json or {}
    tier = data.get('tier')
    if tier not in ['monthly', 'annual', 'free']:
        return jsonify({"success": False, "message": "Invalid plan tier."}), 400
        
    database.update_subscription_tier(session['user_id'], tier)
    return jsonify({"success": True, "message": f"Successfully updated subscription to {tier.capitalize()} plan."})

# --- Export Routes ---

@app.route('/curriculum/<int:id>/export/pdf')
@login_required
def export_pdf(id):
    curr = database.get_curriculum(id)
    if not curr:
        flash("Curriculum not found.", "danger")
        return redirect(url_for('dashboard_page'))
        
    database.log_download(session['user_id'], id, 'pdf')
    parsed_json = json.loads(curr['raw_json'])
    pdf_bytes = pdf_generator.generate_curriculum_pdf(parsed_json)
    
    filename = f"{curr['title'].replace(' ', '_')}_curriculum.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )

@app.route('/curriculum/<int:id>/export/json')
@login_required
def export_json(id):
    curr = database.get_curriculum(id)
    if not curr:
        flash("Curriculum not found.", "danger")
        return redirect(url_for('dashboard_page'))
        
    database.log_download(session['user_id'], id, 'json')
    
    # Backfill resources if missing in raw_json to ensure no blank resources
    parsed_json = json.loads(curr['raw_json'])
    updated = False
    for sem in parsed_json.get("semesters", []):
        for course in sem.get("courses", []):
            if "resources" not in course or not course["resources"]:
                skills = course.get("key_skills", ["IT"])
                skill_name = skills[0] if skills else "Technology"
                course["resources"] = [
                    {"title": f"ACM/IEEE Guidelines on {course.get('name', 'this course')}", "url": "https://www.acm.org"},
                    {"title": f"IBM Cognitive Class: {skill_name}", "url": "https://cognitiveclass.ai"}
                ]
                updated = True
                
    raw_json_output = json.dumps(parsed_json, indent=2) if updated else curr['raw_json']
    
    filename = f"{curr['title'].replace(' ', '_')}_curriculum.json"
    return send_file(
        io.BytesIO(raw_json_output.encode('utf-8')),
        mimetype='application/json',
        as_attachment=True,
        download_name=filename
    )

@app.route('/curriculum/<int:id>/export/csv')
@login_required
def export_csv(id):
    curr = database.get_curriculum(id)
    if not curr:
        flash("Curriculum not found.", "danger")
        return redirect(url_for('dashboard_page'))
        
    database.log_download(session['user_id'], id, 'csv')
    parsed_json = json.loads(curr['raw_json'])
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Curriculum", "Field of Study", "Semester", "Course Code", 
        "Course Name", "Credits", "Difficulty", "Weekly Hours", 
        "Learning Outcomes", "Prerequisites", "Assessment Methods",
        "Reference Resources"
    ])
    
    for sem in parsed_json.get('semesters', []):
        sem_num = sem.get('semester_number')
        for course in sem.get('courses', []):
            # Retrieve or backfill course resources
            resources = course.get('resources')
            if not resources:
                skills = course.get("key_skills", ["IT"])
                skill_name = skills[0] if skills else "Technology"
                resources = [
                    {"title": f"ACM/IEEE Guidelines on {course.get('name', 'this course')}", "url": "https://www.acm.org"},
                    {"title": f"IBM Cognitive Class: {skill_name}", "url": "https://cognitiveclass.ai"}
                ]
            resources_str = "; ".join([f"{r['title']} ({r['url']})" for r in resources])
            
            writer.writerow([
                parsed_json.get('title'),
                parsed_json.get('field_of_study'),
                f"Semester {sem_num}",
                course.get('code'),
                course.get('name'),
                course.get('credits'),
                course.get('difficulty'),
                course.get('weekly_hours'),
                "; ".join(course.get('learning_outcomes', [])),
                "; ".join(course.get('prerequisites', [])),
                "; ".join(course.get('assessment_methods', [])),
                resources_str
            ])
            
    output.seek(0)
    filename = f"{curr['title'].replace(' ', '_')}_curriculum.csv"
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

# --- AI-Powered Standalone Report Generator ---

@app.route('/api/report/generate', methods=['POST'])
@login_required
def api_report_generate():
    """
    Fully AI-driven report generator.
    Accepts a free-text topic + format, builds everything from scratch.
    """
    data = request.json or {}
    topic  = (data.get('topic') or '').strip()
    fmt    = (data.get('format') or 'pdf').lower()
    curr_id = data.get('curriculum_id')       # optional — used to seed curriculum data

    if not topic:
        return jsonify({"success": False, "message": "Topic is required."}), 400

    # Build curriculum base: use existing curriculum if provided, else generate from topic
    if curr_id:
        curr = database.get_curriculum(int(curr_id))
        curriculum_data = json.loads(curr['raw_json']) if curr else None
    else:
        curriculum_data = None

    if not curriculum_data:
        curriculum_data = llm_service.generate_curriculum_from_llm(
            title=topic, field=topic, duration=4
        )

    # Generate enriched report sections from the topic
    report = llm_service.generate_full_report_data(topic, curriculum_data)

    safe = topic.replace(' ', '_').replace('/', '-')
    database.log_download(session['user_id'], curr_id or 0, fmt)

    if fmt == 'pdf':
        pdf_bytes = pdf_generator.generate_professional_report(topic, curriculum_data, report)
        return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf',
                         as_attachment=True, download_name=f"{safe}.pdf")

    if fmt == 'docx':
        docx_bytes = pdf_generator.generate_docx_report(topic, curriculum_data, report)
        return send_file(io.BytesIO(docx_bytes),
                         mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                         as_attachment=True, download_name=f"{safe}.docx")

    if fmt == 'txt':
        txt = pdf_generator.generate_txt_report(topic, curriculum_data, report)
        return send_file(io.BytesIO(txt.encode('utf-8')), mimetype='text/plain',
                         as_attachment=True, download_name=f"{safe}.txt")

    if fmt == 'md':
        md = pdf_generator.generate_md_report(topic, curriculum_data, report)
        return send_file(io.BytesIO(md.encode('utf-8')), mimetype='text/markdown',
                         as_attachment=True, download_name=f"{safe}.md")

    if fmt == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["LERNIX AI — Professional Curriculum Report"])
        writer.writerow(["Topic", topic])
        writer.writerow(["Field", curriculum_data.get('field_of_study', topic)])
        writer.writerow(["Duration", f"{curriculum_data.get('duration_semesters',4)} Semesters"])
        writer.writerow(["Generated By", "LERNIX AI"])
        writer.writerow(["Date", datetime.now().strftime('%Y-%m-%d')])
        writer.writerow([])
        writer.writerow(["== SEMESTER-WISE CURRICULUM =="])
        writer.writerow(["Semester","Code","Course","Credits","Difficulty",
                         "Weekly Hrs","Relevance","Prerequisites",
                         "Learning Outcomes","Career Pathways","Capstone","Assessments"])
        for sem in curriculum_data.get('semesters', []):
            for c in sem.get('courses', []):
                writer.writerow([
                    f"Semester {sem.get('semester_number')}",
                    c.get('code',''), c.get('name',''), c.get('credits',''),
                    c.get('difficulty',''), c.get('weekly_hours',''),
                    f"{c.get('industry_relevance_score',90)}%",
                    "; ".join(c.get('prerequisites',[])),
                    "; ".join(c.get('learning_outcomes',[])),
                    "; ".join(c.get('career_opportunities',[])),
                    "; ".join(c.get('recommended_projects',[])),
                    "; ".join(c.get('assessment_methods',[]))
                ])
        writer.writerow([])
        writer.writerow(["== SKILLS =="])
        for cat, items in report.get('skills_mapping', {}).items():
            writer.writerow([cat.replace('_',' ').title(), "; ".join(items)])
        writer.writerow([])
        writer.writerow(["== CAREER ROADMAP =="])
        writer.writerow(["Level","Role","Description","Salary"])
        for level, roles in report.get('career_roadmap', {}).items():
            for r in (roles if isinstance(roles, list) else []):
                writer.writerow([level.replace('_',' ').title(),
                                  r.get('title',''), r.get('description',''), r.get('salary','')])
        writer.writerow([])
        writer.writerow(["== CAPSTONE PROJECTS =="])
        writer.writerow(["Title","Description","Tech Stack","Deliverables","Difficulty"])
        for p in report.get('capstone_projects', []):
            writer.writerow([p.get('title',''), p.get('description',''),
                              "; ".join(p.get('tech_stack',[])),
                              "; ".join(p.get('deliverables',[])),
                              p.get('difficulty','')])
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                         mimetype='text/csv', as_attachment=True,
                         download_name=f"{safe}.csv")

    return jsonify({"success": False, "message": f"Unsupported format: {fmt}"}), 400

# --- Capstone & Custom Export APIs ---

@app.route('/api/curriculum/<int:id>/capstone')
@login_required
def api_capstone(id):
    curr = database.get_curriculum(id)
    if not curr:
        return jsonify({"success": False, "message": "Curriculum not found."}), 404
    parsed = json.loads(curr['raw_json'])
    courses_summary = ", ".join(
        course.get('name', '')
        for sem in parsed.get('semesters', [])
        for course in sem.get('courses', [])
    )
    guidelines = llm_service.generate_capstone_guidelines(
        curr['title'], curr['field_of_study'], courses_summary
    )
    return jsonify({"success": True, "capstone": guidelines})

@app.route('/api/curriculum/<int:id>/export/custom', methods=['POST'])
@login_required
def export_custom(id):
    curr = database.get_curriculum(id)
    if not curr:
        return jsonify({"success": False, "message": "Curriculum not found."}), 404

    data = request.json or {}
    topic = data.get('topic', curr['title']).strip() or curr['title']
    fmt = data.get('format', 'pdf').lower()

    parsed_json = json.loads(curr['raw_json'])
    database.log_download(session['user_id'], id, fmt)
    safe_topic = topic.replace(' ', '_').replace('/', '-')

    if fmt == 'json':
        for sem in parsed_json.get('semesters', []):
            for course in sem.get('courses', []):
                if not course.get('resources'):
                    course['resources'] = [
                        {"title": f"ACM/IEEE on {course.get('name','')}", "url": "https://www.acm.org"},
                        {"title": "IBM Cognitive Class", "url": "https://cognitiveclass.ai"}
                    ]
        parsed_json['export_topic'] = topic
        # Enrich with full report data
        report = llm_service.generate_full_report_data(topic, parsed_json)
        parsed_json['report'] = report
        return send_file(
            io.BytesIO(json.dumps(parsed_json, indent=2).encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=f"{safe_topic}.json"
        )

    if fmt == 'csv':
        report = llm_service.generate_full_report_data(topic, parsed_json)
        output = io.StringIO()
        writer = csv.writer(output)
        # Cover info
        writer.writerow(["LERNIX Professional Curriculum Report"])
        writer.writerow(["Topic", topic])
        writer.writerow(["Field", parsed_json.get('field_of_study', '')])
        writer.writerow(["Duration", f"{parsed_json.get('duration_semesters','')} Semesters"])
        writer.writerow(["Generated By", "LERNIX AI"])
        writer.writerow(["Date", datetime.now().strftime('%Y-%m-%d')])
        writer.writerow([])
        # Courses
        writer.writerow(["== SEMESTER-WISE CURRICULUM =="])
        writer.writerow(["Semester", "Code", "Course", "Credits", "Difficulty",
                         "Weekly Hours", "Relevance", "Prerequisites",
                         "Learning Outcomes", "Career Pathways", "Capstone", "Assessments"])
        for sem in parsed_json.get('semesters', []):
            for c in sem.get('courses', []):
                writer.writerow([
                    f"Semester {sem.get('semester_number')}",
                    c.get('code'), c.get('name'), c.get('credits'),
                    c.get('difficulty'), c.get('weekly_hours'),
                    f"{c.get('industry_relevance_score',90)}%",
                    "; ".join(c.get('prerequisites', [])),
                    "; ".join(c.get('learning_outcomes', [])),
                    "; ".join(c.get('career_opportunities', [])),
                    "; ".join(c.get('recommended_projects', [])),
                    "; ".join(c.get('assessment_methods', []))
                ])
        writer.writerow([])
        # Skills
        writer.writerow(["== SKILLS MAPPING =="])
        sm = report.get('skills_mapping', {})
        for cat, items in sm.items():
            writer.writerow([cat.replace('_',' ').title(), "; ".join(items)])
        writer.writerow([])
        # Career
        writer.writerow(["== CAREER ROADMAP =="])
        writer.writerow(["Level", "Role", "Description", "Salary"])
        cr = report.get('career_roadmap', {})
        for level, roles in cr.items():
            for r in roles:
                writer.writerow([level.replace('_',' ').title(),
                                  r.get('title',''), r.get('description',''), r.get('salary','')])
        writer.writerow([])
        # Capstone
        writer.writerow(["== CAPSTONE PROJECTS =="])
        writer.writerow(["Title", "Description", "Tech Stack", "Deliverables", "Difficulty"])
        for p in report.get('capstone_projects', []):
            writer.writerow([
                p.get('title',''), p.get('description',''),
                "; ".join(p.get('tech_stack',[])),
                "; ".join(p.get('deliverables',[])),
                p.get('difficulty','')
            ])
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"{safe_topic}.csv"
        )

    # Default: Professional PDF
    report = llm_service.generate_full_report_data(topic, parsed_json)
    pdf_bytes = pdf_generator.generate_professional_report(topic, parsed_json, report)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"{safe_topic}.pdf"
    )

@app.route('/api/agent/ask', methods=['POST'])
@login_required
def api_agent_ask():
    """
    ReAct agent endpoint. The agent autonomously selects tools (get_course_info,
    search_curriculum, get_career_paths, get_quiz_hint) to answer the question.
    """
    data = request.json or {}
    question = (data.get('question') or '').strip()
    curriculum_id = data.get('curriculum_id')

    if not question or not curriculum_id:
        return jsonify({"success": False, "message": "question and curriculum_id are required."}), 400

    answer = lernix_agent.run_agent(question, int(curriculum_id))
    return jsonify({"success": True, "answer": answer})


# --- Admin APIs ---

@app.route('/api/admin/user/role', methods=['POST'])
@login_required
@admin_required
def api_admin_update_role():
    data = request.json or {}
    user_id = data.get('user_id')
    role = data.get('role')
    if not user_id or role not in ['student', 'educator', 'admin']:
        return jsonify({'success': False, 'message': 'Invalid data.'}), 400
    conn = database.get_db_connection()
    conn.execute('UPDATE users SET role = ? WHERE id = ?', (role, user_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': f'Role updated to {role}.'})

@app.route('/api/admin/user/delete', methods=['POST'])
@login_required
@admin_required
def api_admin_delete_user():
    data = request.json or {}
    user_id = data.get('user_id')
    if not user_id or int(user_id) == session['user_id']:
        return jsonify({'success': False, 'message': 'Cannot delete yourself.'}), 400
    conn = database.get_db_connection()
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'User deleted successfully.'})

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
