import json
import re
import random
import os

import rag_engine

from groq import Groq

_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "ibm-granite/granite-3.3-8b-instruct")


def _call_groq(prompt: str, temperature: float = 0.3, timeout: int = 30) -> str:
    """Single Groq API call. Returns response text or empty string on failure."""
    try:
        response = _groq_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=4096,
            timeout=timeout,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return ""

def generate_curriculum_from_llm(title, field, duration, target_audience="Undergraduate"):
    """
    Asks the local Ollama LLM to generate a full structured curriculum in JSON.
    Falls back to mock data if Ollama is unavailable or fails to output correct JSON.
    """
    prompt = f"""
    You are an expert Educational Technology Architect. Generate a complete, industry-aligned semester-wise curriculum.
    
    Details:
    - Title: {title}
    - Field of Study: {field}
    - Duration: {duration} Semesters
    - Target Audience: {target_audience}
    
    You must output a single JSON object. Do not wrap the JSON in ```json``` blocks. Do not add any conversational text before or after the JSON.
    
    JSON Schema:
    {{
      "title": "{title}",
      "field_of_study": "{field}",
      "duration_semesters": {duration},
      "semesters": [
        {{
          "semester_number": 1,
          "courses": [
            {{
              "code": "CS-101",
              "name": "Course Name",
              "description": "Course description (at least 2 sentences)",
              "credits": 4,
              "learning_outcomes": ["outcome 1", "outcome 2"],
              "prerequisites": [],
              "assessment_methods": ["Midterm Exam: 30%", "Final Exam: 40%", "Lab Work: 30%"],
              "difficulty": "Beginner|Intermediate|Advanced",
              "learning_time_weeks": 15,
              "weekly_hours": 6,
              "industry_relevance_score": 95,
              "career_opportunities": ["Job Title 1", "Job Title 2"],
              "recommended_projects": ["Project Idea 1"],
              "key_skills": ["Skill 1", "Skill 2"]
            }}
          ]
        }}
      ]
    }}
    
    Make sure:
    - Every semester has exactly 3-4 courses.
    - Course prerequisites refer to course names/codes from previous semesters.
    - Realistic difficulty level, credits (3 or 4), learning hours, and career opportunities.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": DEFAULT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            },
            timeout=15
        )
        if response.status_code == 200:
            raw_text = response.json().get("response", "").strip()
            cleaned_text = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.MULTILINE).strip()
            
            parsed_json = json.loads(cleaned_text)
            
            # Post-process to ensure course-level resources are set
            for sem in parsed_json.get("semesters", []):
                for course in sem.get("courses", []):
                    if "resources" not in course or not course["resources"]:
                        skills = course.get("key_skills", ["IT"])
                        skill_name = skills[0] if skills else "Technology"
                        course_name_url = course.get('name', 'course').replace(' ', '+')
                        course["resources"] = [
                            {"title": f"Google Scholar: {course.get('name', 'this course')}", "url": f"https://scholar.google.com/scholar?q={course_name_url}"},
                            {"title": f"Coursera: {skill_name}", "url": f"https://www.coursera.org/search?query={skill_name.replace(' ', '%20')}"}
                        ]
            
            parsed_json["resource_sources"] = [
                "ACM/IEEE Joint Curriculum Guidelines",
                "IBM Skills Network Course Frameworks",
                "Hugging Face Applied NLP Curriculum",
                "LERNIX AI Engine via Granite 3.3 2B"
            ]
            return parsed_json
    except Exception as e:
        print(f"Ollama connection or execution failed: {e}. Falling back to high-quality mock curriculum.")
        
    return generate_mock_curriculum(title, field, duration)

def generate_study_plan_from_llm(curriculum_title, course_name, weekly_hours=10):
    """
    Asks the Ollama LLM to generate a customized study plan for a particular course.
    Falls back to mock data if Ollama is unavailable.
    """
    prompt = f"""
    You are an AI Student Assistant. Generate a structured student study guide and schedule for the course: "{course_name}" which belongs to the curriculum "{curriculum_title}".
    The student can dedicate {weekly_hours} hours per week.
    
    Output a single JSON object. Do not wrap in markdown syntax. Do not output anything other than raw JSON.
    
    JSON Schema:
    {{
      "course_name": "{course_name}",
      "weekly_hours_allocated": {weekly_hours},
      "weekly_schedule": [
        {{
          "week": 1,
          "topic": "Topic Name",
          "tasks": ["Study concept X (3 hrs)", "Watch video tutorial Y (2 hrs)", "Practice lab Z (5 hrs)"],
          "resources": [
             {{"title": "Resource Name", "url": "https://example.com/topic"}}
          ],
          "milestone": "Milestone description"
        }}
      ],
      "revision_roadmap": [
        {{
          "phase": "Phase 1: Foundation",
          "timeline": "Weeks 1-4",
          "focus": "Core principles",
          "checkpoints": ["Checkpoint 1"]
        }}
      ],
      "interview_preparation": [
        {{
          "topic": "Core Concept X",
          "sample_questions": [
            {{
              "question": "Sample interview question?",
              "answer": "Accurate, concise industry answer."
            }}
          ]
        }}
      ]
    }}
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": DEFAULT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4
                }
            },
            timeout=15
        )
        if response.status_code == 200:
            raw_text = response.json().get("response", "").strip()
            cleaned_text = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.MULTILINE).strip()
            
            parsed_json = json.loads(cleaned_text)
            parsed_json["resource_sources"] = [
                "LeetCode Interview Guides",
                "GeeksforGeeks Academic Prep Sheets",
                "O'Reilly Learning Pathways",
                "LERNIX Student Assistant Agent"
            ]
            return parsed_json
    except Exception as e:
        print(f"Ollama study plan failed: {e}. Falling back to mock assistant data.")
        
    return generate_mock_study_plan(course_name, weekly_hours)

def generate_custom_interview_answer(course_name, question, **kwargs):
    """
    Queries Ollama to answer a custom student question about a course.
    Falls back to a structured mock answer if Ollama is unreachable.
    """
    q_lower = question.lower()

    # Relevance gate: reject off-topic questions
    course_keywords = set(course_name.lower().split())
    # Question must have BOTH a question word AND a technical/academic subject term
    question_words = {"what", "how", "why", "explain", "define", "difference", "compare", "describe", "is", "are", "does"}
    technical_words = {
        "algorithm", "function", "class", "method", "variable", "data", "type",
        "loop", "array", "list", "object", "model", "network", "database", "query",
        "api", "design", "pattern", "complexity", "sort", "search", "tree", "graph",
        "stack", "queue", "pointer", "memory", "process", "thread", "security",
        "encrypt", "protocol", "syntax", "compile", "debug", "test", "deploy",
        "framework", "library", "module", "concept", "principle", "implement",
        "agile", "scrum", "obe", "outcome", "prerequisite", "curriculum",
        "machine", "learning", "neural", "regression", "cluster", "cloud",
        "tuple", "integer", "string", "boolean", "inheritance", "polymorphism",
        "recursion", "iteration", "index", "hash", "binary", "sql", "nosql",
        "http", "rest", "json", "xml", "python", "java", "javascript", "c++"
    }
    q_words = set(q_lower.split())
    has_question_word = bool(q_words & question_words)
    has_technical_word = bool(q_words & technical_words) or bool(q_words & course_keywords)
    if not (has_question_word and has_technical_word):
        return "Sorry, I can't answer that. Please ask a question related to your course or academic topic."

    # Predefined high-quality answers for common topics
    common_qa_map = {
        "tuple": "In Python, lists are mutable (can be altered in-place), whereas tuples are immutable. Lists use square brackets `[]` while tuples use parentheses `()`. Tuples are faster and safer for write-protected data.",
        "complexity": "The average time complexity of searching in a Hash Table is O(1). This is achieved via a hash function mapping keys to bucket indices. Worst-case degrades to O(n) under severe collisions.",
        "hash table": "The average time complexity of searching in a Hash Table is O(1). This is achieved via a hash function mapping keys to bucket indices. Worst-case degrades to O(n) under severe collisions.",
        "self": "In Python, `self` represents the current instance of the class. It binds attributes and methods to the specific object, allowing access to instance variables within the class definition.",
        "obe": "Outcome-Based Education (OBE) structures curriculum backward from desired graduate competencies, directly aligning learning activities and assessments to verify student capabilities.",
        "outcome-based": "Outcome-Based Education (OBE) structures curriculum backward from desired graduate competencies, directly aligning learning activities and assessments to verify student capabilities.",
        "prerequisite": "A prerequisite is a foundational course that must be completed before enrolling in a more advanced course, ensuring students have the required background knowledge.",
        "agile": "An Agile iteration is a short time-boxed cycle (1-4 weeks) where a team designs, builds, and tests a working product increment, enabling continuous inspection and adaptation."
    }

    for key, ans in common_qa_map.items():
        if key in q_lower:
            return ans

    # Try Ollama — augment prompt with RAG context if curriculum_id provided
    rag_context = ""
    curriculum_id = kwargs.get("curriculum_id") if kwargs else None
    if curriculum_id:
        rag_context = rag_engine.retrieve_context(question, curriculum_id, top_k=3)

    context_block = f"\n{rag_context}\n" if rag_context else ""
    prompt = f"""You are LERNIX AI tutor. Answer the student's interview preparation question about the course '{course_name}'.{context_block}
Question: {question}
Provide a concise, technically accurate response of maximum 4 sentences."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.4}},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        pass

    # Heuristic fallback — only for genuinely academic questions (already passed relevance gate)
    if "how" in q_lower or "explain" in q_lower:
        return f"In {course_name}, this concept is implemented by establishing core environment parameters, writing modular functions, and performing unit verification to guarantee scalability."
    elif "difference" in q_lower or "vs" in q_lower:
        return f"The key difference lies in resource efficiency and data isolation. One approach provides direct access while the other ensures thread safety and security at a minor performance cost."
    else:
        return f"In {course_name}, this is addressed by declaring clean abstractions, validating inputs, handling errors structurally, and following industry coding standards."

def generate_course_quiz(course_name):
    """
    Returns 3 topic-specific quiz questions matched to the week topic.
    Covers all weekly topics generated by generate_mock_study_plan.
    """
    t = course_name.lower()

    # ── Python / Programming weekly topics ─────────────────────────
    if "environment setup" in t or "syntax basics" in t:
        return [
            {"question": "Which command checks your installed Python version?",
             "options": ["python --check", "python --version", "py --info", "python -list"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! `python --version` prints the installed Python version.", "Incorrect.", "Incorrect."]},
            {"question": "What is the correct way to declare a variable in Python?",
             "options": ["int x = 5", "var x = 5", "x = 5", "declare x = 5"],
             "correct_index": 2,
             "explanations": ["Incorrect — that's Java syntax.", "Incorrect — that's JavaScript.", "Correct! Python uses dynamic typing with simple assignment.", "Incorrect."]},
            {"question": "Which of these is NOT a valid Python data type?",
             "options": ["int", "float", "char", "str"],
             "correct_index": 2,
             "explanations": ["Incorrect — int is valid.", "Incorrect — float is valid.", "Correct! Python has no `char` type; single characters are strings.", "Incorrect — str is valid."]},
        ]

    if "functions" in t and ("modules" in t or "data structures" in t):
        return [
            {"question": "What keyword defines a function in Python?",
             "options": ["func", "def", "function", "lambda"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! `def` is used to define functions.", "Incorrect — that's JavaScript.", "Incorrect — lambda creates anonymous one-liners."]},
            {"question": "Which data structure stores key-value pairs in Python?",
             "options": ["List", "Tuple", "Dictionary", "Set"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect.", "Correct! Dictionaries use key:value pairs.", "Incorrect."]},
            {"question": "How do you import the `math` module in Python?",
             "options": ["include math", "using math", "import math", "require math"],
             "correct_index": 2,
             "explanations": ["Incorrect — C/C++ syntax.", "Incorrect.", "Correct! `import math` loads the module.", "Incorrect — Node.js syntax."]},
        ]

    if "object-oriented" in t or "oop" in t:
        return [
            {"question": "What does encapsulation mean in OOP?",
             "options": ["Inheriting from a parent class", "Hiding internal state and requiring all interaction through methods", "Overriding parent methods", "Creating multiple instances"],
             "correct_index": 1,
             "explanations": ["Incorrect — that's inheritance.", "Correct! Encapsulation bundles data with methods that operate on it.", "Incorrect — that's polymorphism/overriding.", "Incorrect."]},
            {"question": "Which Python method is the constructor of a class?",
             "options": ["__start__", "__new__", "__init__", "__create__"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect — __new__ allocates memory but isn't the constructor body.", "Correct! __init__ initialises instance attributes.", "Incorrect."]},
            {"question": "What is polymorphism in OOP?",
             "options": ["A class inheriting multiple parents", "An object having multiple variables", "The ability to use the same interface for different data types", "Hiding class implementation details"],
             "correct_index": 2,
             "explanations": ["Incorrect — that's multiple inheritance.", "Incorrect.", "Correct! Polymorphism lets one interface represent different underlying forms.", "Incorrect — that's encapsulation."]},
        ]

    if "file handling" in t or "apis" in t or "error management" in t:
        return [
            {"question": "Which Python mode opens a file for writing only?",
             "options": ["'r'", "'w'", "'a'", "'x'"],
             "correct_index": 1,
             "explanations": ["Incorrect — 'r' is read-only.", "Correct! 'w' opens for writing, overwriting the file.", "Incorrect — 'a' appends.", "Incorrect — 'x' creates a new file."]},
            {"question": "Which HTTP method is typically used to retrieve data from an API?",
             "options": ["POST", "PUT", "GET", "DELETE"],
             "correct_index": 2,
             "explanations": ["Incorrect — POST creates data.", "Incorrect — PUT updates data.", "Correct! GET retrieves resources.", "Incorrect — DELETE removes data."]},
            {"question": "What block handles exceptions in Python?",
             "options": ["catch", "except", "error", "handle"],
             "correct_index": 1,
             "explanations": ["Incorrect — that's Java/JS.", "Correct! `except` catches exceptions in Python try/except blocks.", "Incorrect.", "Incorrect."]},
        ]

    # ── DSA weekly topics ───────────────────────────────────────────
    if "arrays" in t or "strings" in t or "complexity" in t:
        return [
            {"question": "What is the time complexity of accessing an element in an array by index?",
             "options": ["O(n)", "O(log n)", "O(1)", "O(n²)"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect.", "Correct! Array index access is constant time.", "Incorrect."]},
            {"question": "What does Big-O notation describe?",
             "options": ["Best-case performance", "Average-case performance", "Upper-bound worst-case performance", "Memory usage only"],
             "correct_index": 2,
             "explanations": ["Incorrect — that's Big-Omega.", "Incorrect — that's Big-Theta.", "Correct! Big-O describes the worst-case upper bound.", "Incorrect."]},
            {"question": "Which technique uses two pointers moving toward each other to solve array problems?",
             "options": ["Sliding window", "Divide and conquer", "Two-pointer technique", "Backtracking"],
             "correct_index": 2,
             "explanations": ["Incorrect — sliding window uses a range.", "Incorrect.", "Correct! Two pointers start from opposite ends and converge.", "Incorrect."]},
        ]

    if "linked list" in t or "stacks" in t or "queues" in t:
        return [
            {"question": "Which data structure follows Last-In First-Out (LIFO)?",
             "options": ["Queue", "Linked List", "Stack", "Tree"],
             "correct_index": 2,
             "explanations": ["Incorrect — Queue is FIFO.", "Incorrect.", "Correct! Stacks are LIFO structures.", "Incorrect."]},
            {"question": "In a singly linked list, each node contains:",
             "options": ["Only data", "Data and two pointers", "Data and one pointer to the next node", "Only a pointer"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect — that's doubly linked.", "Correct! Each node has data + next pointer.", "Incorrect."]},
            {"question": "Which operation removes the front element of a Queue?",
             "options": ["pop", "push", "dequeue", "peek"],
             "correct_index": 2,
             "explanations": ["Incorrect — pop removes from a stack.", "Incorrect — push adds.", "Correct! Dequeue removes from the front.", "Incorrect — peek only views."]},
        ]

    if "trees" in t or "heaps" in t or "graphs" in t:
        return [
            {"question": "In a Binary Search Tree, where are smaller values stored?",
             "options": ["Right subtree", "Left subtree", "Root only", "Randomly"],
             "correct_index": 1,
             "explanations": ["Incorrect — right holds larger values.", "Correct! BST stores smaller values in the left subtree.", "Incorrect.", "Incorrect."]},
            {"question": "Which graph traversal algorithm uses a queue?",
             "options": ["DFS", "BFS", "Dijkstra only", "Topological Sort"],
             "correct_index": 1,
             "explanations": ["Incorrect — DFS uses a stack.", "Correct! BFS uses a queue to visit level by level.", "Incorrect.", "Incorrect."]},
            {"question": "What property defines a Min-Heap?",
             "options": ["Parent is always greater than children", "Parent is always smaller than children", "All leaves are at the same level", "Left subtree is always larger"],
             "correct_index": 1,
             "explanations": ["Incorrect — that's a Max-Heap.", "Correct! In a Min-Heap the parent is smaller than its children.", "Incorrect.", "Incorrect."]},
        ]

    if "sorting" in t or "searching" in t or "dynamic programming" in t:
        return [
            {"question": "What is the worst-case time complexity of Merge Sort?",
             "options": ["O(n²)", "O(n log n)", "O(log n)", "O(n)"],
             "correct_index": 1,
             "explanations": ["Incorrect — that's Bubble Sort.", "Correct! Merge Sort is always O(n log n).", "Incorrect.", "Incorrect."]},
            {"question": "Dynamic Programming solves problems by:",
             "options": ["Trying all possibilities randomly", "Breaking into subproblems and storing results to avoid recomputation", "Using greedy selection at each step", "Recursion without memory"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! DP uses memoisation or tabulation to reuse subproblem solutions.", "Incorrect — that's greedy.", "Incorrect."]},
            {"question": "Binary Search requires the array to be:",
             "options": ["Unsorted", "Sorted", "Containing only integers", "Of even length"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Binary search only works on sorted arrays.", "Incorrect.", "Incorrect."]},
        ]

    # ── Healthcare / Health Informatics weekly topics ───────────────
    if "health informatics" in t or "ehr" in t or "fhir" in t or "health data" in t:
        return [
            {"question": "What does EHR stand for in healthcare IT?",
             "options": ["Electronic Health Record", "Encoded Health Report", "External Health Registry", "Encrypted Hospital Record"],
             "correct_index": 0,
             "explanations": ["Correct! EHR = Electronic Health Record.", "Incorrect.", "Incorrect.", "Incorrect."]},
            {"question": "What is FHIR primarily used for?",
             "options": ["Encrypting patient files", "Standardising healthcare data exchange between systems", "Training AI models", "Hospital billing only"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! FHIR standardises electronic health information exchange.", "Incorrect.", "Incorrect."]},
            {"question": "Which regulation governs patient data privacy in the United States?",
             "options": ["GDPR", "HIPAA", "PCI-DSS", "SOX"],
             "correct_index": 1,
             "explanations": ["Incorrect - GDPR is EU.", "Correct! HIPAA protects patient health information.", "Incorrect.", "Incorrect."]},
            {"question": "What does HL7 stand for in healthcare IT?",
             "options": ["High Level 7", "Health Level Seven", "Hospital Link 7", "Health Layer 7"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! HL7 = Health Level Seven, the standards body for healthcare data interoperability.", "Incorrect.", "Incorrect."]},
            {"question": "What is the primary purpose of a Health Information Exchange (HIE)?",
             "options": ["Billing patients faster", "Securely sharing patient data across different healthcare organisations", "Training medical staff", "Storing imaging files only"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! HIEs enable authorised sharing of patient records across organisations to improve care coordination.", "Incorrect.", "Incorrect."]},
        ]

    if "medical data" in t or "clinical dataset" in t or "patient" in t:
        return [
            {"question": "What is the main challenge with clinical datasets?",
             "options": ["Too much labeled data", "Missing values, imbalanced classes, and privacy constraints", "Always perfectly structured", "Only available in real-time"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Clinical data has missing values, class imbalance, and strict privacy rules.", "Incorrect.", "Incorrect."]},
            {"question": "Which metric is most important when evaluating a disease detection model?",
             "options": ["Accuracy", "Sensitivity (Recall)", "Throughput", "Code coverage"],
             "correct_index": 1,
             "explanations": ["Incorrect - accuracy is misleading on imbalanced data.", "Correct! Sensitivity measures true positive rate - critical to avoid missing disease cases.", "Incorrect.", "Incorrect."]},
            {"question": "What is data de-identification in healthcare?",
             "options": ["Deleting all patient records", "Removing PII so data can be used for research without exposing identity", "Encrypting data for transfer", "Converting data to JSON"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! De-identification removes name, DOB, SSN so patients cannot be identified.", "Incorrect.", "Incorrect."]},
            {"question": "What is class imbalance in clinical ML datasets?",
             "options": ["Equal numbers of sick and healthy patients", "When disease-positive cases are far fewer than negative cases", "A data formatting error", "When features have different scales"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Rare diseases create severe imbalance - e.g. 1% positive cases vs 99% negative.", "Incorrect.", "Incorrect."]},
            {"question": "Which Python library is most commonly used for clinical data analysis?",
             "options": ["TensorFlow", "Pandas", "Flask", "Django"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Pandas provides DataFrames ideal for loading, cleaning, and analysing clinical tabular data.", "Incorrect.", "Incorrect."]},
        ]

    if "medical imaging" in t or "x-ray" in t or "mri" in t or ("cnn" in t and "health" in t) or ("computer vision" in t and "health" in t):
        return [
            {"question": "Which deep learning architecture is most commonly used for medical image classification?",
             "options": ["RNN", "CNN (Convolutional Neural Network)", "Decision Tree", "Linear Regression"],
             "correct_index": 1,
             "explanations": ["Incorrect - RNNs are for sequences.", "Correct! CNNs extract spatial features from images using convolutional filters.", "Incorrect.", "Incorrect."]},
            {"question": "What is transfer learning in medical imaging?",
             "options": ["Training from scratch on X-rays", "Using a pre-trained model (e.g. ResNet) fine-tuned on medical data", "Copying weights between hospitals", "Scanning images twice"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Transfer learning reuses ImageNet-trained weights and fine-tunes on smaller medical datasets.", "Incorrect.", "Incorrect."]},
            {"question": "What does Grad-CAM provide in medical AI?",
             "options": ["Faster training", "Visual heatmap showing which image regions influenced the prediction", "Data augmentation", "Batch normalisation"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Grad-CAM highlights regions driving the model decision - key for clinical trust.", "Incorrect.", "Incorrect."]},
            {"question": "Why is data augmentation important in medical imaging datasets?",
             "options": ["To increase image resolution", "To artificially expand small datasets by applying rotations, flips, and contrast changes", "To remove noise from scans", "To convert DICOM to PNG"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Medical datasets are small; augmentation prevents overfitting by generating varied training samples.", "Incorrect.", "Incorrect."]},
            {"question": "What is a DICOM file in medical imaging?",
             "options": ["A Python imaging library", "The standard format for storing and transmitting medical images and metadata", "A type of CNN layer", "A hospital database format"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! DICOM (Digital Imaging and Communications in Medicine) is the universal standard for medical images.", "Incorrect.", "Incorrect."]},
        ]

    if "clinical nlp" in t or "biobert" in t or "clinical text" in t or "discharge" in t or ("nlp" in t and "clinical" in t) or ("natural language" in t and "clinical" in t):
        return [
            {"question": "What is BioBERT?",
             "options": ["A hospital billing system", "A BERT model pre-trained on biomedical literature for clinical NLP", "A medical imaging tool", "A Python library for EHR"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! BioBERT is trained on PubMed and PMC texts for biomedical NLP tasks.", "Incorrect.", "Incorrect."]},
            {"question": "What is Named Entity Recognition (NER) used for in clinical text?",
             "options": ["Encrypting patient notes", "Identifying medical terms like diagnoses, drugs, and symptoms in text", "Generating reports", "Scheduling appointments"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! NER extracts structured medical entities from unstructured clinical notes.", "Incorrect.", "Incorrect."]},
            {"question": "Why is NLP on clinical notes harder than general text?",
             "options": ["Notes are too short", "Medical abbreviations, typos, domain jargon, and privacy constraints", "Doctors write perfectly structured notes", "Too much data available"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Clinical text has non-standard abbreviations, misspellings, and highly specialised terminology.", "Incorrect.", "Incorrect."]},
            {"question": "What does ICD-10 coding refer to in clinical NLP?",
             "options": ["A Python version", "International Classification of Diseases codes used to classify diagnoses and procedures", "A network protocol", "An imaging format"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! ICD-10 provides standardised codes that NLP models extract from clinical notes for billing and analytics.", "Incorrect.", "Incorrect."]},
            {"question": "What is relation extraction in clinical NLP?",
             "options": ["Removing duplicate records", "Identifying relationships between medical entities such as drug-dosage or disease-symptom pairs", "Extracting images from reports", "Converting text to speech"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Relation extraction identifies semantic links between entities like 'metformin treats diabetes'.", "Incorrect.", "Incorrect."]},
        ]

    if "decision support" in t or "cdss" in t or "clinical decision" in t or ("clinical" in t and "support" in t):
        return [
            {"question": "What is a Clinical Decision Support System (CDSS)?",
             "options": ["A scheduling app", "An AI system providing evidence-based recommendations to clinicians at point of care", "A patient portal", "A billing tool"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! CDSS analyses patient data and knowledge bases to assist clinical decisions.", "Incorrect.", "Incorrect."]},
            {"question": "Which standard integrates CDSS with EHR systems?",
             "options": ["REST only", "HL7 FHIR CDS Hooks", "SOAP XML", "CSV export"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! CDS Hooks is the modern standard for triggering CDSS services within EHR workflows.", "Incorrect.", "Incorrect."]},
            {"question": "What is alert fatigue in CDSS?",
             "options": ["System slowdown", "Clinicians ignoring alerts because too many are irrelevant or low-priority", "Database timeout", "Model overfitting"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Alert fatigue leads clinicians to override alerts without reading them, undermining patient safety.", "Incorrect.", "Incorrect."]},
            {"question": "What type of knowledge base does a rule-based CDSS rely on?",
             "options": ["Neural network weights", "Curated clinical guidelines and IF-THEN logic rules", "Real-time social media feeds", "Random patient surveys"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Rule-based CDSS encodes clinical guidelines as explicit IF-THEN rules (e.g. if HbA1c > 7%, suggest diabetes review).", "Incorrect.", "Incorrect."]},
            {"question": "What is a key advantage of ML-based CDSS over rule-based systems?",
             "options": ["Always interpretable", "Can learn complex patterns from large patient datasets without manually coded rules", "Requires no training data", "Cheaper to build"],
             "correct_index": 1,
             "explanations": ["Incorrect - ML models are often black boxes.", "Correct! ML-based CDSS discovers patterns in data automatically, handling complexity beyond hand-coded rules.", "Incorrect.", "Incorrect."]},
        ]

    if "healthcare data privacy" in t or ("ethics" in t and "health" in t) or "hipaa" in t or ("privacy" in t and "health" in t):
        return [
            {"question": "What does HIPAA primarily protect?",
             "options": ["Hospital financial records", "Patients protected health information (PHI)", "Doctor salaries", "Medical device patents"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! HIPAA protects individually identifiable health information held by covered entities.", "Incorrect.", "Incorrect."]},
            {"question": "What is data de-identification under HIPAA?",
             "options": ["Deleting all patient records", "Removing 18 specific identifiers so data cannot be linked to an individual", "Encrypting data for transfer", "Anonymising only names"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! HIPAA Safe Harbor requires removal of 18 identifiers including name, DOB, zip code, and SSN.", "Incorrect.", "Incorrect."]},
            {"question": "What is algorithmic bias in healthcare AI?",
             "options": ["Intentional manipulation", "Systematic errors due to unrepresentative training data across demographic groups", "Random noise", "Slow inference"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Models trained on non-diverse data perform worse for underrepresented groups, causing inequitable outcomes.", "Incorrect.", "Incorrect."]},
            {"question": "What is federated learning and why is it important for healthcare privacy?",
             "options": ["Training on centralised hospital data", "Training models locally at each hospital without sharing raw patient data", "Sharing patient records globally", "A type of data encryption"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Federated learning keeps patient data on-premise - only model updates are shared, preserving privacy.", "Incorrect.", "Incorrect."]},
            {"question": "What is the principle of data minimisation in healthcare AI ethics?",
             "options": ["Using the largest possible dataset", "Collecting and processing only the minimum patient data necessary for the task", "Deleting all data after use", "Sharing data freely for research"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Data minimisation reduces privacy risk by limiting collection to what is strictly needed.", "Incorrect.", "Incorrect."]},
        ]

    if ("fda" in t and "health" in t) or "healthcare ai deployment" in t or ("regulation" in t and "health" in t) or "medical device" in t:
        return [
            {"question": "Under which FDA category do most AI/ML diagnostic tools fall?",
             "options": ["Class I", "Class II/III - Software as a Medical Device (SaMD)", "Over-the-counter", "Food additives"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! AI diagnostic tools are regulated as SaMD, typically Class II or III depending on risk.", "Incorrect.", "Incorrect."]},
            {"question": "What does the FDA AI/ML action plan require for adaptive models?",
             "options": ["All models must be open-source", "Ongoing monitoring, performance validation, and transparency", "Paper submissions only", "No post-market surveillance"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! The FDA requires pre-market transparency and post-market performance monitoring for AI/ML SaMD.", "Incorrect.", "Incorrect."]},
            {"question": "What is distribution shift in deployed healthcare AI?",
             "options": ["Model is always accurate", "Performance degrades when real-world patient population differs from training data", "Too much compute available", "No regulatory oversight"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Distribution shift causes model performance to degrade when deployment conditions differ from training.", "Incorrect.", "Incorrect."]},
            {"question": "What is model drift monitoring in healthcare AI?",
             "options": ["Checking server uptime", "Continuously tracking model performance post-deployment to detect degradation", "Retraining every month automatically", "Archiving old model versions"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Drift monitoring detects when model accuracy drops due to changes in patient population or clinical practices.", "Incorrect.", "Incorrect."]},
            {"question": "What does CE marking indicate for AI medical devices in Europe?",
             "options": ["A quality award", "Compliance with EU Medical Device Regulation (MDR) requirements", "A software certification", "An ISO standard"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! CE marking indicates the device meets EU MDR safety and performance requirements for market approval.", "Incorrect.", "Incorrect."]},
        ]

    # ── ML / AI weekly topics ───────────────────────────────────────
    if "ml fundamentals" in t or "data preprocessing" in t or "preprocessing" in t:
        return [
            {"question": "What is the purpose of train/test split in machine learning?",
             "options": ["To reduce dataset size", "To evaluate model performance on unseen data", "To increase training accuracy", "To remove outliers"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Test data is held out to evaluate generalisation.", "Incorrect.", "Incorrect."]},
            {"question": "What does data normalisation do?",
             "options": ["Removes duplicate rows", "Scales features to a common range", "Fills missing values", "Encodes categorical data"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Normalisation rescales values to a standard range like [0,1].", "Incorrect — that's imputation.", "Incorrect."]},
            {"question": "Which library is commonly used for exploratory data analysis in Python?",
             "options": ["TensorFlow", "Pandas", "Scikit-learn", "PyTorch"],
             "correct_index": 1,
             "explanations": ["Incorrect — TensorFlow is for deep learning.", "Correct! Pandas is the primary EDA library.", "Incorrect — Scikit-learn is for ML models.", "Incorrect."]},
        ]

    if "regression" in t or "classification" in t:
        return [
            {"question": "Which algorithm is best suited for binary classification?",
             "options": ["Linear Regression", "Logistic Regression", "K-Means", "PCA"],
             "correct_index": 1,
             "explanations": ["Incorrect — Linear Regression predicts continuous values.", "Correct! Logistic Regression outputs probabilities for classification.", "Incorrect — K-Means is clustering.", "Incorrect — PCA is dimensionality reduction."]},
            {"question": "What does precision measure in a classification model?",
             "options": ["How many actual positives were found", "Of all predicted positives, how many were actually positive", "Overall accuracy across all classes", "Training loss"],
             "correct_index": 1,
             "explanations": ["Incorrect — that's recall.", "Correct! Precision = TP / (TP + FP).", "Incorrect.", "Incorrect."]},
            {"question": "Random Forest is an ensemble of:",
             "options": ["Neural networks", "Decision trees", "SVMs", "K-Nearest Neighbours"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Random Forest aggregates many decision trees.", "Incorrect.", "Incorrect."]},
        ]

    if "unsupervised" in t or "clustering" in t or "pca" in t or "nlp" in t:
        return [
            {"question": "What does K-Means clustering minimise?",
             "options": ["Number of clusters", "Within-cluster sum of squared distances", "Cross-entropy loss", "Precision score"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! K-Means minimises intra-cluster variance.", "Incorrect.", "Incorrect."]},
            {"question": "PCA is primarily used for:",
             "options": ["Classification", "Dimensionality reduction", "Time series forecasting", "Text generation"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! PCA reduces dimensions while retaining variance.", "Incorrect.", "Incorrect."]},
            {"question": "Which NLP technique converts words to dense vector representations?",
             "options": ["One-hot encoding", "TF-IDF", "Word embeddings (Word2Vec)", "Label encoding"],
             "correct_index": 2,
             "explanations": ["Incorrect — one-hot is sparse.", "Incorrect — TF-IDF is frequency-based.", "Correct! Word2Vec produces dense semantic vector embeddings.", "Incorrect."]},
        ]

    if "neural network" in t or "deep learning" in t or "deployment" in t or "cnn" in t:
        return [
            {"question": "What is the role of an activation function in a neural network?",
             "options": ["To initialise weights", "To introduce non-linearity into the model", "To normalise input data", "To split training data"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Activation functions allow networks to learn complex non-linear patterns.", "Incorrect.", "Incorrect."]},
            {"question": "Which algorithm is used to train neural networks by adjusting weights?",
             "options": ["Forward propagation", "Backpropagation", "Gradient boosting", "Random search"],
             "correct_index": 1,
             "explanations": ["Incorrect — forward pass computes output.", "Correct! Backpropagation calculates gradients and updates weights.", "Incorrect.", "Incorrect."]},
            {"question": "CNNs are primarily designed for:",
             "options": ["Text classification", "Time series analysis", "Image recognition", "Tabular data"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect.", "Correct! CNNs use convolutional filters ideal for image data.", "Incorrect."]},
        ]

    # ── Web Development weekly topics ──────────────────────────────
    if "html" in t or "css" in t:
        return [
            {"question": "Which HTML tag is used for the largest heading?",
             "options": ["<h6>", "<heading>", "<h1>", "<title>"],
             "correct_index": 2,
             "explanations": ["Incorrect — h6 is the smallest.", "Incorrect.", "Correct! <h1> is the largest heading tag.", "Incorrect."]},
            {"question": "What does CSS flexbox `justify-content: center` do?",
             "options": ["Centers items vertically", "Centers items horizontally along the main axis", "Adds padding", "Sets font size"],
             "correct_index": 1,
             "explanations": ["Incorrect — use align-items for vertical.", "Correct! justify-content controls main-axis alignment.", "Incorrect.", "Incorrect."]},
            {"question": "Which CSS unit is relative to the viewport width?",
             "options": ["px", "em", "vw", "rem"],
             "correct_index": 2,
             "explanations": ["Incorrect — px is absolute.", "Incorrect — em is relative to parent font.", "Correct! vw = 1% of viewport width.", "Incorrect — rem is relative to root font size."]},
        ]

    if "javascript" in t or "dom" in t:
        return [
            {"question": "Which method selects an element by its ID in JavaScript?",
             "options": ["document.querySelector", "document.getElementById", "document.getElement", "document.findById"],
             "correct_index": 1,
             "explanations": ["Incorrect — querySelector uses CSS selectors.", "Correct! getElementById returns the element with the given ID.", "Incorrect.", "Incorrect."]},
            {"question": "What does `async/await` do in JavaScript?",
             "options": ["Runs code in parallel threads", "Handles asynchronous operations in a synchronous style", "Blocks the main thread", "Creates new variables"],
             "correct_index": 1,
             "explanations": ["Incorrect — JS is single-threaded.", "Correct! async/await makes promise-based code easier to read and write.", "Incorrect.", "Incorrect."]},
            {"question": "Which Fetch API method sends data to a server?",
             "options": ["GET", "POST", "FETCH", "SEND"],
             "correct_index": 1,
             "explanations": ["Incorrect — GET retrieves data.", "Correct! POST sends data to the server.", "Incorrect.", "Incorrect."]},
        ]

    if "backend" in t or "flask" in t or "node" in t or "rest api" in t:
        return [
            {"question": "What does REST stand for?",
             "options": ["Rapid Execution State Transfer", "Representational State Transfer", "Remote Event Service Technology", "Resource Execution Standard Transfer"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! REST = Representational State Transfer.", "Incorrect.", "Incorrect."]},
            {"question": "Which HTTP status code indicates a resource was successfully created?",
             "options": ["200", "201", "404", "500"],
             "correct_index": 1,
             "explanations": ["Incorrect — 200 is OK.", "Correct! 201 Created is the standard response for POST.", "Incorrect — 404 is Not Found.", "Incorrect — 500 is Server Error."]},
            {"question": "In Flask, which decorator registers a route?",
             "options": ["@flask.route", "@app.route", "@route", "@endpoint"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! @app.route('/path') registers URL routes in Flask.", "Incorrect.", "Incorrect."]},
        ]

    # ── Database weekly topics ──────────────────────────────────────
    if "sql basics" in t or "relational" in t:
        return [
            {"question": "Which SQL command retrieves data from a table?",
             "options": ["INSERT", "UPDATE", "SELECT", "DELETE"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect.", "Correct! SELECT fetches data from a table.", "Incorrect."]},
            {"question": "What constraint ensures a column has no duplicate values?",
             "options": ["NOT NULL", "PRIMARY KEY", "UNIQUE", "CHECK"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect — PRIMARY KEY also prevents nulls.", "Correct! UNIQUE enforces distinct values.", "Incorrect."]},
            {"question": "Which SQL clause filters rows after grouping?",
             "options": ["WHERE", "HAVING", "ORDER BY", "LIMIT"],
             "correct_index": 1,
             "explanations": ["Incorrect — WHERE filters before grouping.", "Correct! HAVING filters groups after GROUP BY.", "Incorrect.", "Incorrect."]},
        ]

    if "advanced sql" in t or "joins" in t:
        return [
            {"question": "Which JOIN returns all rows from both tables, filling NULLs where no match?",
             "options": ["INNER JOIN", "LEFT JOIN", "FULL OUTER JOIN", "CROSS JOIN"],
             "correct_index": 2,
             "explanations": ["Incorrect — INNER only returns matched rows.", "Incorrect — LEFT returns all left rows.", "Correct! FULL OUTER JOIN returns all rows from both tables.", "Incorrect."]},
            {"question": "What is a CTE in SQL?",
             "options": ["A type of index", "A Common Table Expression — a temporary named result set", "A stored procedure", "A trigger function"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! CTEs use WITH keyword to define a temporary result set.", "Incorrect.", "Incorrect."]},
            {"question": "What does a window function compute?",
             "options": ["Aggregates across the entire table", "Values across a defined partition without collapsing rows", "Only the first row of each group", "Cross-table joins"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Window functions operate over a partition while keeping individual rows.", "Incorrect.", "Incorrect."]},
        ]

    # ── Cloud / DevOps weekly topics ────────────────────────────────
    if "cloud computing fundamentals" in t or "iaas" in t or "paas" in t:
        return [
            {"question": "What does IaaS provide?",
             "options": ["Managed applications", "Virtual machines and networking infrastructure", "Development platforms only", "Only storage services"],
             "correct_index": 1,
             "explanations": ["Incorrect — that's SaaS.", "Correct! IaaS provides virtualised compute, storage, and networking.", "Incorrect — that's PaaS.", "Incorrect."]},
            {"question": "What is an AWS Availability Zone?",
             "options": ["A geographic region", "An isolated data centre within a region", "A CDN edge location", "A managed Kubernetes cluster"],
             "correct_index": 1,
             "explanations": ["Incorrect — that's a Region.", "Correct! AZs are isolated data centres inside a region.", "Incorrect.", "Incorrect."]},
            {"question": "What does IAM stand for in AWS?",
             "options": ["Internet Access Management", "Identity and Access Management", "Instance Application Monitor", "Infrastructure Automation Module"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! IAM controls who can access AWS resources.", "Incorrect.", "Incorrect."]},
        ]

    if "docker" in t or "containeris" in t:
        return [
            {"question": "What is a Docker image?",
             "options": ["A running container instance", "A read-only template used to create containers", "A Docker network configuration", "A registry account"],
             "correct_index": 1,
             "explanations": ["Incorrect — that's a container.", "Correct! Images are immutable blueprints for containers.", "Incorrect.", "Incorrect."]},
            {"question": "Which file defines a Docker image build process?",
             "options": ["docker-compose.yml", "Dockerfile", "container.json", "image.config"],
             "correct_index": 1,
             "explanations": ["Incorrect — docker-compose orchestrates multi-container apps.", "Correct! Dockerfile contains build instructions.", "Incorrect.", "Incorrect."]},
            {"question": "What does `docker-compose up` do?",
             "options": ["Builds a single container", "Starts all services defined in docker-compose.yml", "Pushes images to a registry", "Removes all containers"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! docker-compose up starts all defined services.", "Incorrect.", "Incorrect."]},
        ]

    if "ci/cd" in t or "pipeline" in t or "github actions" in t:
        return [
            {"question": "What does CI stand for in CI/CD?",
             "options": ["Container Integration", "Continuous Integration", "Code Inspection", "Cloud Infrastructure"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! CI = Continuous Integration — automatically building and testing on every commit.", "Incorrect.", "Incorrect."]},
            {"question": "Which file configures a GitHub Actions workflow?",
             "options": [".github/workflow.json", ".github/workflows/*.yml", "Jenkinsfile", "pipeline.config"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! GitHub Actions uses YAML files in .github/workflows/.", "Incorrect — Jenkinsfile is for Jenkins.", "Incorrect."]},
            {"question": "What is the main benefit of automated deployment pipelines?",
             "options": ["Slower releases", "Consistent, repeatable deployments with fewer human errors", "Manual testing only", "Increased infrastructure cost"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Automation reduces errors and speeds up release cycles.", "Incorrect.", "Incorrect."]},
        ]

    # ── Cybersecurity weekly topics ─────────────────────────────────
    if "threat" in t or "owasp" in t or "security fundamentals" in t:
        return [
            {"question": "What does the CIA triad stand for?",
             "options": ["Code, Interface, Application", "Confidentiality, Integrity, Availability", "Cloud, Infrastructure, Access", "Cipher, Identity, Authentication"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! CIA = Confidentiality, Integrity, Availability.", "Incorrect.", "Incorrect."]},
            {"question": "What is SQL injection?",
             "options": ["Encrypting SQL queries", "Inserting malicious SQL code into input fields to manipulate a database", "Backing up a SQL database", "Optimising SQL queries"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! SQL injection exploits unsanitised input to execute arbitrary queries.", "Incorrect.", "Incorrect."]},
            {"question": "What is phishing?",
             "options": ["A network scanning technique", "A social engineering attack using deceptive messages to steal credentials", "A type of firewall", "A password hashing algorithm"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Phishing tricks users into revealing sensitive information.", "Incorrect.", "Incorrect."]},
        ]

    if "cryptography" in t or "encryption" in t or "hashing" in t:
        return [
            {"question": "Which encryption type uses the same key for both encryption and decryption?",
             "options": ["Asymmetric", "Symmetric", "Public-key", "Hashing"],
             "correct_index": 1,
             "explanations": ["Incorrect — asymmetric uses two keys.", "Correct! Symmetric encryption uses a single shared secret key.", "Incorrect — another name for asymmetric.", "Incorrect — hashing is one-way."]},
            {"question": "What makes SHA-256 a hashing algorithm rather than encryption?",
             "options": ["It uses a key", "It is irreversible — you cannot recover the original input", "It produces variable-length output", "It requires two parties"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Hashes are one-way; encrypted data can be decrypted.", "Incorrect — SHA-256 always outputs 256 bits.", "Incorrect."]},
            {"question": "What does TLS protect in web communications?",
             "options": ["Server uptime", "Data in transit between client and server", "Database records", "Source code"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! TLS encrypts data transmitted over the network.", "Incorrect.", "Incorrect."]},
        ]

    # -- Disaster Management --
    if any(x in t for x in ["disaster", "emergency", "hazard", "relief", "rescue", "humanitarian", "resilience", "incident command", "vulnerability assessment", "drr", "recovery", "reconstruction", "preparedness"]):
        return [
            {"question": "What are the four phases of the disaster management cycle?",
             "options": ["Plan, Act, Check, Improve", "Mitigation, Preparedness, Response, Recovery", "Prevention, Training, Deployment, Rebuild", "Alert, Mobilise, Respond, Debrief"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! The four phases are Mitigation, Preparedness, Response, and Recovery.", "Incorrect.", "Incorrect."]},
            {"question": "What does ICS stand for in emergency management?",
             "options": ["International Crisis System", "Incident Command System", "Integrated Coordination Service", "Immediate Crisis Support"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! ICS (Incident Command System) is the standardised approach to command and coordination of emergency response.", "Incorrect.", "Incorrect."]},
            {"question": "Which UN framework guides global disaster risk reduction (2015-2030)?",
             "options": ["Kyoto Protocol", "Paris Agreement", "Sendai Framework", "Hyogo Framework"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect.", "Correct! The Sendai Framework for Disaster Risk Reduction 2015-2030 is the global blueprint for DRR.", "Incorrect - Hyogo was the previous framework."]},
            {"question": "What does WASH stand for in humanitarian relief?",
             "options": ["Water, Agriculture, Shelter, Health", "Water, Sanitation and Hygiene", "Welfare, Aid, Safety, Habitation", "Warning, Assessment, Support, Help"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! WASH covers Water, Sanitation and Hygiene - critical components of humanitarian response.", "Incorrect.", "Incorrect."]},
            {"question": "What is a Post-Disaster Needs Assessment (PDNA)?",
             "options": ["A damage insurance form", "A systematic evaluation of disaster impacts on livelihoods, infrastructure, and services to guide recovery planning", "A real-time early warning alert", "A volunteer recruitment process"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! A PDNA quantifies human, physical, and economic losses to inform recovery strategies.", "Incorrect.", "Incorrect."]},
            {"question": "What is the 'Build Back Better' principle in disaster recovery?",
             "options": ["Rebuilding faster than before", "Restoring and improving pre-disaster conditions to reduce future vulnerability", "Using cheaper construction materials", "Prioritising urban areas first"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Build Back Better means using recovery as an opportunity to increase resilience, not just restore what existed.", "Incorrect.", "Incorrect."]},
            {"question": "Which agency leads UN humanitarian coordination in emergencies?",
             "options": ["UNICEF", "UNHCR", "OCHA", "WFP"],
             "correct_index": 2,
             "explanations": ["Incorrect - UNICEF focuses on children.", "Incorrect - UNHCR focuses on refugees.", "Correct! OCHA (Office for the Coordination of Humanitarian Affairs) coordinates the global emergency response.", "Incorrect - WFP focuses on food security."]},
            {"question": "What is vulnerability in the context of disaster risk?",
             "options": ["The magnitude of a natural hazard", "The conditions that make people susceptible to harm from a hazard", "The speed of emergency response", "The number of people affected"],
             "correct_index": 1,
             "explanations": ["Incorrect - that describes hazard intensity.", "Correct! Vulnerability refers to social, economic, and physical conditions that increase a community's susceptibility to disaster impacts.", "Incorrect.", "Incorrect."]},
            {"question": "What is the purpose of an Early Warning System (EWS) in disaster management?",
             "options": ["To document past disasters", "To detect hazard onset and alert at-risk communities with enough lead time to take protective action", "To train emergency responders", "To distribute aid supplies"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! EWS integrates monitoring, forecasting, communication, and community response capacity.", "Incorrect.", "Incorrect."]},
            {"question": "What does climate change primarily increase in the context of disasters?",
             "options": ["Response team sizes", "Frequency, intensity, and unpredictability of hydro-meteorological hazards", "International aid budgets", "Building construction costs"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Climate change amplifies hazards like floods, droughts, and cyclones, increasing disaster risk.", "Incorrect.", "Incorrect."]},
        ]

    # -- Generic healthcare catch-all --
    if "healthcare" in t or ("health" in t and "ai" in t) or ("health" in t and "machine learning" in t):
        return [
            {"question": "What does AI in healthcare primarily aim to improve?",
             "options": ["Hospital architecture", "Clinical decision-making accuracy and patient outcomes", "Administrative salaries", "Insurance pricing only"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! AI improves diagnostic accuracy, treatment recommendations, and patient outcome predictions.", "Incorrect.", "Incorrect."]},
            {"question": "Which ML model type is used for disease risk prediction?",
             "options": ["Unsupervised clustering", "Supervised classification (e.g. logistic regression, random forest)", "Reinforcement learning only", "GANs"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Supervised classification learns from labelled patient data to predict disease risk.", "Incorrect.", "Incorrect."]},
            {"question": "Why is model explainability important in healthcare AI?",
             "options": ["It speeds up training", "Clinicians must understand AI recommendations before acting on them", "It reduces storage costs", "Regulations do not require it"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Explainability (SHAP, LIME, Grad-CAM) allows clinicians to verify AI reasoning and maintain accountability.", "Incorrect.", "Incorrect."]},
        ]

    # -- Fallback: original question bank --
    c_lower = t
    if "python" in c_lower or "programming" in c_lower:
        return [
            {"question": "Which of the following data types in Python is immutable?",
             "options": ["List", "Dictionary", "Tuple", "Set"],
             "correct_index": 2,
             "explanations": ["Incorrect. Lists are mutable.", "Incorrect. Dictionaries are mutable.", "Correct! Tuples are immutable.", "Incorrect. Sets are mutable."]},
            {"question": "What does the 'self' keyword represent in a Python class?",
             "options": ["The class object", "The current instance", "A global variable", "A parent class initializer"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! self references the current instance.", "Incorrect.", "Incorrect."]},
            {"question": "What is the primary use of list comprehension?",
             "options": ["Compress lists", "Document lists", "Generate a new list in one line", "Sort lists"],
             "correct_index": 2,
             "explanations": ["Incorrect.", "Incorrect.", "Correct! List comprehensions create lists concisely.", "Incorrect."]},
        ]
    if "data structures" in c_lower or "algorithms" in c_lower:
        return [
            {"question": "Average time complexity of Hash Table search?",
             "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
             "correct_index": 0,
             "explanations": ["Correct! Hash tables achieve O(1) on average.", "Incorrect.", "Incorrect.", "Incorrect."]},
            {"question": "Which structure follows FIFO?",
             "options": ["Stack", "Queue", "Tree", "Graph"],
             "correct_index": 1,
             "explanations": ["Incorrect — Stack is LIFO.", "Correct! Queues are FIFO.", "Incorrect.", "Incorrect."]},
            {"question": "Merge Sort worst-case complexity?",
             "options": ["O(n²)", "O(n log n)", "O(log n)", "O(1)"],
             "correct_index": 1,
             "explanations": ["Incorrect.", "Correct! Merge Sort is always O(n log n).", "Incorrect.", "Incorrect."]},
        ]
    # Dynamic generic fallback for any unknown topic
    topic_display = course_name.title()
    return [
        {"question": f"What is the primary objective of studying {topic_display}?",
         "options": ["To memorise facts", "To develop domain expertise and apply knowledge to real-world problems", "To pass exams only", "To learn unrelated skills"],
         "correct_index": 1,
         "explanations": ["Incorrect.", f"Correct! {topic_display} equips students with specialised knowledge and practical skills applicable to real-world challenges.", "Incorrect.", "Incorrect."]},
        {"question": f"Which approach best describes effective learning in {topic_display}?",
         "options": ["Reading textbooks only", "Combining theory with practical application and critical analysis", "Memorising definitions", "Watching lectures passively"],
         "correct_index": 1,
         "explanations": ["Incorrect.", "Correct! Deep learning integrates theoretical foundations with hands-on practice and reflective thinking.", "Incorrect.", "Incorrect."]},
        {"question": f"What role does research play in {topic_display}?",
         "options": ["No role", "It generates new knowledge and validates existing theories through evidence", "Only used by academics", "Slows down practical work"],
         "correct_index": 1,
         "explanations": ["Incorrect.", "Correct! Research advances the field by testing hypotheses, uncovering patterns, and building an evidence base.", "Incorrect.", "Incorrect."]},
        {"question": f"What is a key career pathway for graduates in {topic_display}?",
         "options": ["Only academia", "Diverse roles in industry, government, NGOs, and research organisations", "Only private sector", "No career opportunities"],
         "correct_index": 1,
         "explanations": ["Incorrect.", f"Correct! {topic_display} graduates work across sectors including industry, policy, research, and consulting.", "Incorrect.", "Incorrect."]},
        {"question": f"How does ethics apply to the practice of {topic_display}?",
         "options": ["It doesn't apply", "Ethical principles guide responsible conduct, data integrity, and stakeholder accountability", "Only required for legal compliance", "Covered only in philosophy courses"],
         "correct_index": 1,
         "explanations": ["Incorrect.", "Correct! Ethics ensures practitioners act responsibly, maintain integrity, and consider societal impacts.", "Incorrect.", "Incorrect."]},
    ]

def generate_semester_quiz(course_names_list):
    """
    Generates exactly 10 unique questions based on the semester's course topics.
    Builds a pool from all courses (5 q each = 15+ unique), shuffles, returns 10.
    """
    import random
    pool = []
    seen = set()
    for name in course_names_list:
        for q in generate_course_quiz(name):
            key = q['question'][:60]
            if key not in seen:
                seen.add(key)
                pool.append(q)
    # If somehow still under 10 (very unlikely with 5 q/course), cycle again
    i = 0
    while len(pool) < 10 and i < len(course_names_list) * 10:
        extra = generate_course_quiz(course_names_list[i % len(course_names_list)])
        for q in extra:
            key = q['question'][:58] + str(len(pool))
            if key not in seen:
                seen.add(key)
                pool.append(dict(q))
            if len(pool) >= 10:
                break
        i += 1
    random.shuffle(pool)
    return pool[:10]


def generate_full_report_data(topic, curriculum_data):
    """
    Generates enriched report sections: overview, resources, skills,
    career roadmap, capstone projects, and skill levels.
    Falls back to rich mock data if Ollama unavailable.
    """
    field = curriculum_data.get("field_of_study", topic)
    courses = [c.get("name", "") for s in curriculum_data.get("semesters", []) for c in s.get("courses", [])]
    courses_str = ", ".join(courses[:10])

    prompt = (
        f'You are an academic curriculum expert. Generate a detailed JSON report for a curriculum on "{topic}" '
        f'(Field: {field}). Courses covered: {courses_str}.\n\n'
        'Output ONLY a valid JSON object. No markdown. No extra text.\n'
        '{"target_audience":"string","program_description":"3-4 sentences",'
        '"learning_objectives":["obj1","obj2","obj3","obj4","obj5"],'
        '"learning_resources":{"books":[{"title":"T","author":"A","url":"U"}],'
        '"documentation":[{"title":"T","url":"U"}],"video_courses":[{"title":"T","url":"U"}],'
        '"research_papers":[{"title":"T","author":"A"}],"industry_references":[{"title":"T","url":"U"}]},'
        '"skills_mapping":{"technical_skills":["s1"],"tools":["t1"],"frameworks":["f1"],"industry_competencies":["c1"]},'
        '"skill_levels":{"Skill":80},'
        '"career_roadmap":{"beginner_roles":[{"title":"R","description":"D","salary":"$X"}],'
        '"intermediate_roles":[{"title":"R","description":"D","salary":"$X"}],'
        '"advanced_roles":[{"title":"R","description":"D","salary":"$X"}]},'
        '"salary_insights":{"Entry Level":"$55k"},'
        '"capstone_projects":[{"title":"T","description":"D","tech_stack":["t1"],"deliverables":["d1"],"difficulty":"Advanced"}]}'
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3}},
            timeout=20
        )
        if response.status_code == 200:
            raw = response.json().get("response", "").strip()
            raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
    except Exception as e:
        print(f"Full report generation failed: {e}")
    return _mock_full_report(topic, field)


def _mock_full_report(topic, field):
    return {
        "target_audience": "Undergraduate & Professional",
        "program_description": (
            f"This comprehensive {topic} program provides students with both theoretical foundations "
            f"and industry-ready practical skills in {field}. Students progress through structured "
            "semesters covering core principles, applied development, and advanced system design. "
            "Graduates are prepared for immediate employment or advanced academic research."
        ),
        "learning_objectives": [
            f"Master foundational and advanced concepts of {topic}",
            f"Apply industry-standard tools and frameworks in {field}",
            "Design and implement scalable, production-grade systems",
            "Collaborate effectively in Agile / DevOps environments",
            "Demonstrate research and innovation capabilities",
        ],
        "learning_resources": {
            "books": [
                {"title": f"Fundamentals of {topic}", "author": "MIT OpenCourseWare", "url": "https://ocw.mit.edu"},
                {"title": f"Professional {topic} Handbook", "author": "O'Reilly Media", "url": "https://www.oreilly.com"},
                {"title": f"{field} in Practice", "author": "Manning Publications", "url": "https://www.manning.com"},
            ],
            "documentation": [
                {"title": "ACM Digital Library", "url": "https://dl.acm.org"},
                {"title": "IEEE Xplore", "url": "https://ieeexplore.ieee.org"},
            ],
            "video_courses": [
                {"title": f"{topic} — Coursera Specialisation", "url": "https://www.coursera.org"},
                {"title": f"{topic} Full Course — Udemy", "url": "https://www.udemy.com"},
                {"title": "IBM Skills Network", "url": "https://cognitiveclass.ai"},
            ],
            "research_papers": [
                {"title": f"Advances in {topic}: A Survey", "author": "ACM / IEEE 2024"},
                {"title": f"Industry Adoption of {field}", "author": "Gartner Research"},
            ],
            "industry_references": [
                {"title": "GitHub Awesome Lists", "url": f"https://github.com/topics/{topic.lower().replace(' ','-')}"},
                {"title": "Stack Overflow Developer Survey", "url": "https://survey.stackoverflow.co"},
            ],
        },
        "skills_mapping": {
            "technical_skills": [f"{topic} Core", "Algorithm Design", "Data Structures", "System Architecture", "Testing & QA"],
            "tools": ["Git", "Docker", "VS Code", "Postman", "Jupyter"],
            "frameworks": ["Flask / FastAPI", "React / Vue", "TensorFlow / PyTorch", "Kubernetes"],
            "industry_competencies": ["Agile / Scrum", "CI/CD Pipelines", "Code Review", "Technical Documentation"],
        },
        "skill_levels": {
            f"{topic} Fundamentals": 95,
            "System Design": 80,
            "Data Structures": 85,
            "Cloud & DevOps": 70,
            "Testing & QA": 75,
            "Documentation": 65,
            "Collaboration": 88,
        },
        "career_roadmap": {
            "beginner_roles": [
                {"title": f"{topic} Junior Developer", "description": "Entry-level implementation and support roles.", "salary": "$55k–$75k/yr"},
                {"title": "Technical Analyst", "description": "Data gathering, report generation.", "salary": "$50k–$70k/yr"},
            ],
            "intermediate_roles": [
                {"title": f"{topic} Engineer", "description": "Design and develop core systems.", "salary": "$80k–$110k/yr"},
                {"title": "Solutions Architect", "description": "Blueprint scalable enterprise solutions.", "salary": "$90k–$120k/yr"},
            ],
            "advanced_roles": [
                {"title": f"Lead {topic} Architect", "description": "Drive technical strategy and roadmap.", "salary": "$130k–$160k/yr"},
                {"title": "Engineering Director", "description": "Manage teams, budgets, and delivery.", "salary": "$150k+/yr"},
            ],
        },
        "salary_insights": {
            "Entry Level": "$55k–$75k / year",
            "Mid Level": "$80k–$110k / year",
            "Senior / Lead": "$120k–$160k / year",
            "Director / Principal": "$160k+ / year",
        },
        "capstone_projects": [
            {
                "title": f"End-to-End {topic} Platform",
                "description": f"Build a fully functional {topic} platform integrating all course concepts, deployed on cloud infrastructure with CI/CD pipelines.",
                "tech_stack": ["Python", "Docker", "PostgreSQL", "React", "AWS/GCP"],
                "deliverables": ["Working prototype", "Technical report (15+ pages)", "Live demo presentation", "GitHub repository"],
                "difficulty": "Advanced",
            },
            {
                "title": f"{field} Analytics Dashboard",
                "description": f"Design an interactive analytics dashboard processing real-world {field} data and visualising insights.",
                "tech_stack": ["Python", "Pandas", "Chart.js", "Flask", "SQLite"],
                "deliverables": ["Interactive dashboard", "Data pipeline", "API documentation"],
                "difficulty": "Intermediate",
            },
            {
                "title": f"AI-Powered {topic} Recommendation Engine",
                "description": f"Develop a machine-learning recommendation system for {field} use cases with explainability features.",
                "tech_stack": ["Python", "Scikit-Learn", "FastAPI", "React"],
                "deliverables": ["ML model", "REST API", "Front-end UI", "Model evaluation report"],
                "difficulty": "Advanced",
            },
        ],
    }


def generate_capstone_guidelines(curriculum_title, field, courses_summary):
    """
    Generates a structured capstone project guideline for the full curriculum.
    Falls back to a high-quality mock if Ollama is unavailable.
    """
    prompt = f"""You are an academic curriculum expert. Generate a detailed capstone project guideline for a curriculum titled "{curriculum_title}" in the field of "{field}".

Courses covered: {courses_summary}

Output a single JSON object only. No markdown, no extra text.

JSON Schema:
{{
  "title": "Capstone Project Title",
  "overview": "2-3 sentence project overview",
  "objectives": ["objective 1", "objective 2", "objective 3"],
  "deliverables": [
    {{"name": "Deliverable Name", "description": "What must be submitted"}}
  ],
  "milestones": [
    {{"phase": "Phase Name", "duration": "X weeks", "tasks": ["task 1", "task 2"]}}
  ],
  "evaluation_criteria": [
    {{"criterion": "Criterion Name", "weight": "X%", "description": "How it is assessed"}}
  ],
  "recommended_tools": ["Tool 1", "Tool 2"],
  "suggested_topics": ["Topic idea 1", "Topic idea 2", "Topic idea 3"]
}}"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
            timeout=15
        )
        if response.status_code == 200:
            raw = response.json().get("response", "").strip()
            raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
    except Exception as e:
        print(f"Capstone generation failed: {e}")
    return _mock_capstone(curriculum_title, field)

def _mock_capstone(title, field):
    return {
        "title": f"Capstone Project: {title}",
        "overview": f"This capstone project challenges students to integrate all concepts learned across the {title} curriculum into a single, deployable, industry-grade solution in the domain of {field}.",
        "objectives": [
            "Apply theoretical knowledge to a real-world problem",
            "Demonstrate full-stack or end-to-end system design skills",
            "Produce professional documentation and a working prototype",
            "Present findings to a panel of evaluators"
        ],
        "deliverables": [
            {"name": "Project Proposal", "description": "1-2 page document outlining problem statement, objectives, and tech stack"},
            {"name": "Working Prototype", "description": "Functional application or model meeting defined requirements"},
            {"name": "Technical Report", "description": "10-15 page report covering architecture, implementation, and evaluation"},
            {"name": "Final Presentation", "description": "15-minute demo and Q&A session with evaluators"}
        ],
        "milestones": [
            {"phase": "Research & Planning", "duration": "2 weeks", "tasks": ["Define problem scope", "Review literature", "Select technology stack"]},
            {"phase": "Design & Architecture", "duration": "2 weeks", "tasks": ["Create system diagrams", "Define data models", "Set up project repository"]},
            {"phase": "Implementation", "duration": "4 weeks", "tasks": ["Build core features", "Integrate components", "Write unit tests"]},
            {"phase": "Testing & Refinement", "duration": "1 week", "tasks": ["User acceptance testing", "Bug fixes", "Performance optimization"]},
            {"phase": "Documentation & Presentation", "duration": "1 week", "tasks": ["Finalize report", "Prepare slides", "Record demo video"]}
        ],
        "evaluation_criteria": [
            {"criterion": "Technical Complexity", "weight": "30%", "description": "Depth of implementation and appropriate use of course concepts"},
            {"criterion": "Functionality", "weight": "25%", "description": "Working prototype meeting all defined requirements"},
            {"criterion": "Documentation Quality", "weight": "20%", "description": "Clarity, completeness, and professional standard of written report"},
            {"criterion": "Presentation & Defence", "weight": "15%", "description": "Clear communication and ability to answer questions"},
            {"criterion": "Innovation", "weight": "10%", "description": "Originality and creative problem-solving approach"}
        ],
        "recommended_tools": ["Git / GitHub", "VS Code", "Docker", "Postman", "Figma", "Jupyter Notebook"],
        "suggested_topics": [
            f"AI-powered {field} recommendation system",
            f"Full-stack {field} management platform",
            f"Data analytics dashboard for {field} metrics"
        ]
    }



def generate_mock_curriculum(title, field, duration):
    """
    Generates realistic, highly structured curricula based on keywords in title/field.
    """
    duration = int(duration)
    field_lower = field.lower()
    title_lower = title.lower()
    combined = field_lower + " " + title_lower

    if any(k in combined for k in ["healthcare", "health", "medical", "clinical", "biomedical", "hospital", "patient", "diagnosis"]):
        topic_pool = [
            ("HI101", "Introduction to Health Informatics", "Overview of healthcare systems, electronic health records (EHR), health data standards (HL7, FHIR), and patient data management.", 3, "Beginner", 92, ["Health Informatics Analyst"], ["EHR System Prototype"], ["Health Data", "HL7"],
             [{"title": "HL7 FHIR Documentation", "url": "https://hl7.org/fhir/"}, {"title": "Health IT on Coursera", "url": "https://www.coursera.org/search?query=health+informatics"}]),
            ("HI102", "Medical Data Analysis with Python", "Applying Python (Pandas, NumPy) to clean, analyse, and visualise clinical datasets including patient records and lab results.", 4, "Beginner", 93, ["Clinical Data Analyst"], ["Patient Readmission Analysis"], ["Python", "Medical Data"],
             [{"title": "Kaggle Healthcare Datasets", "url": "https://www.kaggle.com/search?q=healthcare"}, {"title": "Google Scholar: Medical Data Analysis", "url": "https://scholar.google.com/scholar?q=medical+data+analysis+python"}]),
            ("HI201", "AI & Machine Learning in Healthcare", "Supervised and unsupervised ML applied to disease prediction, medical image classification, and patient outcome modelling.", 4, "Intermediate", 98, ["Healthcare AI Engineer"], ["Diabetes Risk Predictor"], ["Machine Learning", "Scikit-Learn"],
             [{"title": "Coursera: AI for Medicine", "url": "https://www.coursera.org/specializations/ai-for-medicine"}, {"title": "Google Scholar: ML Healthcare", "url": "https://scholar.google.com/scholar?q=machine+learning+healthcare"}]),
            ("HI202", "Medical Imaging & Computer Vision", "CNN-based analysis of X-ray, MRI, and CT scan images for automated diagnosis support using TensorFlow and Keras.", 4, "Intermediate", 97, ["Medical Imaging Specialist"], ["Chest X-Ray Classifier"], ["Computer Vision", "CNN"],
             [{"title": "Kaggle: Chest X-Ray Dataset", "url": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"}, {"title": "Coursera: AI for Medical Diagnosis", "url": "https://www.coursera.org/learn/ai-for-medical-diagnosis"}]),
            ("HI301", "Natural Language Processing for Clinical Text", "NLP techniques for processing clinical notes, discharge summaries, and medical literature using BERT and transformer models.", 4, "Intermediate", 96, ["Clinical NLP Engineer"], ["Medical Named Entity Recognition System"], ["NLP", "BERT"],
             [{"title": "HuggingFace BioBERT", "url": "https://huggingface.co/dmis-lab/biobert-v1.1"}, {"title": "Google Scholar: Clinical NLP", "url": "https://scholar.google.com/scholar?q=clinical+NLP+BERT"}]),
            ("HI302", "Healthcare Data Privacy & Ethics", "HIPAA compliance, data de-identification, algorithmic bias in clinical AI, and responsible AI deployment in healthcare.", 3, "Advanced", 91, ["Healthcare Compliance Officer"], ["Bias Audit of Clinical AI Model"], ["HIPAA", "Ethics"],
             [{"title": "HHS HIPAA Guidelines", "url": "https://www.hhs.gov/hipaa/index.html"}, {"title": "WHO Ethics & AI in Health", "url": "https://www.who.int/publications/i/item/9789240029200"}]),
            ("HI401", "Clinical Decision Support Systems", "Architecture of AI-powered CDSS, integration with EHR platforms, real-time patient alert systems, and validation frameworks.", 4, "Advanced", 97, ["CDSS Developer", "Health AI Architect"], ["Drug Interaction Alert System"], ["CDSS", "HL7 FHIR"],
             [{"title": "HL7 CDS Hooks", "url": "https://cds-hooks.hl7.org/"}, {"title": "Google Scholar: CDSS", "url": "https://scholar.google.com/scholar?q=clinical+decision+support+systems+AI"}]),
            ("HI402", "Healthcare AI Deployment & Regulation", "FDA guidelines for AI/ML-based medical devices, deployment pipelines, monitoring, and regulatory submission processes.", 3, "Advanced", 94, ["Regulatory Affairs Specialist"], ["AI Model Validation Report"], ["FDA Regulation", "Model Monitoring"],
             [{"title": "FDA AI/ML Action Plan", "url": "https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices"}, {"title": "Google Scholar: FDA AI", "url": "https://scholar.google.com/scholar?q=FDA+AI+ML+medical+devices"}])
        ]
    elif any(k in combined for k in ["disaster", "emergency", "crisis", "relief", "rescue", "hazard", "risk management", "humanitarian"]):
        topic_pool = [
            ("DM101", "Introduction to Disaster Management", "Fundamentals of disaster risk reduction, types of hazards (natural/man-made), disaster cycle phases, and national/international frameworks (Sendai, Hyogo).", 3, "Beginner", 88, ["Disaster Risk Officer"], ["Hazard Risk Assessment Report"], ["Risk Assessment", "DRR"],
             [{"title": "UNDRR Sendai Framework", "url": "https://www.undrr.org/implementing-sendai-framework"}, {"title": "Coursera: Disaster Preparedness", "url": "https://www.coursera.org/search?query=disaster+management"}]),
            ("DM102", "Hazard & Vulnerability Assessment", "Methodologies for identifying, mapping, and analysing natural and technological hazards, community vulnerability, and exposure analysis using GIS tools.", 4, "Beginner", 90, ["Risk Analyst", "GIS Specialist"], ["Community Vulnerability Map"], ["GIS", "Risk Mapping"],
             [{"title": "FEMA Hazard Mitigation", "url": "https://www.fema.gov/emergency-managers/risk-management/hazard-mitigation-planning"}, {"title": "Google Scholar: Hazard Assessment", "url": "https://scholar.google.com/scholar?q=hazard+vulnerability+assessment"}]),
            ("DM201", "Emergency Response & Operations", "Search and rescue operations, incident command systems (ICS), mass casualty management, logistics coordination, and field communication protocols.", 4, "Intermediate", 93, ["Emergency Response Coordinator"], ["Incident Command Simulation"], ["ICS", "Emergency Operations"],
             [{"title": "FEMA ICS Training", "url": "https://training.fema.gov/ics/"}, {"title": "WHO Emergency Response", "url": "https://www.who.int/emergencies/operations"}]),
            ("DM202", "Disaster Preparedness & Community Resilience", "Community-based disaster preparedness planning, early warning systems, public awareness campaigns, and building local resilience frameworks.", 3, "Intermediate", 89, ["Community Resilience Officer"], ["Local Disaster Preparedness Plan"], ["Community DRR", "Early Warning"],
             [{"title": "UNDP Community Resilience", "url": "https://www.undp.org/tag/community-resilience"}, {"title": "Google Scholar: Disaster Preparedness", "url": "https://scholar.google.com/scholar?q=community+disaster+preparedness"}]),
            ("DM301", "Disaster Relief & Humanitarian Operations", "Principles of humanitarian assistance, refugee camp management, WASH (water/sanitation/hygiene), food security, and inter-agency coordination (UN OCHA, ICRC, NGOs).", 4, "Intermediate", 92, ["Humanitarian Aid Worker", "Relief Coordinator"], ["Refugee Camp Resource Plan"], ["Humanitarian Aid", "WASH"],
             [{"title": "OCHA Relief Operations", "url": "https://www.unocha.org/"}, {"title": "ICRC Humanitarian Law", "url": "https://www.icrc.org/en/war-and-law"}]),
            ("DM302", "Climate Change & Disaster Risk", "Link between climate change and increasing disaster frequency, climate adaptation strategies, loss and damage frameworks, and green resilience infrastructure.", 3, "Intermediate", 91, ["Climate Risk Analyst"], ["Climate Adaptation Plan for a Coastal City"], ["Climate Adaptation", "DRR"],
             [{"title": "IPCC Climate Reports", "url": "https://www.ipcc.ch/reports/"}, {"title": "UNDRR Climate & Risk", "url": "https://www.undrr.org/implementing-sendai-framework/sendai-framework-disaster-risk-reduction"}]),
            ("DM401", "Disaster Recovery & Reconstruction", "Post-disaster needs assessment, livelihood restoration, infrastructure reconstruction, psychosocial support, and building back better principles.", 4, "Advanced", 90, ["Recovery Programme Manager"], ["Post-Disaster Needs Assessment Report"], ["Recovery Planning", "Build Back Better"],
             [{"title": "World Bank Disaster Recovery", "url": "https://www.worldbank.org/en/topic/disasterriskmanagement"}, {"title": "Google Scholar: Disaster Recovery", "url": "https://scholar.google.com/scholar?q=disaster+recovery+reconstruction"}]),
            ("DM402", "Policy, Law & International Frameworks", "Disaster risk governance, national disaster management acts, international humanitarian law, Sustainable Development Goals (SDGs), and institutional coordination mechanisms.", 3, "Advanced", 88, ["Policy Analyst", "Disaster Risk Governance Officer"], ["National DRR Policy Brief"], ["Disaster Policy", "International Law"],
             [{"title": "Sendai Framework Monitor", "url": "https://sendaimonitor.undrr.org/"}, {"title": "UN SDG & DRR", "url": "https://sdgs.un.org/goals"}])
        ]
    elif any(k in combined for k in ["computer", "software", "programming", "engineering", "web", "mobile", "backend", "frontend"]):
        topic_pool = [
            ("CS101", "Introduction to Python Programming", "Fundamental control flows, variables, data structures (lists, dicts), and writing clean modular programs in Python.", 3, "Beginner", 90, ["Python Developer"], ["CLI Calculator", "Contact Book App"], ["Python", "Algorithms"]),
            ("CS102", "Object-Oriented Design", "Advanced concepts of OOP including encapsulation, inheritance, polymorphism, design patterns, and UML diagramming.", 4, "Beginner", 88, ["Software Intern"], ["Library Management System"], ["OOP", "UML"]),
            ("CS201", "Data Structures & Algorithms", "Detailed study of arrays, linked lists, stacks, queues, trees, graphs, sorting/searching algorithms, and big-O time complexity analysis.", 4, "Intermediate", 95, ["Backend Engineer"], ["Maze Solver using DFS/BFS"], ["Data Structures", "Algorithms"]),
            ("CS202", "Database Management Systems", "Relational database concepts, SQL query optimization, database design principles (normalization), and intro to NoSQL databases.", 3, "Intermediate", 92, ["Database Administrator", "Backend Developer"], ["E-commerce Database Design"], ["SQL", "Relational DB"]),
            ("CS301", "Web Development & APIs", "Full-stack development foundations. Front-end HTML/CSS/JS alongside backend Flask server creation, REST APIs, and AJAX requests.", 4, "Intermediate", 96, ["Full-Stack Developer"], ["Social Media API clone"], ["JavaScript", "APIs", "Flask"]),
            ("CS302", "Artificial Intelligence Foundations", "Foundations of AI, machine learning pipelines, regression, classification, clustering, neural networks, and prompt engineering.", 4, "Advanced", 98, ["AI Engineer", "ML Practitioner"], ["Spam Email Classifier", "Gemini API Assistant"], ["Machine Learning", "Neural Networks"]),
            ("CS401", "Software Engineering & Devops", "Agile methodologies, testing (unit, integration), version control with Git, CI/CD pipelines, Docker containerization, and cloud deployment.", 4, "Advanced", 94, ["DevOps Engineer", "Release Engineer"], ["Automated CI/CD Pipeline deployment"], ["Docker", "CI/CD", "Git"]),
            ("CS402", "Cybersecurity & Networks", "Fundamentals of computer networks, TCP/IP stack, encryption algorithms, hashing, public key cryptography, and common web security threats.", 3, "Advanced", 93, ["Security Consultant", "Network Analyst"], ["Secure File Transfer Utility"], ["Encryption", "Security"])
        ]
    elif any(k in combined for k in ["data science", "data analytics", "machine learning", "ml", "statistics", "artificial intelligence", "neural", "deep learning", "data engineering"]):
        topic_pool = [
            ("DS101", "Foundations of Data Science", "Introduction to Jupyter Notebooks, Pandas, Numpy, data cleaning, and basic exploratory data analysis (EDA).", 3, "Beginner", 94, ["Data Analyst"], ["Exploratory Analysis of Housing Market"], ["Pandas", "Python"]),
            ("DS102", "Applied Probability & Statistics", "Probability theory, random variables, hypothesis testing, ANOVA, linear regression, and statistical inference.", 4, "Beginner", 90, ["Junior Statistician"], ["A/B Testing Simulator"], ["Statistics", "R"]),
            ("DS201", "Data Visualization Techniques", "Creating impactful visualizations using Matplotlib, Seaborn, Tableau, and Chart.js. Storytelling with data.", 3, "Intermediate", 91, ["BI Analyst"], ["Interactive COVID-19 Dashboard"], ["Tableau", "Chart.js"]),
            ("DS202", "Machine Learning: Supervised Learning", "Linear & logistic regression, decision trees, random forests, support vector machines, and ensemble methods.", 4, "Intermediate", 97, ["Machine Learning Engineer"], ["Predicting Customer Churn"], ["Scikit-Learn", "Machine Learning"]),
            ("DS301", "Unsupervised Learning & NLP", "Clustering (K-means, Hierarchical), Principal Component Analysis (PCA), text tokenization, sentiment analysis, and vector embeddings.", 4, "Intermediate", 95, ["NLP Specialist"], ["Customer Segmentation Tool"], ["NLP", "Clustering"]),
            ("DS302", "Big Data Engineering", "Data pipelines using Apache Spark, Hadoop, SQL databases, ETL flows, and running analytics on distributed clusters.", 4, "Advanced", 96, ["Data Engineer"], ["Log Analytics Pipeline with Spark"], ["Spark", "ETL"]),
            ("DS401", "Deep Learning & AI Systems", "Multi-layer perceptrons, Convolutional Neural Networks (CNNs) for vision, and Recurrent Neural Networks (RNNs) for sequential data.", 4, "Advanced", 98, ["Computer Vision Engineer"], ["Digit Recognizer with PyTorch"], ["Deep Learning", "PyTorch"]),
            ("DS402", "Data Ethics & Privacy", "Understanding GDPR, ethical biases in algorithms, data anonymization techniques, and transparency in predictive modeling.", 3, "Advanced", 89, ["Compliance Officer"], ["Audit Report of Bias in Hiring AI"], ["Ethics", "Bias Auditing"])
        ]
    else:
        # Dynamic fallback: generate topic-aware courses from the title/field directly
        # This handles ANY topic the user enters that doesn't match known domains
        words = [w.capitalize() for w in title.split() if len(w) > 3]
        short = " ".join(words[:3]) if words else title
        T = title  # full title for course names
        F = field  # field of study

        topic_pool = [
            ("F101", f"Introduction to {T}",
             f"Foundational concepts, history, and scope of {T}. Overview of key theories, terminologies, and the role of {F} in modern society.",
             3, "Beginner", 85,
             [f"{short} Analyst", f"Junior {short} Specialist"],
             [f"Introductory {short} Case Study Report"],
             [f"{short} Fundamentals", "Research Methods"],
             [{"title": f"Google Scholar: {T}", "url": f"https://scholar.google.com/scholar?q={T.replace(' ','+')}"},
              {"title": f"Coursera: {T}", "url": f"https://www.coursera.org/search?query={T.replace(' ','%20')}"}]),

            ("F102", f"Principles and Theories of {T}",
             f"In-depth study of core principles, models, and theoretical frameworks underpinning {T}. Comparative analysis of classical and contemporary approaches.",
             4, "Beginner", 87,
             [f"{short} Researcher", f"Policy Analyst"],
             [f"Theoretical Framework Analysis of {short}"],
             [f"{short} Theory", "Critical Analysis"],
             [{"title": f"Wikipedia: {T}", "url": f"https://en.wikipedia.org/wiki/Special:Search?search={T.replace(' ','_')}"},
              {"title": f"YouTube: {T} Explained", "url": f"https://www.youtube.com/results?search_query={T.replace(' ','+')}+explained"}]),

            ("F201", f"Research Methods in {T}",
             f"Quantitative and qualitative research design, data collection techniques, statistical analysis, and academic writing conventions applied to {F}.",
             3, "Intermediate", 88,
             [f"Research Associate", f"{short} Consultant"],
             [f"Research Proposal on a {short} Problem"],
             ["Research Design", "Data Analysis"],
             [{"title": f"Google Scholar: {T} Research", "url": f"https://scholar.google.com/scholar?q={T.replace(' ','+')}+research+methods"},
              {"title": "Kaggle Datasets", "url": f"https://www.kaggle.com/search?q={T.replace(' ','%20')}"}]),

            ("F202", f"Applied {T}: Case Studies",
             f"Real-world application of {T} principles through case studies, problem-based learning, and field analysis. Students analyse successes and failures in {F}.",
             4, "Intermediate", 90,
             [f"Field Practitioner", f"{short} Officer"],
             [f"Case Study Analysis: {short} in Practice"],
             ["Case Analysis", "Problem Solving"],
             [{"title": f"GitHub: {T} Projects", "url": f"https://github.com/search?q={T.replace(' ','+')}+&type=repositories"},
              {"title": f"ResearchGate: {T}", "url": f"https://www.researchgate.net/search?q={T.replace(' ','%20')}"}]),

            ("F301", f"Technology & Innovation in {T}",
             f"Emerging technologies transforming {F}: digital tools, data-driven approaches, automation, and innovation ecosystems relevant to {T}.",
             4, "Intermediate", 92,
             [f"{short} Technology Specialist", "Innovation Manager"],
             [f"Technology Impact Assessment for {short}"],
             ["Digital Innovation", "Technology Adoption"],
             [{"title": f"Coursera: Technology in {T}", "url": f"https://www.coursera.org/search?query={T.replace(' ','%20')}+technology"},
              {"title": f"Google Scholar: Innovation in {T}", "url": f"https://scholar.google.com/scholar?q=innovation+in+{T.replace(' ','%20')}"}]),

            ("F302", f"Ethics, Policy & Governance in {T}",
             f"Ethical frameworks, regulatory landscape, policy development, and governance structures governing {F}. Stakeholder analysis and accountability mechanisms.",
             3, "Intermediate", 89,
             ["Policy Officer", f"{short} Governance Analyst"],
             [f"Policy Brief on {short} Regulation"],
             ["Ethics", "Policy Analysis"],
             [{"title": f"UN Policy Documents on {T}", "url": f"https://www.un.org/en/search?q={T.replace(' ','%20')}"},
              {"title": f"Google Scholar: {T} Policy", "url": f"https://scholar.google.com/scholar?q={T.replace(' ','+')}+policy+governance"}]),

            ("F401", f"Strategic Management of {T}",
             f"Strategic planning, programme management, resource allocation, monitoring and evaluation frameworks applied to {F} organisations and initiatives.",
             4, "Advanced", 91,
             [f"{short} Programme Manager", "Strategy Consultant"],
             [f"Strategic Plan for a {short} Organisation"],
             ["Strategic Planning", "M&E Frameworks"],
             [{"title": f"Coursera: Strategy & Management", "url": "https://www.coursera.org/search?query=strategic+management"},
              {"title": f"Google Scholar: {T} Management", "url": f"https://scholar.google.com/scholar?q={T.replace(' ','+')}+strategic+management"}]),

            ("F402", f"Capstone Project: {T}",
             f"Integrative capstone project requiring students to apply all acquired knowledge and skills to design, execute, and present a comprehensive {F} project addressing a real-world problem.",
             4, "Advanced", 94,
             [f"Senior {short} Specialist", "Project Lead"],
             [f"End-to-End {short} Capstone Project"],
             ["Project Management", "Research & Development"],
             [{"title": f"Google Scholar: {T} Capstone", "url": f"https://scholar.google.com/scholar?q={T.replace(' ','+')}+capstone+project"},
              {"title": f"GitHub: {T} Projects", "url": f"https://github.com/search?q={T.replace(' ','+')}+project"}]),
        ]

    semesters = []
    course_index = 0
    
    for sem in range(1, duration + 1):
        semester_courses = []
        num_courses = 2 if duration >= 6 else 3
        for _ in range(num_courses):
            orig_course = topic_pool[course_index % len(topic_pool)]
            prereqs = []
            if sem > 1:
                prev_sem_c = semesters[sem - 2]['courses'][0]['code']
                prereqs.append(prev_sem_c)
                
            course_data = {
                "code": f"{orig_course[0]}-S{sem}",
                "name": orig_course[1],
                "description": orig_course[2],
                "credits": orig_course[3],
                "learning_outcomes": [
                    f"Master the core principles of {orig_course[1]}.",
                    f"Successfully complete the hands-on project: {orig_course[6][0]}.",
                    f"Demonstrate fluency in using: {', '.join(orig_course[8])}."
                ],
                "prerequisites": prereqs,
                "assessment_methods": [
                    "Individual Practical Assignments: 40%",
                    "Mid-term Conceptual Test: 30%",
                    "Final Project Submission: 30%"
                ],
                "difficulty": orig_course[4],
                "learning_time_weeks": 15,
                "weekly_hours": orig_course[3] * 2,
                "industry_relevance_score": orig_course[5] + random.randint(-5, 2),
                "career_opportunities": orig_course[6],
                "recommended_projects": orig_course[7],
                "key_skills": orig_course[8],
                "resources": orig_course[9] if len(orig_course) > 9 else [
                    {"title": f"Google Scholar: {orig_course[1]}", "url": f"https://scholar.google.com/scholar?q={orig_course[1].replace(' ', '+')}"},
                    {"title": f"Coursera: {orig_course[8][0] if orig_course[8] else 'Introduction'}", "url": f"https://www.coursera.org/search?query={orig_course[8][0].replace(' ', '%20') if orig_course[8] else 'IT'}"}
                ]
            }
            if course_data["industry_relevance_score"] > 100:
                course_data["industry_relevance_score"] = 100
                
            semester_courses.append(course_data)
            course_index += 1
            
        semesters.append({
            "semester_number": sem,
            "courses": semester_courses
        })
        
    return {
        "title": title,
        "field_of_study": field,
        "duration_semesters": duration,
        "semesters": semesters,
        "resource_sources": [
            "ACM/IEEE Joint Curriculum Guidelines (Mock Fallback)",
            "IBM Skills Network Course Frameworks (Mock Fallback)",
            "Hugging Face Applied NLP Curriculum (Mock Fallback)",
            "LERNIX Smart Heuristic Pipeline"
        ]
    }

def generate_mock_study_plan(course_name, weekly_hours):
    """Generates a detailed study plan with unique topics per week based on the course."""
    weekly_hours = int(weekly_hours)
    cn = course_name.lower()

    # Course-aware week topic progressions
    if any(k in cn for k in ["disaster", "emergency", "crisis", "hazard", "relief", "rescue", "humanitarian", "resilience", "recovery"]):
        week_topics = [
            ("Disaster Risk Fundamentals & Hazard Assessment",
             ["Disaster cycle phases: mitigation, preparedness, response, recovery", "Types of hazards: natural, technological, complex emergencies", "GIS-based hazard mapping and vulnerability analysis"],
             "Produce a hazard risk assessment map for a sample region"),
            ("Emergency Response Operations & ICS",
             ["Incident Command System (ICS) structure and roles", "Search and rescue coordination and triage protocols", "Logistics, communication, and resource mobilisation in the field"],
             "Run a tabletop emergency response simulation exercise"),
            ("Humanitarian Relief & Community Resilience",
             ["WASH, food security, and shelter standards in relief operations", "Community-based disaster preparedness planning", "Inter-agency coordination: UN OCHA, ICRC, NGO clusters"],
             "Develop a community disaster preparedness plan"),
            ("Recovery, Policy & International Frameworks",
             ["Post-disaster needs assessment and build-back-better principles", "Sendai Framework, SDGs, and national DRR legislation", "Climate change adaptation and long-term resilience strategies"],
             "Write a policy brief on a disaster recovery programme"),
        ]
    elif any(k in cn for k in ["healthcare", "health", "medical", "clinical", "biomedical"]):
        week_topics = [
            ("Health Data Foundations & EHR Systems",
             ["Overview of EHR systems and HL7/FHIR standards", "Loading and exploring clinical datasets with Pandas", "Data cleaning: missing values, outliers in patient records"],
             "Perform exploratory analysis on a public health dataset (e.g. MIMIC-III sample)"),
            ("AI & ML Models for Disease Prediction",
             ["Supervised learning for diagnosis classification (diabetes, heart disease)", "Feature engineering on clinical lab values", "Model evaluation: AUC-ROC, sensitivity, specificity"],
             "Build a diabetes risk prediction model using Scikit-Learn"),
            ("Medical Imaging & Computer Vision",
             ["CNN architecture for X-ray / MRI classification", "Transfer learning with pre-trained models (ResNet, EfficientNet)", "Grad-CAM explainability for clinical decisions"],
             "Train a chest X-ray classifier to detect pneumonia"),
            ("Clinical NLP, Ethics & Deployment",
             ["NLP on clinical notes using BERT/BioBERT", "HIPAA compliance, data de-identification, algorithmic bias", "Deploy healthcare AI model with audit logging"],
             "Build a clinical named-entity recognition (NER) pipeline"),
        ]
    elif any(k in cn for k in ["python", "programming", "software", "problem solving", " c ", "c language", "c programming", "through c"]):
        week_topics = [
            ("Environment Setup & Python Syntax Basics",
             ["Install Python & VS Code, write first scripts", "Variables, data types, operators", "Control flow: if/else, loops"],
             "Write a working CLI menu application"),
            ("Functions, Modules & Data Structures",
             ["Define and call functions with arguments", "Lists, tuples, dictionaries, sets", "Import and use standard library modules"],
             "Build a personal data organiser using dicts and lists"),
            ("Object-Oriented Programming",
             ["Classes, objects, constructors", "Inheritance, polymorphism, encapsulation", "Design and implement a mini OOP project"],
             "Implement a bank account class hierarchy"),
            ("File Handling, APIs & Error Management",
             ["Read/write files and CSV data", "Consume REST APIs with the requests library", "Exception handling with try/except"],
             "Build a weather app consuming a public API"),
        ]
    elif any(k in cn for k in ["data structures", "algorithms", "dsa"]):
        week_topics = [
            ("Arrays, Strings & Complexity Analysis",
             ["Big-O notation and time/space complexity", "Array traversals and two-pointer technique", "String manipulation problems"],
             "Solve 5 LeetCode Easy array problems"),
            ("Linked Lists, Stacks & Queues",
             ["Singly and doubly linked lists", "Stack operations and use cases", "Queue and deque implementations"],
             "Implement a stack-based expression evaluator"),
            ("Trees, Heaps & Graphs",
             ["Binary trees, BST operations", "BFS and DFS traversals", "Graph representation and shortest path"],
             "Implement BFS/DFS on a maze grid"),
            ("Sorting, Searching & Dynamic Programming",
             ["Merge sort, quick sort, heap sort", "Binary search and its variations", "DP fundamentals: memoisation and tabulation"],
             "Solve 3 medium DP problems on LeetCode"),
        ]
    elif any(k in cn for k in ["machine learning", "ml", "artificial intelligence", "ai"]):
        week_topics = [
            ("ML Fundamentals & Data Preprocessing",
             ["Supervised vs unsupervised learning concepts", "Data cleaning, normalisation, train/test split", "Exploratory data analysis with Pandas"],
             "Clean and analyse a real Kaggle dataset"),
            ("Regression & Classification Models",
             ["Linear and logistic regression from scratch", "Decision trees and random forests", "Model evaluation: accuracy, precision, recall, F1"],
             "Train a customer churn prediction model"),
            ("Unsupervised Learning & Feature Engineering",
             ["K-means and hierarchical clustering", "PCA for dimensionality reduction", "Feature selection and encoding strategies"],
             "Cluster customer segments on retail data"),
            ("Neural Networks & Model Deployment",
             ["Perceptrons, activation functions, backpropagation", "Build a CNN with TensorFlow/Keras", "Deploy model as a Flask REST API"],
             "Deploy a digit recognition model to a web API"),
        ]
    elif any(k in cn for k in ["web", "html", "css", "javascript", "react", "flask", "node"]):
        week_topics = [
            ("HTML5 & CSS3 Foundations",
             ["Document structure, semantic tags, accessibility", "CSS selectors, box model, flexbox layout", "Responsive design with media queries"],
             "Build a fully responsive portfolio landing page"),
            ("JavaScript & DOM Manipulation",
             ["Variables, functions, arrays, objects in JS", "DOM selection, events, and dynamic updates", "Fetch API and async/await patterns"],
             "Build an interactive quiz app in vanilla JS"),
            ("Backend Development with Flask/Node",
             ["REST API design principles", "Routes, request handling, JSON responses", "Database integration with SQLite/MongoDB"],
             "Build a CRUD REST API for a task manager"),
            ("Full-Stack Integration & Deployment",
             ["Connect frontend to backend API", "Authentication with JWT / session cookies", "Deploy to Render / Railway / Vercel"],
             "Deploy a full-stack task manager application live"),
        ]
    elif any(k in cn for k in ["database", "sql", "dbms"]):
        week_topics = [
            ("Relational Database Concepts & SQL Basics",
             ["Tables, primary keys, foreign keys, constraints", "SELECT, INSERT, UPDATE, DELETE statements", "Filtering with WHERE, ORDER BY, GROUP BY"],
             "Design and query a student records database"),
            ("Advanced SQL & Joins",
             ["INNER, LEFT, RIGHT, FULL OUTER JOINs", "Subqueries, views, and CTEs", "Aggregate functions and window functions"],
             "Write complex multi-table join queries on sample data"),
            ("Database Design & Normalisation",
             ["1NF, 2NF, 3NF normalisation rules", "ER diagrams and schema design", "Indexing and query optimisation"],
             "Normalise a denormalised e-commerce schema to 3NF"),
            ("NoSQL & Database Administration",
             ["MongoDB documents, collections, CRUD operations", "Redis caching fundamentals", "Backup, restore, and transaction management"],
             "Implement a product catalogue in MongoDB"),
        ]
    elif any(k in cn for k in ["cloud", "devops", "docker", "kubernetes", "aws", "azure"]):
        week_topics = [
            ("Cloud Computing Fundamentals",
             ["IaaS, PaaS, SaaS service models", "AWS/GCP/Azure core services overview", "Regions, availability zones, IAM basics"],
             "Launch and configure an EC2/Compute Engine instance"),
            ("Containerisation with Docker",
             ["Docker images, containers, Dockerfile syntax", "docker-compose for multi-service apps", "Container networking and volumes"],
             "Dockerise a Flask API with a PostgreSQL database"),
            ("CI/CD Pipelines",
             ["Git branching strategies and code reviews", "GitHub Actions workflow configuration", "Automated testing and deployment pipelines"],
             "Build a CI/CD pipeline that deploys on every push"),
            ("Kubernetes & Monitoring",
             ["Pods, deployments, services, and ingress", "Horizontal pod autoscaling", "Prometheus and Grafana monitoring basics"],
             "Deploy a containerised app on a local Kubernetes cluster"),
        ]
    elif any(k in cn for k in ["security", "cybersecurity", "network", "crypto"]):
        week_topics = [
            ("Security Fundamentals & Threat Landscape",
             ["CIA triad, threat actors, attack vectors", "OWASP Top 10 vulnerabilities overview", "Social engineering and phishing awareness"],
             "Perform a basic vulnerability scan with Nmap"),
            ("Cryptography & Secure Communications",
             ["Symmetric vs asymmetric encryption", "Hashing algorithms: SHA-256, bcrypt", "TLS/SSL handshake and certificate management"],
             "Implement AES encryption/decryption in Python"),
            ("Network Security & Firewalls",
             ["TCP/IP stack and packet analysis with Wireshark", "Firewall rules, VPNs, and DMZ design", "Intrusion detection systems (IDS/IPS)"],
             "Capture and analyse HTTP/HTTPS packets"),
            ("Ethical Hacking & Incident Response",
             ["Penetration testing methodology", "SQL injection, XSS, CSRF exploitation labs", "Incident response plan and forensics basics"],
             "Complete a CTF challenge on HackTheBox/TryHackMe"),
        ]
    elif any(k in cn for k in ["data science", "analytics", "statistics"]):
        week_topics = [
            ("Statistics & Probability Foundations",
             ["Descriptive statistics: mean, median, variance", "Probability distributions and Bayes theorem", "Hypothesis testing and p-values"],
             "Perform statistical analysis on a housing dataset"),
            ("Data Wrangling with Pandas & NumPy",
             ["DataFrame operations, merging, groupby", "Handling missing values and outliers", "Data type conversions and feature extraction"],
             "Clean a messy real-world CSV dataset end-to-end"),
            ("Data Visualisation",
             ["Matplotlib and Seaborn for statistical plots", "Interactive charts with Plotly", "Dashboard design with Streamlit"],
             "Build an interactive COVID-19 data dashboard"),
            ("Predictive Analytics & Reporting",
             ["Time series analysis and forecasting", "A/B testing and experiment design", "Business report generation and storytelling with data"],
             "Forecast next-quarter sales using time series models"),
        ]
    else:
        # Generic course-specific progression
        cn_title = course_name.split()[:2]
        base = " ".join(cn_title) if cn_title else course_name
        week_topics = [
            (f"Foundations of {course_name}",
             [f"Core terminology and concepts of {base}",
              f"Environment setup and tool installation for {base}",
              f"First hands-on exercise with {base}"],
             f"Complete introductory lab for {base}"),
            (f"Core Principles & Architecture of {course_name}",
             [f"Deep-dive into {base} architecture and design patterns",
              f"Study real-world {base} case studies",
              f"Implement a basic {base} module from scratch"],
             f"Build and document a {base} prototype component"),
            (f"Applied Implementation: {course_name}",
             [f"Integrate {base} with external tools and APIs",
              f"Debug and optimise a {base} project",
              f"Write unit tests for {base} logic"],
             f"Submit a working {base} mini-project with tests"),
            (f"Advanced Topics & Project Finalisation: {course_name}",
             [f"Explore advanced {base} patterns and edge cases",
              f"Conduct peer code review of {base} implementations",
              f"Prepare final project documentation"],
             f"Present and demo completed {base} capstone project"),
        ]

    q = course_name.replace(' ', '+')
    resource_pool = [
        [
            {"title": f"Google Scholar: {course_name}", "url": f"https://scholar.google.com/scholar?q={q}"},
            {"title": "Coursera Courses", "url": f"https://www.coursera.org/search?query={q}"},
        ],
        [
            {"title": "YouTube Tutorials", "url": f"https://www.youtube.com/results?search_query={q}+tutorial"},
            {"title": "GeeksforGeeks", "url": f"https://www.geeksforgeeks.org/search/?q={q}"},
        ],
        [
            {"title": "GitHub Repositories", "url": f"https://github.com/search?q={q}&type=repositories"},
            {"title": "Kaggle", "url": f"https://www.kaggle.com/search?q={q}"},
        ],
        [
            {"title": "Stack Overflow", "url": f"https://stackoverflow.com/search?q={q}"},
            {"title": "Wikipedia Overview", "url": f"https://en.wikipedia.org/wiki/Special:Search?search={q}"},
        ],
    ]

    study_schedule = []
    for i, (topic, tasks_raw, milestone) in enumerate(week_topics):
        h = weekly_hours
        total = h * 10  # proportional split basis
        tasks = [
            f"{tasks_raw[0]}  ({int(h*0.3)} hrs)",
            f"{tasks_raw[1]}  ({int(h*0.3)} hrs)",
            f"{tasks_raw[2]}  ({int(h*0.4)} hrs)",
        ]
        study_schedule.append({
            "week": i + 1,
            "topic": topic,
            "tasks": tasks,
            "resources": resource_pool[i % len(resource_pool)],
            "milestone": milestone,
        })

    revision_roadmap = [
        {
            "phase": "Phase 1: Foundation Building",
            "timeline": "Week 1",
            "focus": f"Core terminology, tooling, and environment setup for {course_name}.",
            "checkpoints": ["Complete setup checklist", "Finish Week 1 quiz"]
        },
        {
            "phase": "Phase 2: Concept Deep Dive",
            "timeline": "Weeks 2–3",
            "focus": f"Architecture, design patterns, and applied coding in {course_name}.",
            "checkpoints": ["Submit Week 2 mini-project", "Pass mid-point assessment"]
        },
        {
            "phase": "Phase 3: Project & Review",
            "timeline": "Week 4",
            "focus": f"Integration, optimisation, and capstone demonstration for {course_name}.",
            "checkpoints": ["Run full test suite", "Present final project demo"]
        },
    ]

    interview_prep = [
        {
            "topic": f"Core {course_name} Concepts",
            "sample_questions": [
                {
                    "question": f"What are the primary design principles applied in {course_name}?",
                    "answer": "The primary focus is efficiency, modular architecture, and industry compliance — ensuring systems are scalable, reusable, and maintainable."
                },
                {
                    "question": f"How would you handle scalability challenges in {course_name}?",
                    "answer": "Scale is addressed through horizontal replication, caching layers, and load balancing. Error propagation is managed via structured exception handling and centralised logging."
                },
                {
                    "question": f"Describe a real-world project you would build using {course_name}.",
                    "answer": f"An end-to-end {course_name} system covering data ingestion, processing logic, a REST API layer, and a web dashboard — deployed on cloud infrastructure with CI/CD automation."
                }
            ]
        }
    ]

    return {
        "course_name": course_name,
        "weekly_hours_allocated": weekly_hours,
        "weekly_schedule": study_schedule,
        "revision_roadmap": revision_roadmap,
        "interview_preparation": interview_prep,
        "resource_sources": [
            "LeetCode Interview Guides",
            "GeeksforGeeks Academic Prep Sheets",
            "LERNIX Smart Heuristic Student Assistant"
        ]
    }
