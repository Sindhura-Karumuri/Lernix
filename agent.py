"""
Agent — ReAct-style tool-calling agent for Lernix.

The agent exposes a set of tools to the LLM. On each turn it:
  1. Sends the user question + available tools to the LLM.
  2. Parses the LLM's chosen action (tool name + input).
  3. Executes the tool and feeds the observation back.
  4. Repeats until the LLM emits a final answer (max 3 iterations).

Tools available:
  - get_course_info     : look up a course by name in the curriculum
  - search_curriculum   : semantic search over curriculum chunks (RAG)
  - get_quiz_hint       : return a relevant quiz question for a topic
  - get_career_paths    : list career opportunities for a course/skill
"""

import os
import json
import re
import requests

import database
import rag_engine

_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL = f"{_base}/api/generate"
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "granite3.3:latest")
MAX_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Tool definitions (schema shown to the LLM)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_course_info",
        "description": "Retrieve detailed information about a specific course by name from the curriculum.",
        "parameters": {"course_name": "string — the course name to look up"},
    },
    {
        "name": "search_curriculum",
        "description": "Semantically search the curriculum for content relevant to a topic or question.",
        "parameters": {"query": "string — the topic or question to search for"},
    },
    {
        "name": "get_career_paths",
        "description": "List career opportunities and key skills associated with a course or skill area.",
        "parameters": {"topic": "string — course name or skill area"},
    },
    {
        "name": "get_quiz_hint",
        "description": "Return a sample quiz question and answer for a given topic to help with interview prep.",
        "parameters": {"topic": "string — the topic to get a quiz hint for"},
    },
]

TOOLS_SCHEMA = json.dumps(TOOLS, indent=2)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_get_course_info(course_name: str, curriculum_data: dict) -> str:
    for sem in curriculum_data.get("semesters", []):
        for course in sem.get("courses", []):
            if course_name.lower() in course.get("name", "").lower():
                return (
                    f"Course: {course['name']} ({course.get('code','')})\n"
                    f"Semester: {sem['semester_number']}\n"
                    f"Description: {course.get('description','')}\n"
                    f"Credits: {course.get('credits','')}\n"
                    f"Difficulty: {course.get('difficulty','')}\n"
                    f"Learning Outcomes: {'; '.join(course.get('learning_outcomes',[]))}\n"
                    f"Key Skills: {', '.join(course.get('key_skills',[]))}\n"
                    f"Prerequisites: {', '.join(course.get('prerequisites',[])) or 'None'}"
                )
    return f"No course matching '{course_name}' found in this curriculum."


def _tool_search_curriculum(query: str, store: rag_engine.VectorStore) -> str:
    context = rag_engine.retrieve_context_from_store(query, store, top_k=3)
    return context or "No relevant curriculum content found for that query."


def _tool_get_career_paths(topic: str, curriculum_data: dict) -> str:
    results = []
    for sem in curriculum_data.get("semesters", []):
        for course in sem.get("courses", []):
            name = course.get("name", "")
            if topic.lower() in name.lower() or any(
                topic.lower() in s.lower() for s in course.get("key_skills", [])
            ):
                careers = course.get("career_opportunities", [])
                skills = course.get("key_skills", [])
                results.append(
                    f"{name}: Careers — {', '.join(careers)}; Skills — {', '.join(skills)}"
                )
    return "\n".join(results) if results else f"No career data found for '{topic}'."


def _tool_get_quiz_hint(topic: str) -> str:
    # Import here to avoid circular dependency
    from llm_service import generate_course_quiz
    questions = generate_course_quiz(topic)
    if not questions:
        return "No quiz hint available for this topic."
    q = questions[0]
    correct_idx = q.get("correct_index", 0)
    options = q.get("options", [])
    answer = options[correct_idx] if correct_idx < len(options) else ""
    explanation = q.get("explanations", [""])[correct_idx] if q.get("explanations") else ""
    return (
        f"Sample question: {q['question']}\n"
        f"Answer: {answer}\n"
        f"Explanation: {explanation}"
    )


def _execute_tool(tool_name: str, tool_input: dict,
                  curriculum_data: dict, store: rag_engine.VectorStore) -> str:
    if tool_name == "get_course_info":
        return _tool_get_course_info(tool_input.get("course_name", ""), curriculum_data)
    if tool_name == "search_curriculum":
        return _tool_search_curriculum(tool_input.get("query", ""), store)
    if tool_name == "get_career_paths":
        return _tool_get_career_paths(tool_input.get("topic", ""), curriculum_data)
    if tool_name == "get_quiz_hint":
        return _tool_get_quiz_hint(tool_input.get("topic", ""))
    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, temperature: float = 0.3) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": DEFAULT_MODEL, "prompt": prompt,
                  "stream": False, "options": {"temperature": temperature}},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})", re.DOTALL
)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def run_agent(question: str, curriculum_id: int) -> str:
    """
    Run the ReAct agent for a student question against a specific curriculum.
    Returns the agent's final answer string.
    """
    curr = database.get_curriculum(curriculum_id)
    if not curr:
        return "Curriculum not found."

    curriculum_data = json.loads(curr["raw_json"])
    store = rag_engine.build_store_from_curriculum(curriculum_id)
    curriculum_title = curriculum_data.get("title", "this curriculum")

    system_prompt = f"""You are LERNIX AI Agent, an academic assistant for the curriculum "{curriculum_title}".
You have access to the following tools:

{TOOLS_SCHEMA}

Use this format strictly:
Thought: <your reasoning>
Action: <tool_name>
Action Input: {{"param": "value"}}

After receiving an Observation, continue reasoning. When you have enough information, respond with:
Final Answer: <your complete answer>

Only call one tool per step. Do not make up tool results."""

    history = ""
    for iteration in range(MAX_ITERATIONS):
        prompt = f"{system_prompt}\n\n{history}Question: {question}\n"
        raw = _call_llm(prompt)

        if not raw:
            break

        # Check for final answer
        final_match = _FINAL_RE.search(raw)
        if final_match:
            return final_match.group(1).strip()

        # Check for tool call
        action_match = _ACTION_RE.search(raw)
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                tool_input = json.loads(action_match.group(2))
            except json.JSONDecodeError:
                tool_input = {}

            observation = _execute_tool(tool_name, tool_input, curriculum_data, store)
            history += (
                f"Thought: {raw.split('Action:')[0].replace('Thought:','').strip()}\n"
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(tool_input)}\n"
                f"Observation: {observation}\n"
            )
        else:
            # LLM gave a direct response without tool call — treat as final
            return raw.strip()

    # Fallback: use RAG context directly
    context = rag_engine.retrieve_context_from_store(question, store, top_k=3)
    if context:
        fallback_prompt = (
            f"You are a helpful academic assistant for the curriculum '{curriculum_title}'.\n"
            f"{context}\n\nAnswer this question concisely: {question}"
        )
        answer = _call_llm(fallback_prompt, temperature=0.4)
        if answer:
            return answer
    return "I could not find a specific answer in this curriculum. Please consult your course materials."
