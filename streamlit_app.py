import pandas as pd
import google.generativeai as genai
import sys
import json
import re
import os
import time 
import streamlit as st 

st.set_page_config(page_title="AI Question Generator", layout="wide")

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Ensure these files exist in your folder
MATH_EXCEL_FILE_PATH = 'Math.xlsx'
SCIENCE_EXCEL_FILE_PATH = 'Science.xlsx'

GEMINI_MODEL = 'gemini-2.5-flash' 
QUESTION_COLUMN_NAME = 'Question Text'
QUESTIONS_TO_SELECT = 50 
MAX_RETRIES = 5 
API_DELAY_SECONDS = 5

# ==========================================
# 2. SESSION STATE
# ==========================================
def initialize_session_state():
    default_values = {
        "total_tokens_used": 0,
        "all_generated_questions": [],
        "latest_generated_list": [],
        "current_index": 0,
        "answer_checked": False, 
        "grading_feedback": None, 
        "question_type": "MCQ"    
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==========================================
# 3. CORE LOGIC
# ==========================================

def load_and_select_questions(file_path, num_questions, column_name):
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found.")
        return None

    st.info(f"Loading reference questions from '{file_path}'...")
    try:
        df = pd.read_excel(file_path) 
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

    if column_name not in df.columns:
        st.error(f"Error: Your Excel file must contain a column named '{column_name}'")
        return None
        
    df.dropna(subset=[column_name], inplace=True)
    available_questions = len(df)
    
    if available_questions == 0:
        st.error(f"Error: No questions found in the column '{column_name}'.")
        return None
        
    if available_questions < num_questions:
        st.warning(f"Warning: Only {available_questions} questions found. Selecting all of them.")
        num_questions = available_questions
        
    st.success(f"Successfully loaded {available_questions} total questions. Selecting {num_questions} random questions for this run...")
    questions_series = df[column_name].sample(n=num_questions)
    return questions_series

# --- UPDATED: Virtual Marker Logic ---
def grade_student_answer(question, model_answer, marking_scheme, student_answer, subject):
    """
    Grades the answer using the generated Marking Scheme.
    """
    if subject == "Math":
        grading_criteria = (
            "1. **Method Marks (M1):** Award marks if the correct equation/method is shown, even if the final answer is wrong.\n"
            "2. **Answer Marks (A1):** Award marks for the correct final value.\n"
            "3. **Units:** Deduct marks if units are missing/wrong."
        )
    else: # Science
        grading_criteria = (
            "1. **Keywords:** Check if the student used the specific keywords listed in the Marking Scheme.\n"
            "2. **Concept:** Is the scientific explanation correct?\n"
            "3. **Link:** Did they link the concept back to the question scenario?"
        )

    prompt = (
        f"You are a strict Primary 6 {subject} marker in Singapore.\n"
        f"**Question:** {question}\n"
        f"**Model Answer:** {model_answer}\n"
        f"**Marking Scheme:** {marking_scheme}\n"
        f"**Student Answer:** {student_answer}\n\n"
        "**Your Task:**\n"
        f"{grading_criteria}\n"
        "Evaluate the answer strictly based on the Marking Scheme.\n\n"
        "**Output Format:**\n"
        "**Estimated Score:** [e.g., 1/2 or 2/2]\n"
        "**Feedback:** [Specific comments on M1/A1 or missing keywords]\n"
        "**Improvement:** [What specifically to add]"
    )
    try:
        response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        return response.text
    except Exception as e:
        time.sleep(API_DELAY_SECONDS)
        return f"Error grading answer: {e}"

def format_prompt_for_generation(questions_series, subject, num_to_generate, specific_topic, question_type):
    # Set up Difficulty Context
    if subject == 'math':
        subject_name = "Math"
        difficulty_text = "EXTREMELY CHALLENGING (PSLE AL1 / Olympiad Standard)"
        open_ended_instruction = (
            "Generate complex word problems involving **heuristics** (e.g., Working Backwards, Assumption, Grouping).\n"
            "**MANDATORY:** The question text MUST end with: **'Write your equation clearly and find the answer.'**"
        )
        marking_instruction = "Provide a marking scheme: e.g., 'M1 for correct equation', 'A1 for final answer'."
    else:
        subject_name = "Science" 
        difficulty_text = "EXTREMELY CHALLENGING (PSLE AL1 / Application Standard)"
        open_ended_instruction = (
            "Generate questions based on **experimental setups** or **analyzing graphs**.\n"
            "The question should require explaining 'Why' or 'How' using scientific concepts."
        )
        marking_instruction = "Provide a marking scheme listing the **Essential Keywords** that MUST be present for marks."

    # --- PROMPTS ---
    if question_type == "MCQ":
        instruction = (
            f"Your task is to generate {num_to_generate} new **Multiple-Choice Questions (MCQ)**."
            "Provide 4 options (A, B, C, D) and one clear reasoning step."
        )
        output_format_example = (
            "[Reference: 0]\n"
            "Question: ...\n"
            "Difficulty: Hard\n"
            f"Topic: {specific_topic}\n"
            "A) ...\n"
            "B) ...\n"
            "C) ...\n"
            "D) ...\n"
            "Answer: (B)\n"
            "Reasoning: ...\n"
        )
    else: # Open-Ended (With Marking Scheme)
        instruction = (
            f"Your task is to generate {num_to_generate} new **Open-Ended Questions**.\n"
            f"{open_ended_instruction}\n"
            "**DO NOT PROVIDE OPTIONS.** Instead, provide a 'Model Answer' and a 'Marking Scheme'."
        )
        output_format_example = (
            "[Reference: 0]\n"
            "Question: (Complex question text here...)\n"
            "Difficulty: AL1 Hard\n"
            f"Topic: {specific_topic}\n"
            "Answer: (Full model answer)\n"
            f"Marking Scheme: ({marking_instruction})\n"
        )

    system_message = (
        f"You are an expert **{subject_name}** tutor in Singapore. "
        f"I will provide reference questions. {instruction}\n\n"
        f"Topic: **{specific_topic}**\n"
        f"Difficulty: **{difficulty_text}**\n\n"
        "**OUTPUT FORMAT (Strictly Follow):**\n"
        "Your response must be a plain-text list of blocks. Do NOT use JSON.\n"
        f"{output_format_example}\n"
        "\n[Reference: 1]\n..."
    )
    
    user_message_parts = [f"Here are the reference questions:\n\n"]
    for i, question_text in enumerate(questions_series):
        user_message_parts.append(f"{i}. {question_text}\n")
    user_message_parts.append(f"\nPlease generate {num_to_generate} new **{question_type}** questions on **{specific_topic}**.")
    
    return system_message, "".join(user_message_parts)

def call_gemini_api(system_message, user_prompt, model):
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.7)
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_message,
            generation_config=generation_config
        )
        response = gemini_model.generate_content(user_prompt)
        total_tokens = getattr(response.usage_metadata, 'total_token_count', 0)
        return response.text, total_tokens
    except Exception as e:
        st.error(f"Error communicating with Gemini API: {e}. Waiting {API_DELAY_SECONDS}s...")
        time.sleep(API_DELAY_SECONDS)
        return None, 0

def parse_generated_questions(text_blob, questions_list_so_far, question_type):
    new_questions = []
    
    if question_type == "MCQ":
        pattern = re.compile(
            r"\[Reference: \d+\].*?\n\s*"
            r"Question:\s*(.*?)\n\s*"
            r"Difficulty:\s*(.*?)\n\s*"
            r"Topic:\s*(.*?)\n\s*"
            r"A\)\s*(.*?)\n\s*"
            r"B\)\s*(.*?)\n\s*"
            r"C\)\s*(.*?)\n\s*"
            r"D\)\s*(.*?)\n\s*"
            r"Answer:\s*(.*?)\n\s*"
            r"Reasoning:\s*(.*?)"
            r"(?=\n\[Reference:|\Z)",
            re.IGNORECASE | re.DOTALL
        )
        matches = pattern.findall(text_blob)
        for match in matches:
            q, diff, topic, a, b, c, d, ans, reason = match
            new_questions.append({
                "type": "MCQ",
                "question": q.strip(),
                "difficulty": diff.strip(),
                "topic": topic.strip(),
                "options": {"A": a.strip(), "B": b.strip(), "C": c.strip(), "D": d.strip()},
                "answer": ans.strip(),
                "reasoning": reason.strip()
            })
    else:
        # Open-Ended Parser (Now extracts Marking Scheme)
        pattern = re.compile(
            r"\[Reference: \d+\].*?\n\s*"
            r"Question:\s*(.*?)\n\s*"
            r"Difficulty:\s*(.*?)\n\s*"
            r"Topic:\s*(.*?)\n\s*"
            r"Answer:\s*(.*?)\n\s*"
            r"Marking Scheme:\s*(.*?)"
            r"(?=\n\[Reference:|\Z)",
            re.IGNORECASE | re.DOTALL
        )
        matches = pattern.findall(text_blob)
        for match in matches:
            q, diff, topic, ans, marking = match
            new_questions.append({
                "type": "Open-Ended",
                "question": q.strip(),
                "difficulty": diff.strip(),
                "topic": topic.strip(),
                "answer": ans.strip(),
                "marking_scheme": marking.strip() 
            })
            
    return new_questions

def process_generation_loop(file_path, subject_lower, num_to_generate, specific_topic, question_type):
    final_generated_list = []
    retries_used = 0
    questions_series = load_and_select_questions(file_path, QUESTIONS_TO_SELECT, QUESTION_COLUMN_NAME)
    if questions_series is None:
        return []

    while len(final_generated_list) < num_to_generate and retries_used < MAX_RETRIES:
        questions_needed = num_to_generate - len(final_generated_list)
        reference_subset = questions_series.sample(min(QUESTIONS_TO_SELECT, questions_needed * 5, len(questions_series))) 
        
        st.info(f"Attempt {retries_used + 1}/{MAX_RETRIES}: Generating {questions_needed} ({question_type}) questions...")
        time.sleep(API_DELAY_SECONDS) 
        
        system_msg, user_prompt = format_prompt_for_generation(reference_subset, subject_lower, questions_needed, specific_topic, question_type)
        generation_text, tokens_used = call_gemini_api(system_msg, user_prompt, GEMINI_MODEL)
        st.session_state.total_tokens_used += tokens_used
        
        if generation_text:
            newly_parsed = parse_generated_questions(generation_text, st.session_state.all_generated_questions, question_type)
            if newly_parsed:
                final_generated_list.extend(newly_parsed)
                st.session_state.all_generated_questions.extend(newly_parsed) 
                st.success(f"Added {len(newly_parsed)} questions.")
            else:
                st.warning("Parser failed (AI format error). Retrying...")
        else:
            st.error("Generation failed. Stopping.")
            break
        retries_used += 1

    return final_generated_list

# ==========================================
# 4. UI & INTERACTION
# ==========================================

def next_q():
    st.session_state.current_index += 1
    st.session_state.answer_checked = False
    st.session_state.grading_feedback = None 

def prev_q():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.session_state.answer_checked = False
        st.session_state.grading_feedback = None

def check_answer_handler():
    st.session_state.answer_checked = True

def grade_answer_handler(question_text, correct_ans, marking_scheme, user_ans, subject):
    with st.spinner("👩‍🏫 The AI Teacher is marking your answer based on the marking scheme..."):
        feedback = grade_student_answer(question_text, correct_ans, marking_scheme, user_ans, subject)
        st.session_state.grading_feedback = feedback
        st.session_state.answer_checked = True

def display_question_session(subject_name):
    generated_list = st.session_state.latest_generated_list
    total_count = len(generated_list)
    
    if st.session_state.current_index >= total_count:
        st.session_state.current_index = 0

    current_q_index = st.session_state.current_index
    item = generated_list[current_q_index]
    
    # Header
    st.header(f"Question {current_q_index + 1} of {total_count}")
    st.caption(f"Topic: {item.get('topic', 'N/A')} | Mode: {item.get('type', 'N/A')} | Difficulty: {item.get('difficulty', 'Hard')}")
    st.write("---")
    
    # Display Question
    st.code(item.get('question', 'N/A'), language='text')

    # --- MODE 1: MCQ ---
    if item.get('type') == 'MCQ':
        options = item.get('options', {})
        radio_options = [
            f"A) {options.get('A', 'N/A')}",
            f"B) {options.get('B', 'N/A')}",
            f"C) {options.get('C', 'N/A')}",
            f"D) {options.get('D', 'N/A')}"
        ]
        selected_option = st.radio("Select your answer:", radio_options, index=None, key=f"radio_q{current_q_index}")
        
        st.write("")
        col1, col2, col3 = st.columns([1, 2, 1])
        if col2.button("Check Answer", on_click=check_answer_handler, key="btn_check"): pass
        
        if st.session_state.answer_checked:
            st.write("---")
            if selected_option:
                user_letter = selected_option[0]
                correct_letter = item.get('answer', '?')[0]
                if user_letter == correct_letter:
                    st.success(f"✅ Correct! Answer: {correct_letter}")
                else:
                    st.error(f"❌ Incorrect. You picked {user_letter}. Correct: {correct_letter}")
                st.info(f"**Reasoning:** {item.get('reasoning')}")

    # --- MODE 2: OPEN-ENDED (With Marking Scheme) ---
    else:
        placeholder_text = "Type your equation/working..." if subject_name == "Math" else "Type your explanation..."
        user_text = st.text_area("Your Answer:", placeholder=placeholder_text, height=150, key=f"text_q{current_q_index}")
        
        st.write("")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        if col2.button("Submit & Grade", key="btn_grade"):
            if user_text:
                grade_answer_handler(item['question'], item['answer'], item.get('marking_scheme', ''), user_text, subject_name)
            else:
                st.warning("Please type an answer first.")

        if st.session_state.answer_checked and st.session_state.grading_feedback:
            st.write("---")
            st.markdown("### 📝 Teacher's Feedback")
            st.markdown(st.session_state.grading_feedback)
            
            with st.expander("🔑 View Model Answer & Marking Scheme"):
                st.info(f"**Model Answer:**\n{item.get('answer')}")
                st.warning(f"**Marking Scheme:**\n{item.get('marking_scheme')}")

    # Navigation
    st.write("---")
    c1, c2, c3 = st.columns([1, 3, 1])
    if current_q_index > 0: c1.button("⬅️ Prev", on_click=prev_q)
    if current_q_index < total_count - 1: c3.button("Next ➡️", on_click=next_q)

# ==========================================
# 5. MAIN
# ==========================================

def main():
    st.title("📚 AI Question Generator")
    initialize_session_state()

    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input("Enter Google API Key:", type="password")

    # 1. Subject
    subject_name = st.sidebar.selectbox("1. Choose Subject:", ["Math", "Science"])

    # 2. Topic
    selected_topic = "General"
    if subject_name == "Math":
        selected_topic = st.sidebar.selectbox("   Select Topic:", ["Fractions", "Ratio", "Percentage", "Algebra", "Geometry", "Speed", "Volume"])
    elif subject_name == "Science":
        selected_topic = st.sidebar.selectbox("   Select Topic:", ["Diversity", "Cycles", "Systems", "Interactions", "Energy"])

    # 3. Question Type
    question_type = st.sidebar.radio("2. Question Type:", ["MCQ", "Open-Ended"])
    
    num_to_generate = st.sidebar.number_input("3. How many questions?", min_value=1, max_value=10, value=3)

    if st.sidebar.button("🚀 Generate"):
        st.session_state.total_tokens_used = 0 
        st.session_state.latest_generated_list = []
        st.session_state.current_index = 0
        st.session_state.answer_checked = False
        st.session_state.grading_feedback = None
        
        if not api_key:
            st.error("Missing API Key")
        else:
            genai.configure(api_key=api_key)
            with st.spinner("Generating..."):
                file_path = MATH_EXCEL_FILE_PATH if subject_name == 'Math' else SCIENCE_EXCEL_FILE_PATH
                subject_lower = subject_name.lower()
                
                generated_list = process_generation_loop(file_path, subject_lower, num_to_generate, selected_topic, question_type)
                
                if generated_list:
                    st.session_state.latest_generated_list = generated_list
                    st.rerun() 
    
    if st.session_state.latest_generated_list:
        display_question_session(subject_name)

if __name__ == "__main__":
    main()
