import pandas as pd
import google.generativeai as genai
import sys
import json
import re
import os
import time 
import streamlit as st 

st.set_page_config(page_title="AI Question Generator", layout="wide")

# --- Configuration ---
MATH_EXCEL_FILE_PATH = 'Math.xlsx'
SCIENCE_EXCEL_FILE_PATH = 'Science.xlsx'

GEMINI_MODEL = 'gemini-1.5-flash' 
QUESTION_COLUMN_NAME = 'Question Text'
QUESTIONS_TO_SELECT = 50 
MAX_RETRIES = 5 

# --- Session State Initialization ---
def initialize_session_state():
    default_values = {
        "total_tokens_used": 0,
        "all_generated_questions": [],
        "latest_generated_list": [],
        "current_index": 0,
        "answer_checked": False, 
        "grading_feedback": None, # Stores the AI's feedback for the current question
        "question_type": "MCQ"    # Tracks if we are in MCQ or Open-Ended mode
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Core Logic Functions ---

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

# --- NEW: Helper to Grade Answers via API ---
def grade_student_answer(question, model_answer, student_answer):
    """Sends the student's answer to Gemini to be graded."""
    prompt = (
        "You are a strict Primary School Teacher in Singapore. Grade the student's answer.\n"
        f"**Question:** {question}\n"
        f"**Correct Model Answer:** {model_answer}\n"
        f"**Student Answer:** {student_answer}\n\n"
        "**Task:**\n"
        "1. Determine if the student is Correct, Partially Correct, or Incorrect.\n"
        "2. Identify any missing keywords or concepts.\n"
        "3. Provide brief, encouraging feedback on how to improve.\n\n"
        "**Output Format:**\n"
        "**Status:** [Correct/Partial/Incorrect]\n"
        "**Feedback:** [Your feedback here]"
    )
    try:
        response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error grading answer: {e}"

def format_prompt_for_generation(questions_series, subject, num_to_generate, specific_topic, question_type):
    # Set up Difficulty Context
    if subject == 'math':
        subject_name = "Math"
        difficulty_text = "Challenging (PSLE Standard)"
    else:
        subject_name = "Science" 
        difficulty_text = "Challenging (PSLE Standard)"

    # --- DIFFERENT PROMPTS FOR MCQ vs OPEN-ENDED ---
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
    else: # Open-Ended
        instruction = (
            f"Your task is to generate {num_to_generate} new **Open-Ended Questions (Structured)**."
            "**DO NOT PROVIDE OPTIONS.** Instead, provide a comprehensive 'Model Answer' that includes the key marking points/keywords required."
        )
        output_format_example = (
            "[Reference: 0]\n"
            "Question: ...\n"
            "Difficulty: Hard\n"
            f"Topic: {specific_topic}\n"
            "Answer: (The full model answer with keywords)\n"
        )

    system_message = (
        f"You are an expert **{subject_name}** tutor in Singapore. "
        f"I will provide reference questions. {instruction}\n\n"
        f"Topic: **{specific_topic}**\n"
        f"Difficulty: **{difficulty_text}**\n\n"
        "**OUTPUT FORMAT (Strictly Follow):**\n"
        "Your response must be a plain-text list of blocks. Do not use JSON.\n"
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
        st.error(f"Error communicating with Gemini API: {e}")
        return None, 0

def parse_generated_questions(text_blob, questions_list_so_far, question_type):
    new_questions = []
    
    if question_type == "MCQ":
        # MCQ Parser
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
        # Open-Ended Parser
        pattern = re.compile(
            r"\[Reference: \d+\].*?\n\s*"
            r"Question:\s*(.*?)\n\s*"
            r"Difficulty:\s*(.*?)\n\s*"
            r"Topic:\s*(.*?)\n\s*"
            r"Answer:\s*(.*?)"
            r"(?=\n\[Reference:|\Z)",
            re.IGNORECASE | re.DOTALL
        )
        matches = pattern.findall(text_blob)
        for match in matches:
            q, diff, topic, ans = match
            new_questions.append({
                "type": "Open-Ended",
                "question": q.strip(),
                "difficulty": diff.strip(),
                "topic": topic.strip(),
                "answer": ans.strip() # This is the model answer
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
        time.sleep(3) 
        
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
                st.warning("Parser failed. Retrying...")
        else:
            break
        retries_used += 1

    return final_generated_list


# --- INTERACTIVE DISPLAY ---

def next_q():
    st.session_state.current_index += 1
    st.session_state.answer_checked = False
    st.session_state.grading_feedback = None # Reset feedback

def prev_q():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.session_state.answer_checked = False
        st.session_state.grading_feedback = None

def check_answer_handler():
    st.session_state.answer_checked = True

def grade_answer_handler(question_text, correct_ans, user_ans):
    # This function triggers the API grading
    with st.spinner("👩‍🏫 The AI Teacher is marking your answer..."):
        feedback = grade_student_answer(question_text, correct_ans, user_ans)
        st.session_state.grading_feedback = feedback
        st.session_state.answer_checked = True

def display_question_session():
    generated_list = st.session_state.latest_generated_list
    total_count = len(generated_list)
    
    if st.session_state.current_index >= total_count:
        st.session_state.current_index = 0

    current_q_index = st.session_state.current_index
    item = generated_list[current_q_index]
    
    # Header
    st.header(f"Question {current_q_index + 1} of {total_count}")
    st.caption(f"Topic: {item.get('topic', 'N/A')} | Mode: {item.get('type', 'N/A')}")
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

    # --- MODE 2: OPEN-ENDED (Text Input + AI Grading) ---
    else:
        user_text = st.text_area("Type your answer here:", height=150, key=f"text_q{current_q_index}")
        
        st.write("")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # Grading Button
        if col2.button("Submit & Grade", key="btn_grade"):
            if user_text:
                grade_answer_handler(item['question'], item['answer'], user_text)
            else:
                st.warning("Please type an answer first.")

        # Show Feedback
        if st.session_state.answer_checked and st.session_state.grading_feedback:
            st.write("---")
            st.markdown("### 📝 Teacher's Feedback")
            st.markdown(st.session_state.grading_feedback)
            with st.expander("View Model Answer"):
                st.info(item.get('answer'))

    # Navigation
    st.write("---")
    c1, c2, c3 = st.columns([1, 3, 1])
    if current_q_index > 0: c1.button("⬅️ Prev", on_click=prev_q)
    if current_q_index < total_count - 1: c3.button("Next ➡️", on_click=next_q)

# --- Main Streamlit App Function ---

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

    # 3. Question Type (The New Feature!)
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
        display_question_session()

if __name__ == "__main__":
    main()
