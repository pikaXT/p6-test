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

GEMINI_MODEL = 'gemini-2.5-flash' 
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
        "user_selections": {}   
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

def format_prompt_for_comprehension(subject, num_to_generate):
    language = "English"
    if subject == 'chinese':
        language = "Chinese (简体中文)"

    system_message = (
        f"You are a strict, high-standard tutor in Singapore creating {language} comprehension passages for a Primary 6 student (12 years old)."
        f"Your task is to generate a sophisticated short story and **{num_to_generate} highly challenging comprehension questions**."
        "\n\n"
        "**STORY RULES (PSLE STANDARD):**"
        f"1.  The story must be in **{language}**."
        "2.  It should be about **3 paragraphs** long. Vocabulary should be advanced (AL1 standard)."
        "3.  Themes should be mature: **moral dilemmas, regret, sacrifice, or subtle emotional conflicts**."
        "\n\n"
        "**QUESTION RULES (CRITICAL THINKING ONLY):**"
        f"1.  Generate exactly **{num_to_generate} questions**."
        "2.  **DIFFICULTY:** Questions must be 'Inferential' or 'Critical Analysis'. **NO direct look-and-find questions.**"
        "3.  Students must be required to 'read between the lines' to find the answer."
        f"4.  Questions must be in **{language}**."
        "\n\n"
        "**VERIFICATION STEP:**"
        "Before writing the Answer, you must verify that the answer is NOT explicitly written in the text, but logically deduced from it."
        "\n\n"
        "**OUTPUT FORMAT:**"
        "You MUST follow this exact plain-text format:"
        "\n"
        "[Story]"
        "(Your advanced story in {language} goes here...)"
        "\n\n"
        "[Questions]"
        f"1. (Your first {language} question...)\n"
        "...\n"
        f"{num_to_generate}. (Your last {language} question...)\n"
        "\n"
        "[Answers]"
        f"1. (The answer...)\n"
        "...\n"
        f"{num_to_generate}. (The answer...)\n"
    )
    
    user_prompt = f"Please generate one (1) P6 {language} comprehension passage, {num_to_generate} **highly challenging** questions, and the corresponding answers. Follow the format strictly."
    
    return system_message, user_prompt

def format_prompt_for_generation(questions_series, subject, num_to_generate, specific_topic="General"):
    if subject == 'math':
        subject_name = "Math"
        difficulty_text = "EXTREMELY CHALLENGING (PSLE AL1 / A* Standard)"
        difficulty_detail = "require **heuristics, working backwards, or multi-step logic**"
        # Updated instructions for Math verification
        reasoning_text = "Step 1: State the heuristic used. Step 2: Show the calculation. Step 3: Verify why the Distractors are wrong (common mistakes)."
        topic_examples = "Fractions, Algebra, Ratios, Speed, Volume"
    else:
        subject_name = "Science" 
        difficulty_text = "EXTREMELY CHALLENGING (PSLE AL1 / A* Standard)"
        difficulty_detail = "require **application of concepts in unfamiliar scenarios (experimental setups)**"
        # Updated instructions for Science verification
        reasoning_text = "Step 1: Identify the scientific concept. Step 2: Explain the link to the scenario. Step 3: Explain why the other options are plausible misconceptions but incorrect."
        topic_examples = "Energy, Life Cycles, Matter, Forces"

    system_message = (
        f"You are a strict **{subject_name}** setter for the PSLE exams in Singapore. Your goal is to test the top 10% of students."
        "\n\n"
        f"I will provide a list of reference **{subject_name}** questions."
        f"Your task is to generate {num_to_generate} new **High-Difficulty MCQs** specifically about: **{specific_topic}**."
        "\n\n"
        "**DIFFICULTY REQUIREMENT (AL1 STANDARD):**"
        f"The new questions must be **{difficulty_text}**. They must {difficulty_detail}."
        "The questions should not be straightforward. They should contain 'traps' or 'distractors' that look correct if the student misses a small detail."
        "\n\n"
        "**MANDATORY VERIFICATION (Double-Check):**"
        "For every question, before you finalize the output, you must internally calculate the answer to ensure it is 100% correct. **If the logic is weak, discard it and try again.**"
        "\n\n"
        "**CRITICAL OUTPUT CONSTRAINTS:**"
        "1.  **Word-Wrap:** Manually insert `\\n` every 60 characters for Question, Options, and Reasoning."
        "2.  **Options:** Generate 4 options (A, B, C, D). **One is correct. Three are 'Distractors' based on common student errors.**"
        "\n\n"
        "**OUTPUT FORMAT:**"
        "Your response MUST follow this exact plain-text format for each question. Do NOT use JSON."
        "\n\n"
        "[Reference: 0]\n"
        "Question: (Your complex question text...)\n"
        "Difficulty: Hard\n"
        f"Topic: {specific_topic}\n"
        "A) (Distractor 1)\n"
        "B) (Distractor 2)\n"
        "C) (Distractor 3)\n"
        "D) (Correct Answer)\n"
        "Answer: (D)\n"
        f"Reasoning: ({reasoning_text})\n"
        "\n"
        "[Reference: 1]\n"
        "Question: ...\n"
        "(and so on...)\n"
    )
    
    user_message_parts = [f"Here are the reference **{subject_name}** questions (indexed 0-{len(questions_series)-1}):\n\n"]
    for i, question_text in enumerate(questions_series):
        user_message_parts.append(f"{i}. {question_text}\n")
    user_message_parts.append(f"\nPlease generate {num_to_generate} new, **{difficulty_text}** {subject_name} MCQs on **{specific_topic}**. Ensure you verify the answer key is correct.")
    
    return system_message, "".join(user_message_parts)

def call_gemini_api(system_message, user_prompt, model, task_name="task", subject=""):
    # Lower temp for Math/Science to ensure logic is strict
    if subject in ['math', 'science']:
        temp = 0.3 
    else:
        temp = 0.7
        
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=temp
        )
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

def parse_comprehension_response(text_blob):
    story_match = re.search(r"\[Story\](.*?)\[Questions\]", text_blob, re.IGNORECASE | re.DOTALL)
    questions_match = re.search(r"\[Questions\](.*?)\[Answers\]", text_blob, re.IGNORECASE | re.DOTALL)
    answers_match = re.search(r"\[Answers\](.*?)$", text_blob, re.IGNORECASE | re.DOTALL)
    
    if not story_match or not questions_match or not answers_match:
        return None, None, None

    story = story_match.group(1).strip()
    questions = questions_match.group(1).strip()
    answers = answers_match.group(1).strip()
    return story, questions, answers

def parse_generated_questions(text_blob, questions_list_so_far):
    new_questions = []
    unique_content_hashes = set()
    for q in questions_list_so_far:
        content_key = f"{q['question']}|{q['options']['A']}|{q['options']['B']}|{q['options']['C']}|{q['options']['D']}|{q['reasoning']}"
        unique_content_hashes.add(content_key)
    
    pattern = re.compile(
        r"\[Reference: (\d+)\].*?\n\s*"  
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
    
    if not matches:
        st.warning("Parser Warning: The AI response did not contain any correctly formatted question blocks. Will attempt retry.")
        return []

    for i, match in enumerate(matches):
        ref_index, question, difficulty, topic, opt_a, opt_b, opt_c, opt_d, answer, reasoning = match
        
        step_count = reasoning.count('\n') + 1 
        if step_count > 20:
            st.warning(f"⚠️ Parser Skipped: Question #{i+1} has excessive reasoning ({step_count} steps). Skipping this question.")
            continue
        
        content_key = f"{question.strip()}|{opt_a.strip()}|{opt_b.strip()}|{opt_c.strip()}|{opt_d.strip()}|{reasoning.strip()}"
        
        if content_key in unique_content_hashes:
            st.warning(f"⚠️ Parser Skipped: Detected a repeating question block for question #{i+1}. Skipping this question.")
            continue
            
        unique_content_hashes.add(content_key)
        
        question_obj = {
            "question": question.strip(),
            "difficulty": difficulty.strip(),
            "topic": topic.strip(),
            "options": {"A": opt_a.strip(), "B": opt_b.strip(), "C": opt_c.strip(), "D": opt_d.strip()},
            "reference_index": ref_index.strip(),
            "answer": answer.strip(), 
            "reasoning": reasoning.strip()
        }
        new_questions.append(question_obj)
        
    return new_questions

def process_generation_loop(file_path, subject_lower, num_to_generate, specific_topic):
    final_generated_list = []
    retries_used = 0
    questions_series = load_and_select_questions(file_path, QUESTIONS_TO_SELECT, QUESTION_COLUMN_NAME)
    if questions_series is None:
        return []

    while len(final_generated_list) < num_to_generate and retries_used < MAX_RETRIES:
        questions_needed = num_to_generate - len(final_generated_list)
        reference_subset = questions_series.sample(min(QUESTIONS_TO_SELECT, questions_needed * 5, len(questions_series))) 
        
        st.info(f"Attempt {retries_used + 1}/{MAX_RETRIES}: Generating {questions_needed} more questions on '{specific_topic}'...")
        
        time.sleep(5) 
        
        system_msg, user_prompt = format_prompt_for_generation(reference_subset, subject_lower, questions_needed, specific_topic)
        generation_text, tokens_used = call_gemini_api(system_msg, user_prompt, GEMINI_MODEL, f"{subject_lower} MCQ Task", subject_lower)
        st.session_state.total_tokens_used += tokens_used
        
        if generation_text:
            newly_parsed = parse_generated_questions(generation_text, st.session_state.all_generated_questions) 
            if newly_parsed:
                final_generated_list.extend(newly_parsed)
                st.session_state.all_generated_questions.extend(newly_parsed) 
                st.success(f"Successfully added {len(newly_parsed)} questions. Total collected: {len(final_generated_list)}/{num_to_generate}.")
            else:
                st.warning("No new questions were successfully parsed in this attempt. Retrying...")
        else:
            st.error("API call failed during generation attempt. Stopping retry loop.")
            break
        retries_used += 1

    if len(final_generated_list) < num_to_generate:
        st.warning(f"Finished attempts. Could only generate {len(final_generated_list)} out of {num_to_generate} questions.")

    return final_generated_list


# --- HELPER FUNCTIONS FOR INTERACTIVE DISPLAY ---

def next_q():
    st.session_state.current_index += 1
    st.session_state.answer_checked = False 

def prev_q():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.session_state.answer_checked = False 

def check_answer_handler():
    st.session_state.answer_checked = True

def display_mcq_session():
    """Renders interactive MCQ questions one-by-one."""
    generated_list = st.session_state.latest_generated_list
    total_count = len(generated_list)
    
    if st.session_state.current_index >= total_count:
        st.session_state.current_index = 0

    current_q_index = st.session_state.current_index
    item = generated_list[current_q_index]
    options = item.get('options', {})

    # -- Header --
    st.header(f"Question {current_q_index + 1} of {total_count}")
    st.caption(f"Topic: {item.get('topic', 'N/A')} | Difficulty: {item.get('difficulty', 'N/A')}")
    st.write("---")
    
    # -- Display Question Text --
    st.code(item.get('question', 'N/A'), language='text')

    # -- Interactive Radio Options --
    radio_options = [
        f"A) {options.get('A', 'N/A')}",
        f"B) {options.get('B', 'N/A')}",
        f"C) {options.get('C', 'N/A')}",
        f"D) {options.get('D', 'N/A')}"
    ]
    
    selected_option = st.radio(
        "Select your answer:", 
        radio_options, 
        index=None, 
        key=f"radio_q{current_q_index}"
    )

    st.write("") # Spacer

    # -- Controls --
    col1, col2, col3, col4 = st.columns([1, 2, 1, 3])
    
    # Check Answer Button
    if col2.button("Check Answer", on_click=check_answer_handler, key="btn_check"):
        pass 
    
    # Navigation Buttons
    if current_q_index > 0:
        col1.button("⬅️ Prev", on_click=prev_q, key="btn_prev")
    
    if current_q_index < total_count - 1:
        col3.button("Next ➡️", on_click=next_q, key="btn_next")
    elif current_q_index == total_count - 1:
        col3.markdown("**End of Quiz!**")

    # -- Feedback Section --
    if st.session_state.answer_checked:
        st.write("---")
        if selected_option:
            user_letter = selected_option[0] 
            
            correct_answer_raw = item.get('answer', 'N/A')
            correct_letter_match = re.search(r"[ABCD]", correct_answer_raw)
            correct_letter = correct_letter_match.group(0) if correct_letter_match else "?"

            if user_letter == correct_letter:
                st.success(f"✅ **Correct!** The answer is **{correct_letter}**.")
            else:
                st.error(f"❌ **Incorrect.** You selected **{user_letter}**, but the correct answer is **{correct_letter}**.")
            
            st.markdown("**Reasoning (Verified):**")
            st.code(item.get('reasoning', 'N/A'), language='text')
        else:
            st.warning("Please select an option before checking.")

# --- Main Streamlit App Function ---

def main():
    st.title("📚 AI Question Generator")
    
    initialize_session_state()

    st.sidebar.header("⚙️ Configuration")

    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")

    # 1. Main Subject Selection
    subject_name = st.sidebar.selectbox(
        "1. Choose a subject:",
        ["English", "Chinese", "Math", "Science"]
    )

    # 2. Logic for Topics Dropdown
    selected_topic = "General" # Default

    if subject_name == "Math":
        selected_topic = st.sidebar.selectbox(
            "   Select a Math Topic:",
            ["Whole Numbers", "Fractions", "Ratio", "Percentage", "Algebra", "Geometry (Angles)", "Circles", "Speed", "Volume", "Pie Charts"]
        )
    elif subject_name == "Science":
        selected_topic = st.sidebar.selectbox(
            "   Select a Science Topic:",
            ["Diversity (Living Things)", "Cycles (Water/Life)", "Systems (Body/Plant)", "Interactions (Forces/Environment)", "Energy (Light/Heat/Electricity)"]
        )

    if subject_name in ["English", "Chinese"]:
        num_to_generate = st.sidebar.number_input(
            "2. How many comprehension questions?", 
            min_value=1, max_value=20, value=10
        )
    else: 
        num_to_generate = st.sidebar.number_input(
            "2. How many MCQs to generate?", 
            min_value=1, max_value=20, value=5
        )

    generate_button = st.sidebar.button("🚀 Generate Questions")
    
    st.markdown("---")
    
    generation_status_placeholder = st.empty()

    if generate_button:
        st.session_state.total_tokens_used = 0 
        st.session_state.latest_generated_list = []
        st.session_state.current_index = 0
        st.session_state.answer_checked = False
        
        if not api_key:
            st.error("Please enter your Google API Key in the sidebar.")
        else:
            try:
                genai.configure(api_key=api_key)
                generation_status_placeholder.success(f"Generating {subject_name} ({selected_topic}) questions...")
                
                with st.spinner("Generating questions... This may take a moment."):
                    
                    subject_lower = subject_name.lower()
                    
                    if subject_lower in ['english', 'chinese']:
                        # --- COMPREHENSION ---
                        system_msg, user_prompt = format_prompt_for_comprehension(subject_lower, num_to_generate)
                        generation_text, tokens_used = call_gemini_api(system_msg, user_prompt, GEMINI_MODEL, "Comprehension Task", subject_lower)
                        st.session_state.total_tokens_used += tokens_used

                        if generation_text:
                            story, questions, answers = parse_comprehension_response(generation_text)
                            if story:
                                st.header(f"✨ Generated {subject_name} Comprehension Passage")
                                with st.expander("📖 Story (Click to view)", expanded=True):
                                    st.markdown(story)
                                with st.expander(f"❓ Questions ({num_to_generate})", expanded=True):
                                    st.code(questions, language='text')
                                with st.expander("✅ Answers"):
                                    st.code(answers, language='text') 
                            else:
                                st.error("The AI returned an invalid format. Please try again.")
                                st.subheader("Raw AI Response:")
                                st.text(generation_text)
                        else:
                            st.error("Failed to get a response from the AI.")

                    # --- MCQ ---
                    elif subject_lower in ['math', 'science']:
                        
                        if subject_lower == 'math':
                            file_path = MATH_EXCEL_FILE_PATH
                        else: # Science
                            file_path = SCIENCE_EXCEL_FILE_PATH
                            
                        # Pass the 'selected_topic' to the loop
                        generated_list = process_generation_loop(file_path, subject_lower, num_to_generate, selected_topic)
                        
                        if generated_list:
                            st.session_state.latest_generated_list = generated_list
                            st.subheader(f"✅ Generation Complete! Starting Interactive Review ({len(generated_list)}/{num_to_generate})")
                            st.rerun() 
                        elif not generated_list:
                             st.error("Failed to generate any valid questions after all retry attempts.")
                            
            except Exception as e:
                st.error(f"An unexpected error occurred during the process: {e}")
    
    if st.session_state.latest_generated_list:
        st.markdown("---")
        display_mcq_session()
        st.markdown("---")

    st.subheader("💰 Token Usage Summary")
    st.info(f"Total tokens consumed for this session: **{st.session_state.total_tokens_used:,}** tokens.")

if __name__ == "__main__":
    main()

