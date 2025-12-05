import pandas as pd
import google.generativeai as genai
import sys
import json
import re
import os # For API Key

# --- Configuration ---
# These file paths MUST be correct on the computer running the app
MATH_EXCEL_FILE_PATH = r'C:\HJ-Local\XT_TestData\Can you do it for these pdf also and place them i....xlsx'
SCIENCE_EXCEL_FILE_PATH = r'C:\Users\xtxzx\Downloads\Can you just make it so that i can preview.xlsx'

GEMINI_MODEL = 'gemini-2.0-flash-lite'
QUESTION_COLUMN_NAME = 'Question Text'
QUESTIONS_TO_SELECT = 50 # Number of questions to read from Excel for reference
MAX_RETRIES = 5 # Maximum times to try regenerating missing questions
# --- End Configuration ---

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes all session state variables safely at the start."""
    default_values = {
        "total_tokens_used": 0,
        "all_generated_questions": [],
        "latest_generated_list": [],
        "current_index": 0,
        "answer_checked": False, # New: Tracks if user clicked 'Check Answer'
        "user_selections": {}    # New: Stores user answers for persistence
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Core Logic Functions (Generation/Parsing Logic Unchanged) ---

def load_and_select_questions(file_path, num_questions, column_name):
    st.info(f"Loading reference questions from '{file_path}'...")
    try:
        df = pd.read_excel(file_path) 
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please check the file path in the script.")
        return None
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
        f"You are an expert tutor in Singapore creating {language} comprehension passages for a Primary 6 student (11-12 years old)."
        f"Your task is to generate a short story, **{num_to_generate} comprehension questions**, and the **answers** to those questions."
        "\n\n"
        "**STORY RULES:**"
        f"1.  The story must be in **{language}**."
        "2.  It should be about **3 very short paragraphs** long, suitable for a 12-year-old, focusing on themes like **moral dilemmas, complex emotions, or subtle conflicts**."
        "3.  The story must be engaging and contain themes they can understand."
        "\n\n"
        "**QUESTION RULES: (CRITICAL - PSLE HIGH DIFFICULTY)**"
        f"1.  You must generate exactly **{num_to_generate} questions** based on the story."
        "2.  The questions must be **highly challenging**, suitable for a **Primary 6 examination (PSLE level)**."
        "3.  Questions must focus on advanced comprehension skills such as **author's intent, tone, mood, figurative language meaning, implied motives, and predicting outcomes based on subtle textual evidence.**"
        "4.  **AVOID** any direct-recall questions."
        f"5.  Questions must be in **{language}**."
        "\n\n"
        "**ANSWER RULES:**"
        f"1.  You must provide clear, simple answers for **all {num_to_generate} questions**."
        f"2.  The answers must be concise, **no longer than two short sentences each**, to ensure they fit neatly on a display without scrolling."
        f"3.  Answers must be in **{language}**."
        "\n\n"
        "**OUTPUT FORMAT:**"
        "You MUST follow this exact plain-text format:"
        "\n"
        "[Story]"
        "(Your 3-paragraph story in {language} goes here...)"
        "\n\n"
        "[Questions]"
        f"1. (Your first {language} question...)\n"
        "...\n"
        f"{num_to_generate}. (Your last {language} question...)\n"
        "\n"
        "[Answers]"
        f"1. (The answer to question 1...)\n"
        "...\n"
        f"{num_to_generate}. (The answer to question {num_to_generate}...)\n"
    )
    
    user_prompt = f"Please generate one (1) P6 {language} comprehension passage, {num_to_generate} **highly challenging PSLE-level** questions, and the corresponding answers. Follow the format strictly."
    
    return system_message, user_prompt

def format_prompt_for_generation(questions_series, subject, num_to_generate):
    if subject == 'math':
        subject_name = "Math"
        difficulty_text = "moderately harder"
        difficulty_detail = "require **one or two extra steps**"
        reasoning_text = "Provide the **full arithmetic solution** using a concise, numbered sequence of calculations. **DO NOT use algebra**. The reasoning must be **manually word-wrapped** by inserting a **newline character (\\n)** at the nearest word break so that **no line exceeds 60 characters**."
        topic_examples = "Fractions, Algebra, Ratios"
    else:
        subject_name = "Science"
        difficulty_text = "moderately harder"
        difficulty_detail = "require **one or two extra steps**"
        reasoning_text = "Provide 2-3 key scientific facts or principles necessary to reach the conclusion. The reasoning must be **manually word-wrapped** by inserting a **newline character (\\n)** at the nearest word break so that **no line exceeds 60 characters**."
        topic_examples = "Energy, Life Cycles, Matter"

    system_message = (
        f"You are an expert **{subject_name}** tutor in Singapore. Your task is to help a Primary 6 student (11-12 years old)."
        "\n\n"
        f"I will provide a list of reference **{subject_name}** questions (indexed 0, 1, 2, etc.)."
        f"Your task is to generate {num_to_generate} new **multiple-choice {subject_name} questions (MCQs)**."
        "\n\n"
        "**DIFFICULTY REQUIREMENT:**"
        f"The new questions must be **{difficulty_text}** than the reference questions. They should test the same core concept but {difficulty_detail}, while still being solvable by a P6 student."
        "\n\n"
        "**CRITICAL OUTPUT CONSTRAINTS FOR DISPLAY (MANDATORY):**"
        "To ensure the questions fit neatly on a computer screen without horizontal scrolling, the following rules must be strictly adhered to:"
        "1.  The main **Question** text itself must be **manually word-wrapped** by inserting a **newline character (\\n)** at the nearest word break so that **no line exceeds 60 characters**."
        "2.  All **Options (A, B, C, D)** must also be **manually word-wrapped** by inserting a **newline character (\\n)** at the nearest word break so that **no line exceeds 60 characters**."
        "3.  The **Reasoning** must follow the same rule: **manually word-wrap** by inserting a **newline character (\\n)** at the nearest word break so that **no line exceeds 60 characters**."
        "\n\n"
        "**INTERNAL REVIEW PROCESS (MANDATORY):**"
        "For each new question you create, you must **first think step-by-step**:"
        f"1.  **Look at the reference question's topic** (e.g., {topic_examples})."
        "2.  **Think of a new, similar scenario** that is different from the reference."
        f"3.  **Apply Difficulty:** How can I make this new scenario **{difficulty_text}**?"
        "4.  **Identify Fields:** What is the specific `Topic`? What is the `Difficulty`? What is the `Reasoning` for the correct answer (must be a numbered breakdown/fact list)?"
        "5.  **Check for Variety:** Is this new question too similar to one I've already made? If yes, pick a different reference question."
        "\n\n"
        "**OUTPUT FORMAT:**"
        "Your response MUST follow this exact plain-text format for each question. Do NOT use JSON."
        "\n\n"
        "**CRITICAL:** You MUST NOT include any of your internal thoughts, explanations, or any text outside of the final, formatted question blocks. Your response must begin *immediately* with `[Reference: ...]` and contain *only* the {num_to_generate} question blocks."
        "\n\n"
        "[Reference: 0]\n"
        "Question: Your new question text...\n"
        "Difficulty: (e.g., Medium, Hard)\n"
        f"Topic: (e.g., {topic_examples})\n"
        "A) Option A\n"
        "B) Option B\n"
        "C) Option C\n"
        "D) Option D\n"
        "Answer: (B)\n"
        f"Reasoning: ({reasoning_text})\n"
        "\n"
        "[Reference: 1]\n"
        "Question: ...\n"
        "(and so on...)\n"
    )
    
    user_message_parts = [f"Here are the reference **{subject_name}** questions (indexed 0-{len(questions_series)-1}):\n\n"]
    for i, question_text in enumerate(questions_series):
        user_message_parts.append(f"{i}. {question_text}\n")
    user_message_parts.append(f"\nPlease generate {num_to_generate} new, unique, **{difficulty_text}** {subject_name} MCQs with all required fields, following the exact output format.")
    
    return system_message, "".join(user_message_parts)

def call_gemini_api(system_message, user_prompt, model, task_name="task", subject=""):
    if subject in ['english', 'chinese', 'science']:
        temp = 0.7
    else:
        temp = 0.4
        
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
            "answer": answer.strip(), # Expected format: "(B)" or "B"
            "reasoning": reasoning.strip()
        }
        new_questions.append(question_obj)
        
    return new_questions

def process_generation_loop(file_path, subject_lower, num_to_generate):
    final_generated_list = []
    retries_used = 0
    questions_series = load_and_select_questions(file_path, QUESTIONS_TO_SELECT, QUESTION_COLUMN_NAME)
    if questions_series is None:
        return []

    while len(final_generated_list) < num_to_generate and retries_used < MAX_RETRIES:
        questions_needed = num_to_generate - len(final_generated_list)
        reference_subset = questions_series.sample(min(QUESTIONS_TO_SELECT, questions_needed * 5, len(questions_series))) 
        
        st.info(f"Attempt {retries_used + 1}/{MAX_RETRIES}: Generating {questions_needed} more questions...")
        
        system_msg, user_prompt = format_prompt_for_generation(reference_subset, subject_lower, questions_needed)
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
    st.session_state.answer_checked = False # Reset for new question

def prev_q():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.session_state.answer_checked = False # Reset for new question

def check_answer_handler():
    st.session_state.answer_checked = True

def display_mcq_session():
    """Renders interactive MCQ questions one-by-one."""
    generated_list = st.session_state.latest_generated_list
    total_count = len(generated_list)
    
    # Safety Check
    if st.session_state.current_index >= total_count:
        st.session_state.current_index = 0

    current_q_index = st.session_state.current_index
    item = generated_list[current_q_index]
    options = item.get('options', {})

    # -- Header --
    st.header(f"Question {current_q_index + 1} of {total_count}")
    st.caption(f"Topic: {item.get('topic', 'N/A')} | Difficulty: {item.get('difficulty', 'N/A')}")
    st.write("---")
    
    # -- Display Question Text (using st.code for consistent font) --
    st.code(item.get('question', 'N/A'), language='text')

    # -- Interactive Radio Options --
    # Construct options list for radio button
    radio_options = [
        f"A) {options.get('A', 'N/A')}",
        f"B) {options.get('B', 'N/A')}",
        f"C) {options.get('C', 'N/A')}",
        f"D) {options.get('D', 'N/A')}"
    ]
    
    # Use a unique key for the radio based on the question index to preserve selection
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
        pass # Logic is handled by the session state update
    
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
            # Extract the user's selected letter (A, B, C, or D)
            user_letter = selected_option[0] 
            
            # Clean the correct answer string (e.g., "(B)" -> "B")
            correct_answer_raw = item.get('answer', 'N/A')
            correct_letter_match = re.search(r"[ABCD]", correct_answer_raw)
            correct_letter = correct_letter_match.group(0) if correct_letter_match else "?"

            if user_letter == correct_letter:
                st.success(f"✅ **Correct!** The answer is **{correct_letter}**.")
            else:
                st.error(f"❌ **Incorrect.** You selected **{user_letter}**, but the correct answer is **{correct_letter}**.")
            
            st.markdown("**Reasoning:**")
            st.code(item.get('reasoning', 'N/A'), language='text')
        else:
            st.warning("Please select an option before checking.")

# --- Main Streamlit App Function ---

def main():
    st.set_page_config(page_title="AI Question Generator", layout="wide")
    st.title("📚 AI Question Generator")
    
    # --- 1. INITIALIZE SESSION STATE ---
    initialize_session_state()

    # --- Sidebar for Controls ---
    st.sidebar.header("⚙️ Configuration")

    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")

    subject_name = st.sidebar.selectbox(
        "1. Choose a subject:",
        ["English", "Chinese", "Math", "Science"]
    )

    # Logic for num_questions
    if subject_name in ["English", "Chinese"]:
        num_to_generate = st.sidebar.number_input(
            "2. How many comprehension questions?", 
            min_value=1, max_value=20, value=10
        )
    else: # Math or Science
        num_to_generate = st.sidebar.number_input(
            "2. How many MCQs to generate?", 
            min_value=1, max_value=20, value=5
        )

    generate_button = st.sidebar.button("🚀 Generate Questions")
    
    st.markdown("---")
    
    generation_status_placeholder = st.empty()

    if generate_button:
        # Reset specific session vars for new generation
        st.session_state.total_tokens_used = 0 
        st.session_state.latest_generated_list = []
        st.session_state.current_index = 0
        st.session_state.answer_checked = False
        
        # --- Validation ---
        if not api_key:
            st.error("Please enter your Google API Key in the sidebar.")
        else:
            try:
                genai.configure(api_key=api_key)
                generation_status_placeholder.success("API Key configured. Generating...")
                
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

                    elif subject_lower in ['math', 'science']:
                        # --- MCQ ---
                        file_path = MATH_EXCEL_FILE_PATH if subject_lower == 'math' else SCIENCE_EXCEL_FILE_PATH
                        generated_list = process_generation_loop(file_path, subject_lower, num_to_generate)
                        
                        if generated_list:
                            st.session_state.latest_generated_list = generated_list
                            st.subheader(f"✅ Generation Complete! Starting Interactive Review ({len(generated_list)}/{num_to_generate})")
                        elif not generated_list:
                             st.error("Failed to generate any valid questions after all retry attempts.")
                            
            except Exception as e:
                st.error(f"An unexpected error occurred during the process: {e}")
    
    # --- INTERACTIVE MCQ DISPLAY (Persists across reruns) ---
    if st.session_state.latest_generated_list:
        st.markdown("---")
        display_mcq_session()
        st.markdown("---")

    # --- FINAL TOKEN COUNT ---
    st.subheader("💰 Token Usage Summary")
    st.info(f"Total tokens consumed for this session: **{st.session_state.total_tokens_used:,}** tokens.")

# --- This part must be at the very end ---
if __name__ == "__main__":

    main()

