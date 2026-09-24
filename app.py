import re
import json
import asyncio
import requests
import gradio as gr
from huggingface_hub import InferenceClient

# ==============================================================================
# 1. CONFIGURATION & MAPPINGS
# ==============================================================================

SUBJECT_QUESTION_TYPES = {
    "English": ["Comprehension", "Letter Writing", "Essay", "Grammar"],
    "Mathematics": ["Numerical Problems", "Multiple Choice (MCQ)", "Proof / Derivation", "Mixed Formats"],
    "Physics": ["Numerical Problems", "Conceptual / Short Answer", "Derivation", "Mixed Formats"],
    "Chemistry": ["Chemical Equations", "Numerical Problems", "Short Answer", "Mixed Formats"],
    "General Science": ["Multiple Choice (MCQ)", "Short Answer", "Diagram / Lab", "Mixed Formats"],
    "Social Studies": ["Short Answer", "Long Answer / Essay", "Multiple Choice (MCQ)", "Map Work"]
}

DEFAULT_HF_MODELS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

# Modern, Professional Dashboard Theme CSS
CUSTOM_CSS = """
/* Global App Background */
body, .gradio-container {
    background-color: #F4F6F9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1F2937 !important;
}

/* Styled Content Cards */
.form-card {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 20px !important;
}

/* Card Section Titles */
.card-title {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    margin-bottom: 16px !important;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 2px solid #F3F4F6;
    padding-bottom: 8px;
}

/* Primary Action Button */
#generate_btn {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 28px !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
}

#generate_btn:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4) !important;
}

/* Inputs, Dropdowns & Textareas */
input[type="text"], input[type="password"], textarea, select {
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
    background-color: #FAFAFA !important;
    color: #111827 !important;
    padding: 10px 12px !important;
    font-size: 14px !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #10B981 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
    outline: none !important;
}

/* Markdown Generated Content Area */
.markdown-output {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 14px !important;
    padding: 28px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    line-height: 1.7 !important;
    font-size: 15px !important;
    color: #111827 !important;
}

.markdown-output h1, .markdown-output h2, .markdown-output h3 {
    color: #065F46 !important;
    font-weight: 800 !important;
}

/* Top Banner Styling */
.top-hero {
    background: linear-gradient(135deg, #065F46 0%, #047857 100%);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    color: #FFFFFF;
    box-shadow: 0 10px 25px -5px rgba(6, 95, 70, 0.2);
    margin-bottom: 24px;
}

.top-hero h1 {
    color: #FFFFFF !important;
    margin: 0;
    font-size: 28px;
    font-weight: 800;
}

.top-hero p {
    color: #D1FAE5 !important;
    margin-top: 8px;
    font-size: 15px;
}
"""

# ==============================================================================
# 2. UTILITIES & CLEANING
# ==============================================================================

def clean_human_readable_text(text: str) -> str:
    """Cleans raw LLM outputs into clean notation for browser rendering."""
    if not text:
        return ""

    text = text.replace(r'\rightarrow', '→').replace(r'\leftarrow', '←')
    text = text.replace(r'\times', '×').replace(r'\cdot', '·').replace(r'\div', '÷')
    text = text.replace(r'\degree', '°').replace(r'\pm', '±')
    text = text.replace(r'\\', '')

    text = re.sub(r'\_\{([^}]+)\}', r'<sub>\1</sub>', text)
    text = re.sub(r'\_([a-zA-Z0-9]+)', r'<sub>\1</sub>', text)
    text = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', text)
    text = re.sub(r'\^([a-zA-Z0-9]+)', r'<sup>\1</sup>', text)

    text = re.sub(r'\\\(\vert{}\\\)', '', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)

    return text


def validate_answer_alignment(markdown_text: str, expected_count: int) -> tuple[str, str]:
    """Validates that every generated question has a matching solution in the Answer Key."""
    if "### Answer Key" not in markdown_text and "## Answer Key" not in markdown_text:
        return markdown_text, ""

    parts = re.split(r'#{2,3}\s*Answer Key', markdown_text, maxsplit=1)
    a_section = parts[1] if len(parts) > 1 else ""

    answers_found = re.findall(r'(?:S\d+|A\d+|^\d+\.)', a_section, re.MULTILINE)
    answer_count = len(answers_found)

    warning_msg = ""
    if answer_count < expected_count:
        warning_msg = f"⚠️ **Validation Notice**: Requested {expected_count} questions, but detected {answer_count} corresponding answers in the Answer Key."

    return markdown_text, warning_msg


# ==============================================================================
# 3. API CALL HANDLERS
# ==============================================================================

def call_gemini_api(api_key: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 8192}
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API Error ({response.status_code}): {response.text}")
    
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError:
        raise RuntimeError(f"Unexpected response structure from Gemini API: {data}")


def call_huggingface_api(api_key: str, model_name: str, prompt: str) -> str:
    try:
        client = InferenceClient(token=api_key.strip())
        response = client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as err:
        raise RuntimeError(f"Hugging Face API Error: {str(err)}")


# ==============================================================================
# 4. PROMPT BUILDER ENGINE
# ==============================================================================

def construct_batch_prompt(
    output_mode, grade, subject, question_type, difficulty,
    topic, custom_instructions, start_q, end_q, total_questions, syllabus_text
):
    answer_key_instruction = (
        f"For each question from {start_q} to {end_q}, provide a detailed step-by-step solution in a dedicated '### Answer Key' section at the end of this batch."
        if output_mode == "Questions & Answer Key"
        else "DO NOT include answers or solutions. Generate questions ONLY."
    )

    syllabus_context = ""
    if syllabus_text:
        syllabus_context = f"\n---\nRELEVANT SYLLABUS / CONTEXT MATERIAL:\n{syllabus_text[:3000]}\n---\nEnsure questions strictly align with the concepts present in the syllabus extract above.\n"

    prompt = f"""
You are an expert curriculum developer crafting assessment questions.

Generate questions numbered EXACTLY from {start_q} to {end_q} (out of {total_questions} total) based on:

- **Target Level:** {grade}
- **Subject:** {subject}
- **Specific Topic:** {topic}
- **Question Format:** {question_type}
- **Difficulty Level:** {difficulty}
- **Special Instructions:** {custom_instructions if custom_instructions else 'None'}
- **Output Mode:** {output_mode}

{syllabus_context}

### Formatting Guidelines:
1. Number questions strictly from {start_q} to {end_q} (e.g., `{start_q}.`, `{start_q+1}.`).
2. Write chemical formulas and math equations using plain readable format or standard superscript/subscript (e.g., H2O, x2). Do NOT output messy raw LaTeX backslashes or math code blocks.
3. If options are needed (for Multiple Choice), list options as `A)`, `B)`, `C)`, `D)`.
4. Ensure tone and difficulty strictly match **{grade}**.
5. {answer_key_instruction}

Output clean Markdown syntax only.
"""
    return prompt.strip()


# ==============================================================================
# 5. MAIN CONTROLLER FUNCTION
# ==============================================================================

async def generate_llm_worksheet(
    provider, api_key, hf_model, file_text_input, output_mode, grade,
    subject, question_type, difficulty, topic, custom_instructions, num_questions
):
    if not api_key or len(api_key.strip()) == 0:
        return "⚠️ **API Key Missing**: Please enter your API Key or Access Token to proceed.", ""

    if not topic or len(topic.strip()) == 0:
        return "⚠️ **Topic Missing**: Please enter a specific topic or sub-topic.", ""

    num_q = int(num_questions)
    syllabus_text = file_text_input if file_text_input else ""

    batch_size = 10 if num_q > 15 else num_q
    total_batches = (num_q + batch_size - 1) // batch_size

    all_question_parts = []
    all_answer_parts = []

    try:
        for batch_idx in range(total_batches):
            start_q = batch_idx * batch_size + 1
            end_q = min((batch_idx + 1) * batch_size, num_q)

            prompt = construct_batch_prompt(
                output_mode, grade, subject, question_type, difficulty,
                topic, custom_instructions, start_q, end_q, num_q, syllabus_text
            )

            if provider == "Google Gemini":
                batch_response = call_gemini_api(api_key, prompt)
            else:
                batch_response = call_huggingface_api(api_key, hf_model, prompt)

            if "### Answer Key" in batch_response or "## Answer Key" in batch_response:
                parts = re.split(r'#{2,3}\s*Answer Key', batch_response, maxsplit=1)
                all_question_parts.append(parts[0].strip())
                if len(parts) > 1:
                    all_answer_parts.append(parts[1].strip())
            else:
                all_question_parts.append(batch_response.strip())

    except Exception as err:
        return f"❌ **Generation Failed**: {str(err)}", ""

    header_block = f"# Worksheet: {topic}\n"
    header_block += f"**Grade:** {grade} | **Subject:** {subject} | **Difficulty:** {difficulty}\n\n---\n\n"
    
    combined_questions = "\n\n".join(all_question_parts)
    final_markdown = header_block + combined_questions

    if output_mode == "Questions & Answer Key" and all_answer_parts:
        combined_answers = "\n\n".join(all_answer_parts)
        final_markdown += "\n\n---\n\n### Answer Key\n\n" + combined_answers

    cleaned_markdown = clean_human_readable_text(final_markdown)
    validated_markdown, validation_notice = validate_answer_alignment(cleaned_markdown, num_q)

    status_success = f"✅ **Successfully generated** {num_q} questions for **{grade} ({subject})** using **{provider}**."
    if validation_notice:
        status_success += f"\n\n{validation_notice}"

    return status_success, validated_markdown


# ==============================================================================
# 6. GRADIO UI LAYOUT (Cleaned with Professional Cards)
# ==============================================================================

# ==============================================================================
# 6. GRADIO UI LAYOUT
# ==============================================================================

with gr.Blocks(title="AI Exam & Worksheet Generator", css=CUSTOM_CSS) as demo:
    
    gr.HTML(
        """
        <div class="top-hero">
            <h1>📚 Smart AI Worksheet Builder</h1>
            <p>Empowering educators, parents, and students to create customized practice exams and answer keys in seconds.</p>
        </div>
        """
    )

    with gr.Column(elem_classes=["form-card"]):
        gr.HTML('<div class="card-title">⚙️ AI Engine & Credentials</div>')
        with gr.Row():
            provider_dropdown = gr.Dropdown(
                choices=["Google Gemini", "Hugging Face"],
                value="Google Gemini",
                label="AI Engine",
                scale=1
            )
            api_key_input = gr.Textbox(
                label="API Key / Access Token",
                type="password",
                placeholder="Enter your API Key...",
                scale=2
            )

        hf_model_dropdown = gr.Dropdown(
            choices=DEFAULT_HF_MODELS,
            value=DEFAULT_HF_MODELS[0],
            label="Hugging Face Model Choice",
            visible=False
        )

    with gr.Column(elem_classes=["form-card"]):
        gr.HTML('<div class="card-title">📝 Worksheet Parameters</div>')
        
        output_mode_radio = gr.Radio(
            choices=["Questions Only", "Questions & Answer Key"],
            value="Questions & Answer Key",
            label="Output Format"
        )

        with gr.Row():
            grade_dropdown = gr.Dropdown(
                choices=[f"Class {i}" for i in range(1, 13)],
                value="Class 10",
                label="Target Grade / Level"
            )
            subject_dropdown = gr.Dropdown(
                choices=list(SUBJECT_QUESTION_TYPES.keys()),
                value="Mathematics",
                label="Subject"
            )

        with gr.Row():
            q_type_dropdown = gr.Dropdown(
                choices=SUBJECT_QUESTION_TYPES["Mathematics"],
                value="Numerical Problems",
                label="Question Format"
            )
            difficulty_dropdown = gr.Dropdown(
                choices=["Easy", "Medium", "Hard", "Advanced"],
                value="Medium",
                label="Difficulty Level"
            )

    with gr.Column(elem_classes=["form-card"]):
        gr.HTML('<div class="card-title">📖 Syllabus & Topic Configuration</div>')
        
        topic_input = gr.Textbox(
            label="Specific Topic Name",
            value="Linear Equations in Two Variables",
            placeholder="e.g., Photosynthesis, Trigonometry, World War II..."
        )

        file_text_input = gr.Textbox(
            label="Syllabus / Textbook Excerpt (Optional)",
            placeholder="Paste chapter excerpts, curriculum points, or specific textbook notes here...",
            lines=3
        )
        
        custom_instructions = gr.Textbox(
            label="Additional Instructions (Optional)",
            placeholder="e.g., Emphasize real-world word problems, keep questions short, add workspace...",
            lines=2
        )
        
        num_q_slider = gr.Slider(
            minimum=1,
            maximum=50,
            value=10,
            step=1,
            label="Total Number of Questions"
        )

    generate_btn = gr.Button("✨ Generate Practice Worksheet", elem_id="generate_btn", size="lg")

    # FIXED: Removed style parameter to avoid TypeError in Gradio v4+
    with gr.Column(elem_classes=["form-card"]):
        gr.HTML('<div class="card-title">📄 Generated Worksheet Preview</div>')
        status_box = gr.Markdown("Ready to craft your custom worksheet.")
        output_display = gr.Markdown(value="*Your generated practice exam will appear here once ready...*", elem_classes=["markdown-output"])

        
    # Dynamic UI Event Handlers
    def on_subject_change(selected_subject):
        types = SUBJECT_QUESTION_TYPES.get(selected_subject, ["Mixed Formats", "Short Answer"])
        return gr.update(choices=types, value=types[0])

    subject_dropdown.change(
        fn=on_subject_change,
        inputs=[subject_dropdown],
        outputs=[q_type_dropdown]
    )

    def update_hf_models(provider):
        if provider == "Hugging Face":
            return gr.update(visible=True, choices=DEFAULT_HF_MODELS, value=DEFAULT_HF_MODELS[0])
        return gr.update(visible=False)

    provider_dropdown.change(
        fn=update_hf_models,
        inputs=[provider_dropdown],
        outputs=[hf_model_dropdown]
    )

    generate_btn.click(
        fn=generate_llm_worksheet,
        inputs=[
            provider_dropdown,
            api_key_input,
            hf_model_dropdown,
            file_text_input,
            output_mode_radio,
            grade_dropdown,
            subject_dropdown,
            q_type_dropdown,
            difficulty_dropdown,
            topic_input,
            custom_instructions,
            num_q_slider
        ],
        outputs=[status_box, output_display]
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
