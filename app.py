import streamlit as st
import torch
import html
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Story Generator V3",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# PREMIUM CREATIVE STUDIO UI (UPDATED)
# ==================================================

st.markdown("""
<style>
/* Simple, normal, professional sans-serif font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b0f19;
    --text: #ffffff;
    --muted: #94a3b8;
    --accent: #3b82f6;
}

.stApp {
    background-color: var(--bg);
    background-image: 
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
    color: var(--text);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.block-container {
    max-width: 1000px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

/* HERO SECTION */
.hero {
    text-align: center;
    padding: 20px 10px 40px;
}

.hero-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    color: #60a5fa;
    background: rgba(59, 130, 246, 0.1);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero h1 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: clamp(32px, 5vw, 56px);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    letter-spacing: -1px;
}

.hero h1 span {
    background: linear-gradient(to right, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: var(--muted);
    font-size: 18px;
    margin: 20px auto 0;
    max-width: 600px;
    line-height: 1.6;
}

/* ROLES / FEATURES (NO BOXES, 3D ICONS) */
.feature-container {
    text-align: center;
    padding: 10px;
}

.feature-icon {
    font-size: 42px;
    margin-bottom: 16px;
    /* CSS hack to give icons a 3D app-symbol pop */
    text-shadow: 0px 4px 0px rgba(255,255,255,0.2), 
                 0px 8px 15px rgba(0,0,0,0.5), 
                 0px 2px 5px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}

.feature-container h3 {
    color: #f8fafc;
    font-size: 17px;
    font-weight: 600;
    margin: 0 0 8px;
}

.feature-container p {
    color: var(--muted);
    font-size: 14px;
    margin: 0;
    line-height: 1.5;
}

/* TYPOGRAPHY & SECTIONS */
.section-heading {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 600;
    margin: 10px 0 4px;
}

.section-caption {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 20px;
}

/* CLEAN INPUTS */
.stTextArea label,
.stSelectbox label,
.stFileUploader label,
.stCameraInput label,
.stAudioInput label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

.stTextArea textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.6) !important;
    border-color: rgba(148, 163, 184, 0.2) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* BUTTONS */
.stButton > button,
.stDownloadButton > button {
    width: 100%;
    min-height: 44px;
    border-radius: 8px;
    border: none;
    color: #fff;
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    transform: translateY(-1px);
    color: white;
}

/* STORY OUTPUT */
.story-output {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 30px;
    color: #e2e8f0;
    font-size: 16px;
    line-height: 1.8;
    white-space: pre-wrap;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

div[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 8px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 40px 0 10px;
}

hr {
    border-color: rgba(148, 163, 184, 0.1) !important;
}

@media (max-width: 700px) {
    .block-container { padding-left: 1.2rem; padding-right: 1.2rem; }
    .hero { padding-top: 10px; }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

if "story_history" not in st.session_state:
    st.session_state.story_history = []
if "generated_story" not in st.session_state:
    st.session_state.generated_story = ""
if "show_file" not in st.session_state:
    st.session_state.show_file = False
if "show_cam" not in st.session_state:
    st.session_state.show_cam = False
if "show_mic" not in st.session_state:
    st.session_state.show_mic = False


# ==================================================
# LOAD HUGGING FACE MODEL
# ==================================================

@st.cache_resource
def load_model():
    model_path = "varunn27/ai-story-generator-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ==================================================
# GENERATE STORY FUNCTION
# ==================================================

def generate_story(prompt, category, length):
    tokenizer, model, device = load_model()
    token_limits = {"Short": 100, "Medium": 200, "Long": 300}
    max_tokens = token_limits[length]

    formatted_prompt = f"""
Write a creative and engaging {category} story.

Story idea: {prompt}

Story:
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story = generated_text[len(formatted_prompt):].strip()
    return story


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-badge">
        Professional Studio
    </div>
    <h1>
        <span>AI STORY GENERATOR</span>
    </h1>
    <p>
        Turn a simple idea into an unforgettable narrative.
        Choose a genre, set the length, and let the AI generate your content.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# FEATURE ROLES (UNBOXED, 3D ICONS)
# ==================================================

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
<div class="feature-container">
    <div class="feature-icon">💠</div>
    <h3>Creative Engine</h3>
    <p>Transform raw ideas into original stories instantly.</p>
</div>
""", unsafe_allow_html=True)

with f2:
    st.markdown("""
<div class="feature-container">
    <div class="feature-icon">💡</div>
    <h3>Genre Control</h3>
    <p>Dictate the style, from science fiction to drama.</p>
</div>
""", unsafe_allow_html=True)

with f3:
    st.markdown("""
<div class="feature-container">
    <div class="feature-icon">🚀</div>
    <h3>Custom Model</h3>
    <p>Powered by your dedicated Story Generator V3 architecture.</p>
</div>
""", unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)


# ==================================================
# STORY INPUT
# ==================================================

st.markdown("""
<div class="section-heading">
    Create your story
</div>
<div class="section-caption">
    Input your core concept, character, or situation below.
</div>
""", unsafe_allow_html=True)


prompt = st.text_area(
    "Your concept",
    placeholder="Example: A young boy discovers a mysterious door hidden deep inside an ancient forest...",
    height=125
)


# ==================================================
# CATEGORY AND LENGTH
# ==================================================

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox(
        "Genre",
        ["Fantasy", "Adventure", "Horror", "Romance", "Science Fiction", "Mystery", "Comedy", "Drama", "Motivational", "Fairy Tale"]
    )

with col2:
    length = st.selectbox(
        "Length",
        ["Short", "Medium", "Long"]
    )


st.markdown("<br>", unsafe_allow_html=True)


# ==================================================
# ADDITIONAL OPTIONS
# ==================================================

st.markdown("**Optional References**")

opt1, opt2, opt3 = st.columns(3)

with opt1:
    if st.button("📁 Upload file", key="upload_toggle"):
        st.session_state.show_file = not st.session_state.show_file

with opt2:
    if st.button("📷 Take photo", key="camera_toggle"):
        st.session_state.show_cam = not st.session_state.show_cam

with opt3:
    if st.button("🎤 Record audio", key="mic_toggle"):
        st.session_state.show_mic = not st.session_state.show_mic


# FILE UPLOAD
if st.session_state.show_file:
    uploaded_file = st.file_uploader(
        "Upload reference documents or images",
        type=["pdf", "txt", "docx", "png", "jpg"]
    )
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")


# CAMERA
if st.session_state.show_cam:
    camera_photo = st.camera_input("Take a photo for visual inspiration")
    if camera_photo:
        st.success("Photo captured successfully!")


# MICROPHONE
if st.session_state.show_mic:
    voice_prompt = st.audio_input("Record your voice prompt")
    if voice_prompt:
        st.success("Voice recording captured successfully!")


st.divider()


# ==================================================
# ACTION BUTTONS
# ==================================================

action1, action2 = st.columns([3, 1])

with action1:
    generate_button = st.button(
        "Generate Content",
        type="primary",
        use_container_width=True
    )

with action2:
    if st.button("Clear Data", use_container_width=True):
        st.session_state.generated_story = ""
        st.rerun()


# ==================================================
# GENERATE STORY PROCESS
# ==================================================

if generate_button:
    if not prompt.strip():
        st.warning("Please enter a story idea first.")
    else:
        try:
            with st.spinner("Processing generation..."):
                story = generate_story(prompt, category, length)
                st.session_state.generated_story = story
                
                st.session_state.story_history.append({
                    "prompt": prompt,
                    "category": category,
                    "length": length,
                    "story": story
                })
        except Exception as e:
            st.error(f"System Error: {e}")


# ==================================================
# DISPLAY GENERATED STORY
# ==================================================

if st.session_state.generated_story:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div class="section-heading">
    Generated Output
</div>
<div class="section-caption">
    Review your generated content below.
</div>
""", unsafe_allow_html=True)

    safe_story = html.escape(st.session_state.generated_story)

    st.markdown(
        f'<div class="story-output">{safe_story}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="Download Document",
        data=st.session_state.generated_story,
        file_name="AI_Generated_Story.txt",
        mime="text/plain"
    )


# ==================================================
# STORY HISTORY
# ==================================================

if st.session_state.story_history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div class="section-heading">
    Session History
</div>
<div class="section-caption">
    Previous generations from this active session.
</div>
""", unsafe_allow_html=True)

    for index, item in enumerate(reversed(st.session_state.story_history)):
        with st.expander(f"Record {len(st.session_state.story_history) - index} | {item['category']} | {item['length']}"):
            st.write("**Concept:**", item["prompt"])
            st.write("**Length:**", item["length"])
            st.write(item["story"])


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    AI Story Generator V3 • Professional Build
</div>
""", unsafe_allow_html=True)