import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# 1. SECURE SETUP & API CONFIG
# ==========================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY or len(GOOGLE_API_KEY) < 10:
    st.error("🚨 API Key missing or invalid! Please check your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
vlm_model = genai.GenerativeModel('gemini-2.5-flash')
MODEL_PATH = "./checkpoints/segmentation_model"

st.set_page_config(page_title="AgriOmni-VLM", layout="wide")
st.title("🌿 AgriOmni: Multilingual Crop Diagnosis")

# ==========================================
# 2. METADATA & LANGUAGE SIDEBAR
# ==========================================
st.sidebar.header("🌍 Farm Context & Settings")

user_language = st.sidebar.selectbox(
    "🗣️ Preferred Language", 
    ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Spanish", "French"]
)

farm_location = st.sidebar.text_input("📍 Region / Location", "Maharashtra, India")
farm_weather = st.sidebar.selectbox(
    "☁️ Current Weather", 
    ["Hot & Dry", "Monsoon / Humid", "Cool & Damp", "Moderate"]
)

st.sidebar.divider()
st.sidebar.info("AgriOmni uses this metadata to map observable symptoms to localized, context-aware treatment recommendations.")

# Initialize Streamlit memory state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analyzed_image" not in st.session_state:
    st.session_state.analyzed_image = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None

# ==========================================
# 3. LOAD LOCAL SEGMENTATION MODEL
# ==========================================
@st.cache_resource
def load_segmentation_model():
    processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
    return processor, model

processor, seg_model = load_segmentation_model()

# ==========================================
# 4. MULTIMODAL ROUTING
# ==========================================
st.markdown("### 1. Select Input Type")
analysis_mode = st.radio(
    "What are you scanning?", 
    ["🍃 Leaf (Deep AI Segmentation)", "🍎 Fruit / Crop (Direct VLM Analysis)"], 
    horizontal=True
)

# If user switches modes, clear the chat so the AI doesn't get confused
if st.session_state.current_mode != analysis_mode:
    st.session_state.chat_history = []
    st.session_state.analyzed_image = None
    st.session_state.current_mode = analysis_mode

st.markdown("### 2. Upload Image")
uploaded_file = st.file_uploader("Upload an image to begin analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Original Upload", use_container_width=True)
        
    with col2:
        if analysis_mode == "🍃 Leaf (Deep AI Segmentation)":
            with st.spinner("Extracting disease map (Local AI)..."):
                inputs = processor(images=original_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = seg_model(**inputs)
                logits = outputs.logits
                resized_logits = torch.nn.functional.interpolate(
                    logits, size=original_image.size[::-1], mode="bilinear", align_corners=False
                )
                predicted_mask = resized_logits.argmax(dim=1).squeeze().cpu().numpy()
                
                image_np = np.array(original_image)
                red_overlay = np.zeros_like(image_np)
                red_overlay[:, :, 0] = 255 
                alpha = 0.5 
                disease_visual_np = np.where(predicted_mask[..., None] == 1, 
                                          (image_np * (1 - alpha) + red_overlay * alpha).astype(np.uint8), 
                                          image_np)
                
                disease_visual_pil = Image.fromarray(disease_visual_np)
                st.session_state.analyzed_image = disease_visual_pil
                st.image(disease_visual_pil, caption="AgriOmni Disease Map", use_container_width=True)
        else:
            with st.spinner("Routing directly to VLM..."):
                st.info("Bypassing local segmentation. Routing raw high-res feed to Gemini VLM.")
                st.image(original_image, caption="Raw Feed to VLM", use_container_width=True)
                st.session_state.analyzed_image = original_image

    # ==========================================
    # 5. VOICE & CHAT ADVISORY (OMNI-MODAL)
    # ==========================================
    st.divider()
    st.subheader(f"💬 AgriOmni Advisory ({user_language})")
    
    for message in st.session_state.chat_history:
        if message["content"] != "*(Voice message sent)*":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Base prompt that dictates behavior, language, and weather constraints
    base_prompt = f"""
    You are AgriOmni, an expert agricultural AI. 
    FARM CONTEXT: Location: {farm_location} | Weather: {farm_weather}
    
    CRITICAL RULES: 
    - Do NOT provide exact chemical mixing ratios. 
    - Translate your ENTIRE response into {user_language}. Do not use English unless it is a specific scientific name.
    """

    # --- INITIAL DIAGNOSIS TRIGGER ---
    if len(st.session_state.chat_history) == 0:
        if analysis_mode == "🍃 Leaf (Deep AI Segmentation)":
            system_prompt = base_prompt + "\nI am providing a leaf image with diseased areas highlighted in red. Provide a definitive diagnosis, severity, and treatment steps."
        else:
            system_prompt = base_prompt + "\nI am providing a raw image of a crop/fruit. Provide a definitive diagnosis, severity, and treatment steps."
        
        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing and translating to {user_language}..."):
                response = vlm_model.generate_content([system_prompt, st.session_state.analyzed_image])
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})

    # --- MULTIMODAL INPUTS: TEXT OR VOICE ---
    user_question = st.chat_input(f"Type a question in {user_language} or English...")
    audio_value = st.audio_input("🎙️ Or record your question (Voice)")

    active_input = None
    input_type = None

    if user_question:
        active_input = user_question
        input_type = "text"
    elif audio_value:
        active_input = audio_value
        input_type = "audio"

    # --- PROCESS FOLLOW-UP QUESTION ---
    if active_input:
        with st.chat_message("user"):
            if input_type == "text":
                st.markdown(active_input)
                st.session_state.chat_history.append({"role": "user", "content": active_input})
            else:
                st.audio(active_input)
                st.session_state.chat_history.append({"role": "user", "content": "*(Voice message sent)*"})
        
        with st.chat_message("assistant"):
            with st.spinner("Processing query..."):
                first_diagnosis = st.session_state.chat_history[0]["content"]
                follow_up_prompt = base_prompt + f"\nEarlier, you diagnosed: '{first_diagnosis}'. Keep advice consistent. The farmer is now asking a follow-up question."
                
                # Dynamic Payload: Send Text+Image OR Audio+Image
                if input_type == "text":
                    payload = [follow_up_prompt, active_input, st.session_state.analyzed_image]
                else:
                    audio_part = {"mime_type": "audio/wav", "data": active_input.getvalue()}
                    payload = [follow_up_prompt, audio_part, st.session_state.analyzed_image]
                
                response = vlm_model.generate_content(payload)
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})