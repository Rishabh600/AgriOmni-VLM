import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Pointing exactly to where you saved your Kaggle output
MODEL_PATH = "./checkpoints/segmentation_model"

st.set_page_config(page_title="AgriOmni Disease Scanner", layout="wide")
st.title("🌿 AgriOmni: Leaf Disease Segmentation")

# ==========================================
# 2. LOAD THE MODEL (Cached so it's fast)
# ==========================================
# ==========================================
# 2. LOAD THE MODEL (Cached so it's fast)
# ==========================================
@st.cache_resource
def load_ai_model():
    # Force Hugging Face to use the Segformer logic for your local files
    processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
    return processor, model

processor, model = load_ai_model()

# ==========================================
# 3. UPLOAD IMAGE
# ==========================================
uploaded_file = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    st.write("### Analyzing Leaf...")
    
    # ==========================================
# 4. RUN INFERENCE (The Magic)
    # ==========================================
    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt")
    
    # Pass through your trained model
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get the raw predictions and resize them back to the original image size
    logits = outputs.logits
    resized_logits = torch.nn.functional.interpolate(
        logits, 
        size=image.size[::-1], # PIL size is (width, height), PyTorch needs (height, width)
        mode="bilinear", 
        align_corners=False
    )
    
    # Convert prediction to a binary mask (0 for healthy, 1 for diseased)
    predicted_mask = resized_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # ==========================================
# 5. CREATE VISUAL OVERLAY
    # ==========================================
    # Let's paint the diseased area bright red so it looks professional!
    image_np = np.array(image)
    
    # Create a red overlay array
    red_overlay = np.zeros_like(image_np)
    red_overlay[:, :, 0] = 255 # Full Red
    
    # Blend the original image with the red overlay where the mask is '1'
    alpha = 0.5 # Transparency
    disease_visual = np.where(predicted_mask[..., None] == 1, 
                              (image_np * (1 - alpha) + red_overlay * alpha).astype(np.uint8), 
                              image_np)
    
    # ==========================================
# 6. DISPLAY RESULTS
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Upload", use_container_width=True)
        
    with col2:
        st.image(disease_visual, caption="AI Disease Mapping", use_container_width=True)

    st.success("✅ Analysis Complete!")