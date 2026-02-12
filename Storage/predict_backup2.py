import streamlit as st
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import preprocess # Använder din intelligenta preprocessor

# --- 1. KONFIGURATION ---
st.set_page_config(page_title="MNIST Live Analytics", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("mnist_model_final_svc.joblib")

model = load_model()

# --- 2. LAYOUT ---
st.title("🔢 MNIST Live Analytics")

# Flikar för inmatning
tab_draw, tab_upload = st.tabs(["✍️ Rita Siffra", "📁 Ladda upp Bild"])

# Gemensam funktion för prediktion och visning
def perform_live_analysis(img_input):
    if img_input is None:
        return
        
    # Kör din intelligenta preprocess
    features, img_28 = preprocess.preprocess_image(img_input)
    
    # SVC Prediktion
    scores = model.decision_function(features)[0]
    probs = np.exp(scores - np.max(scores)) / np.exp(scores - np.max(scores)).sum()
    pred = np.argmax(probs)
    conf = probs[pred]

    # Kompakt resultatpanel
    st.write("---")
    st.subheader(f"Modellen gissar: {pred} ({conf:.0%})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_input, caption="Indata", use_container_width=True)
    with col2:
        st.image(img_28, caption="Maskinens vy", use_container_width=True)

    # Liggande stapeldiagram för sannolikhet
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.bar(range(10), probs, color=['#3498db']*10)
    ax.patches[pred].set_color('#f1c40f') # Gul färg för vinnaren
    ax.set_xticks(range(10))
    ax.set_yticks([]) 
    st.pyplot(fig)

# --- 3. INPUT-LOGIK (Utan knappar) ---
with tab_draw:
    st.info("Analysen sker automatiskt så fort du har ritat klart ett streck.")
    
    # Vi tar bort 'update_freq' för att lösa felet
    canvas_result = st_canvas(
        fill_color="white", 
        stroke_width=18, 
        stroke_color="black",
        background_color="white", 
        height=280, 
        width=280,
        drawing_mode="freedraw", 
        key="canvas_live"
    )
    
    # Automatisk analys av canvas
    if canvas_result.image_data is not None:
        # Kolla om användaren faktiskt har ritat något (inte bara en tom vit yta)
        img_draw = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        if np.mean(np.array(img_draw)) < 254: # Om medelvärdet är under 255 finns det svärta
            perform_live_analysis(img_draw)

with tab_upload:
    uploaded_file = st.file_uploader("Släpp en bild här", type=["jpg", "png"])
    
    # Automatisk analys av uppladdning
    if uploaded_file is not None:
        img_upload = Image.open(uploaded_file)
        perform_live_analysis(img_upload)