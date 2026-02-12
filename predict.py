import streamlit as st
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import preprocess 

# --- 1. KONFIGURATION & CSS ---
st.set_page_config(page_title="MNIST Projekt", layout="centered")

# CSS för att minska luft mellan rubriker och element
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    h1 { margin-bottom: 0rem !important; padding-bottom: 0rem !important; }
    .subtitle { margin-top: -0.4rem !important; color: #555; margin-bottom: 1rem; }
    hr { margin: 0.5rem 0 !important; }
    .stTabs [data-baseweb="tab-list"] { margin-bottom: -1rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("mnist_model_final_svc.joblib")

model = load_model()

# --- 2. RUBRIKER ---
st.title("MNIST-projekt")
st.markdown('<p class="subtitle">Kunskapskontroll 2 - Michael Broström</p>', unsafe_allow_html=True)

# --- 3. INPUT-FLIKAR ---
tab_draw, tab_upload = st.tabs(["✍️ Rita", "📁 Ladda upp"])

def perform_analysis(img_input):
    features, img_28 = preprocess.preprocess_image(img_input)
    scores = model.decision_function(features)[0]
    probs = np.exp(scores - np.max(scores)) / np.exp(scores - np.max(scores)).sum()
    pred = np.argmax(probs)
    conf = probs[pred]
    return pred, conf, img_28, probs

# --- 4. RITA-FLIK ---
with tab_draw:
    # Sida-vid-sida för bilderna
    col_canvas, col_machine = st.columns(2)
    
    with col_canvas:
        st.caption("1. Rita här")
        canvas_result = st_canvas(
            fill_color="white", stroke_width=18, stroke_color="black",
            background_color="white", height=280, width=280,
            drawing_mode="freedraw", key="canvas_live"
        )
    
    if canvas_result.image_data is not None:
        img_draw = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        # Analysera bara om användaren ritat något
        if np.mean(np.array(img_draw)) < 254:
            pred, conf, img_28, probs = perform_analysis(img_draw)
            
            with col_machine:
                st.caption("2. Maskinens vy (28x28)")
                # Tvingar maskinens vy att matcha canvas-storleken (280px)
                st.image(img_28, width=280)
            
            # --- PREDIKTION TÄTARE INPÅ ---
            # Vi använder Markdown för att få siffra och % på samma rad
            st.markdown(f"### Modellen gissar: **{pred}** &nbsp;&nbsp; <span style='color:green; font-size:1.2rem;'>({conf:.0%} säkerhet)</span>", unsafe_allow_html=True)
            
            # --- STAPELDIAGRAM TÄTARE ---
            fig, ax = plt.subplots(figsize=(10, 2)) # Ännu lägre höjd
            colors = ['#3498db'] * 10
            colors[pred] = '#f1c40f' 
            ax.bar(range(10), probs, color=colors)
            ax.set_xticks(range(10))
            ax.set_yticks([])
            ax.set_ylim(0, 1.1)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            with col_machine:
                # Skapar en tom vit box för att hålla layouten stabil
                st.image(Image.new("L", (280, 280), 255), width=280, caption="Väntar på indata...")

# --- 5. LADDA UPP-FLIK ---
with tab_upload:
    uploaded_file = st.file_uploader("Välj bild", type=["jpg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        col_orig, col_mach_up = st.columns(2)
        img_upload = Image.open(uploaded_file)
        pred, conf, img_28, probs = perform_analysis(img_upload)
        
        with col_orig:
            st.caption("Original")
            st.image(img_upload, width=280)
        with col_mach_up:
            st.caption("Maskinens vy")
            st.image(img_28, width=280)
            
        st.markdown(f"### Modellen gissar: **{pred}** &nbsp;&nbsp; <span style='color:green; font-size:1.2rem;'>({conf:.0%} säkerhet)</span>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(10), probs, color=['#3498db']*10)
        ax.patches[pred].set_color('#f1c40f')
        ax.set_xticks(range(10))
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)