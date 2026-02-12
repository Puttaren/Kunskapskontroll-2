import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import preprocess 

# --- 1. KONFIGURATION & STYLING ---
st.set_page_config(page_title="MNIST-projekt - kunskapskontroll 2", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stTitle { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("mnist_model_final_svc.joblib")

model = load_model()

# --- 2. INTRODUKTION ---
st.title("🔢 MNIST-projekt - kunskapskontroll 2")
st.markdown("""
    **Välkommen till "gissa och vinn"  
    - prediktering av ritade eller uppladdade bilder. 
            
    Om bilderna är brusiga bearbetas de lite extra innan modellen  
    får sätta tänderna i den.
""")

# --- 3. SIDEBAR ---
st.sidebar.header("Inställningar")
mode = st.sidebar.radio("Välj inmatning:", ("Rita siffra", "Ladda upp bild"))

if mode == "Rita siffra":
    stroke_width = st.sidebar.slider("Linjetjocklek:", 10, 25, 18)
else:
    st.sidebar.info("Two-Pass aktiverat för optimal uppskalning av foton.")

# --- 4. ANALYSFUNKTION MED TWO-PASS LOGIK ---
def analyze_and_display(img_input, use_two_pass=False):
    # PASS 1: Grundläggande tvätt och första centrering
    features, img_28 = preprocess.preprocess_image(img_input)
    
    if use_two_pass:
        # TWO-PASS: Vi tar den centrerade bilden, inverterar den till "Ljus vy" 
        # och matar in den igen för att få en perfekt beskärning.
        light_img = ImageOps.invert(img_28)
        features, img_28 = preprocess.preprocess_image(light_img)

    # Beräkna sannolikhet
    scores = model.decision_function(features)[0]
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    pred = np.argmax(probs)
    conf = probs[pred]

    # Panel-vy
    st.write("---")
    status_msg = " (Two-Pass optimerad)" if use_two_pass else ""
    st.subheader(f"Analysresultat: Modellen gissar på en {pred}{status_msg}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.caption("1. Original-indata")
        st.image(img_input, use_container_width=True)
        
    with col2:
        st.caption("2. Beskuren & Centrerad (Ljus vy)")
        # Detta är bilden vi "matade in igen" i Pass 2
        st.image(ImageOps.invert(img_28), use_container_width=True)
        
    with col3:
        st.caption(f"3. Maskinens vy (Pred: {pred})")
        st.image(img_28, use_container_width=True)
        
    with col4:
        st.caption(f"4. Sannolikhet (Gissning: {conf:.0%})")
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ['#3498db'] * 10
        colors[pred] = '#f1c40f'
        ax.bar(range(10), probs, color=colors)
        ax.set_xticks(range(10))
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# --- 5. LOGIK FÖR INMATNING ---
if mode == "Rita siffra":
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300, width=300,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Analysera teckning"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
            # Ritade bilder behöver bara ett pass
            analyze_and_display(img, use_two_pass=False)

else:
    uploaded_file = st.file_uploader("Välj bild...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        if st.button("Analysera uppladdning"):
            # Uppladdade bilder körs två varv för att hantera brus och skalning
            analyze_and_display(img, use_two_pass=False)