import streamlit as st
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import preprocess 
import scipy.ndimage as ndimage
import os
import time
import importlib

importlib.reload(preprocess) # Tvinga användning av senaste koden - no cache please!

# 1. KONFIGURATION OCH STIL
st.set_page_config(page_title="Sifferprediktering", layout="centered")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    h1 { margin-bottom: 0rem !important; padding-bottom: 0rem !important; }
    .subtitle { 
        margin-top: -0.4rem !important; 
        color: #555; 
        margin-bottom: 0.5rem; 
        font-size: 1.1rem; 
    }
    hr { margin: 0.5rem 0 !important; }
    .stRadio [data-baseweb="radio"] { padding-right: 20px; }
    div[data-testid="stFileUploaderDeleteBtn"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODELL-LADDNING
@st.cache_resource
def load_model():
    return joblib.load("mnist_svc_augmented_ultra_model.joblib")

model = load_model()

# 3. TTA-MOTOR
def tta_predict(features, model, n_variants=20):
    img_2d = features.reshape(28, 28)
    variants = [features.flatten()] 
    for _ in range(n_variants - 1):
        angle = np.random.uniform(-10, 10)
        dx, dy = np.random.uniform(-1.2, 1.2, size=2)
        v = ndimage.rotate(img_2d, angle, reshape=False, order=1, mode='constant', cval=0)
        v = ndimage.shift(v, [dy, dx], mode='constant', cval=0)
        variants.append(v.flatten())
    
    all_preds = model.predict(np.array(variants))
    counts = np.bincount(all_preds, minlength=10)
    probs = counts / n_variants 
    pred = np.argmax(counts)
    conf = probs[pred]
    return pred, conf, probs

# 4. ANALYSFUNKTION
def perform_analysis(img_input, is_upload=False): # Lägg till is_upload här
    # Skicka vidare flaggan till din preprocess-motor
    features, img_28, num_blobs, aspect_ratio, holes = preprocess.preprocess_image(img_input, is_upload=is_upload)
    pred, conf, probs = tta_predict(features, model, n_variants=20)

    original_pred = pred
    is_corrected = False
    
    if holes == 1 and pred == 5:
        pred = 6
        is_corrected = True
        conf = probs[6] 
        reasoning = f"Juryn röstade på {original_pred} ({probs[original_pred]:.0%}), men hålet bekräftar en 6:a."
    elif holes >= 2 and pred != 8:
        pred = 8
        is_corrected = True
        conf = probs[8]
        reasoning = f"Siffran har {holes} hål. Juryns val {original_pred} ersattes med 8."
    elif holes == 1 and pred == 1:
        possible_with_holes = [0, 4, 6, 8, 9]
        pred = possible_with_holes[np.argmax(probs[possible_with_holes])]
        is_corrected = True
        conf = probs[pred]
        reasoning = f"Etta med hål detekterad. Korrigerad till {pred}."
    else:
        reasoning = f"Modellen är {conf:.0%} säker på sin gissning ({pred})."

    return pred, conf, img_28, probs, num_blobs, aspect_ratio, is_corrected, original_pred, reasoning, holes

# --- HJÄLPFUNKTION FÖR FEEDBACK ---
def show_feedback_section(pred, img_28):
    st.divider()
    with st.expander("🛠️ Hjälp modellen att bli bättre"):
        st.write("Ange rätt siffra nedan för att spara bilden till framtida träning.")
        
        # Definiera den nya sökvägen (relativt till där predict.py körs)
        target_folder = os.path.join("notebooks", "collected_data")
        
        # Skapa mappen (inklusive föräldramappar om de saknas)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        correct_label = st.selectbox("Vilken siffra ritade du egentligen?", range(10), index=int(pred))
        
        if st.button("Spara bild för träning"):
            save_img = (img_28 * 255).astype(np.uint8) if img_28.max() <= 1.0 else img_28.astype(np.uint8)
            timestamp = int(time.time())
            
            # Skapa filnamnet med os.path.join för att slippa krångel med snedstreck
            filename = os.path.join(target_folder, f"label_{correct_label}_{timestamp}.png")
            
            Image.fromarray(save_img).save(filename)
            st.success(f"Sparad i: {filename}")

# 5. GRÄNSSNITT
st.title("MNIST-projekt")
st.markdown('<p class="subtitle">Prediktering av siffror.  \nEtt projekt av Michael Broström för Kunskapskontroll 2</p>', unsafe_allow_html=True)
st.caption("Robust sifferigenkänning med grundlig modellering och hantering av diverse problem med siffror.  \nVänligen försök inte att 'lura' modellen. Rita/ladda upp rimliga siffror för att testa den maskininlärda  \nmodellen snarare än bildbearbetningen. ")

mode = st.radio("Läge:", ["✍️ Rita", "📁 Ladda upp"], horizontal=True, label_visibility="collapsed")

if mode == "✍️ Rita":
    st.session_state.last_upload = None
    col_canvas, col_machine = st.columns(2)
    
    with col_canvas:
        st.caption("1. Rita här")
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = "canvas_draw"
        canvas_result = st_canvas(
            fill_color="white", stroke_width=12, stroke_color="black",
            background_color="white", height=280, width=280,
            drawing_mode="freedraw", key=st.session_state.canvas_key
        )
        if st.button("Töm ritytan"):
            st.session_state.canvas_key = f"canvas_{np.random.randint(0, 1000)}"
            st.rerun()
    
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        img_draw = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        st.session_state.last_draw = perform_analysis(img_draw, is_upload=False)

    if "last_draw" in st.session_state and st.session_state.last_draw:
        pred, conf, img_28, probs, num_blobs, aspect_ratio, is_corrected, original_pred, reasoning, holes = st.session_state.last_draw
        
        with col_machine:
            st.caption("2. Maskinens vy (28x28)")
            st.image(img_28, width=280)

        if num_blobs > 1:
            st.warning(f"⚠️ Hittade {num_blobs} figurer.")
        
        st.markdown(f"### Resultat: **{pred}** &nbsp;&nbsp; <span style='color:green; font-size:1.2rem;'>({conf:.0%} jury-enighet)</span>", unsafe_allow_html=True)
        
        with st.expander("🔍 Visa teknisk analys"):
            st.write(f"**Beslut:** {reasoning}")
            st.write(f"**Detaljer:** Hål: {holes} | Figurer: {num_blobs} | Ratio: {aspect_ratio:.2f}")

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(10), probs, color=['#3498db']*10)
        ax.patches[pred].set_color('#f1c40f')
        ax.set_xticks(range(10))
        ax.set_yticks([])
        st.pyplot(fig)
        
        # Visa feedback-sektionen här
        show_feedback_section(pred, img_28)

else:
    st.session_state.last_draw = None 
    uploaded_file = st.file_uploader("Välj bild", type=["jpg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        img_upload = Image.open(uploaded_file)
        st.session_state.last_upload = (perform_analysis(img_upload, is_upload=True), img_upload)
    if "last_upload" in st.session_state and st.session_state.last_upload:
        (pred, conf, img_28, probs, num_blobs, aspect_ratio, is_corrected, original_pred, reasoning, holes), original_img = st.session_state.last_upload
        
        col_orig, col_mach_up = st.columns(2)
        with col_orig:
            st.caption("Original")
            st.image(original_img, width=280)
        with col_mach_up:
            st.caption("Maskinens vy")
            st.image(img_28, width=280)

        st.markdown(f"### Resultat: **{pred}** &nbsp;&nbsp; <span style='color:green; font-size:1.2rem;'>({conf:.0%} jury-enighet)</span>", unsafe_allow_html=True)
        
        with st.expander("🔍 Visa teknisk analys"):
            st.write(f"**Beslut:** {reasoning}")
            st.write(f"**Detaljer:** Hål: {holes} | Ratio: {aspect_ratio:.2f}")

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(10), probs, color=['#3498db']*10)
        ax.patches[pred].set_color('#f1c40f')
        ax.set_xticks(range(10))
        ax.set_yticks([])
        st.pyplot(fig)

        # Visa feedback-sektionen här
        show_feedback_section(pred, img_28)