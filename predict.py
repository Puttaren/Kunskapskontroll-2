import streamlit as st
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import preprocess 
import scipy.ndimage as ndimage

# Skapa ett kompakt format så allt kan få plats på en sida
st.set_page_config(page_title="MNIST Projekt", layout="centered")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    h1 { margin-bottom: 0rem !important; padding-bottom: 0rem !important; }
        
    .subtitle { 
        margin-top: -0.4rem !important; 
        color: #555; 
        margin-bottom: 1.5rem; 
        font-size: 1.1rem; 
    }
    
    hr { margin: 0.5rem 0 !important; }
    
    /* Radiomeny med flikar */
    .stRadio [data-baseweb="radio"] { padding-right: 20px; }

    /* Döljer krysset/radera-knappen helt för användaren */
    div[data-testid="stFileUploaderDeleteBtn"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Läs in modellen för prediktion
@st.cache_resource
def load_model():
    return joblib.load("mnist_svc_deskew_agument_model.joblib")

model = load_model()

# TTA-motor som skapar 20 varianter med små geometriska transformationer. 
# Därefter får modellen analysera dem och fatta ett majoritetsbeslut.
def tta_predict(features, model, n_variants=20):
    img_2d = features.reshape(28, 28)
    variants = [features.flatten()] 
    
    for _ in range(n_variants - 1):
        angle = np.random.uniform(-4, 4)
        dx, dy = np.random.uniform(-0.8, 0.8, size=2)
        
        v = ndimage.rotate(img_2d, angle, reshape=False, order=1, mode='constant', cval=0)
        v = ndimage.shift(v, [dy, dx], mode='constant', cval=0)
        variants.append(v.flatten())
    
    # Juryn talar!
    all_preds = model.predict(np.array(variants))
    
    # Räkna röster per klass (0-9)
    counts = np.bincount(all_preds, minlength=10)
    probs = counts / n_variants # Andel röster per siffra
    pred = np.argmax(counts)
    conf = probs[pred]
    
    return pred, conf, probs

# Rubriker
st.title("MNIST-projekt")
st.markdown('<p class="subtitle">Kunskapskontroll 2 - Michael Broström</p>', unsafe_allow_html=True)

# Radio används som "meny" för att ge nollställning vid växling
mode = st.radio("Läge:", ["✍️ Rita", "📁 Ladda upp"], horizontal=True, label_visibility="collapsed")

def perform_analysis(img_input):
    # 1. Preprocessing (inkl. din nya deskew-logik)
    features, img_28, num_blobs, aspect_ratio = preprocess.preprocess_image(img_input)
    
    # 2. TTA-prediktion (istället för decision_function)
    # Detta ger oss ett mer robust svar baserat på 20 analyser
    pred, conf, probs = tta_predict(features, model, n_variants=20)

    # Returnera även statistiken till session_state
    return pred, conf, img_28, probs, num_blobs, aspect_ratio


# Rita egen bild
if mode == "✍️ Rita":
    # Nollställ uppladdningsminnet vid växling
    st.session_state.last_upload = None
    
    col_canvas, col_machine = st.columns(2)
    
    with col_canvas:
        st.caption("1. Rita här")
        
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = "canvas_draw"

        canvas_result = st_canvas(
            fill_color="white", stroke_width=18, stroke_color="black",
            background_color="white", height=280, width=280,
            drawing_mode="freedraw", key=st.session_state.canvas_key
        )

        # Knapp som nollställer rutan men behåller analysen i minnet
        if st.button("Töm ritytan"):
            st.session_state.canvas_key = f"canvas_{np.random.randint(0, 1000)}"
            st.rerun()
    
    # Uppdatera bara om rutan faktiskt innehåller objekt
    has_drawing = canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0
    
    if has_drawing:
        img_draw = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        # Spara i session_state för att behålla resultatet vid "sudda" 
        st.session_state.last_draw = perform_analysis(img_draw)

    # Visa resultatet om det finns i minnet (även om rutan nyss tömts) 
    if "last_draw" in st.session_state and st.session_state.last_draw:
        # Hämta in även blobbar och aspektförhållande
        pred, conf, img_28, probs, num_blobs, aspect_ratio = st.session_state.last_draw
        
        with col_machine:
            st.caption("2. Maskinens vy (28x28)")
            st.image(img_28, width=280)

        # Ge användaren feedback baserat på analysen ("lurendrejeri" och 1/9-problematik).
        if num_blobs > 1:
            st.warning(f"⚠️ Jag hittade {num_blobs} figurer. Rita bara en siffra för bäst resultat.")
        
        if pred == 9 and aspect_ratio < 0.35:
            st.info("💡 Figuren är väldigt smal för en 9:a och kan eventuellt vara en 1:a med serif")
        
        st.markdown(f"### Modellen gissar: **{pred}** &nbsp;&nbsp; <span style='color:green; font-size:1.2rem;'>({conf:.0%} säkerhet)</span>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(10), probs, color=['#3498db']*10)
        ax.patches[pred].set_color('#f1c40f')
        ax.set_xticks(range(10))
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)
    else:
        # Se till att det är tomt vid start
        pass

# Uppladdning
else:
    # Rensar gammalt ritminne så att vi får en tom sida 
    st.session_state.last_draw = None 
    
    uploaded_file = st.file_uploader("Välj bild", type=["jpg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        img_upload = Image.open(uploaded_file)
        # Spara både analys och bild i session_state för persistens 
        st.session_state.last_upload = (perform_analysis(img_upload), img_upload)

    if "last_upload" in st.session_state and st.session_state.last_upload:
        # --- Hämta även blobbar och aspektförhållande
        (pred, conf, img_28, probs, num_blobs, aspect_ratio), original_img = st.session_state.last_upload
        
        col_orig, col_mach_up = st.columns(2)
        with col_orig:
            st.caption("Original")
            st.image(original_img, width=280)
        with col_mach_up:
            st.caption("Maskinens vy")
            st.image(img_28, width=280)

        # Här kommer feedback-meddelande för uppladdade bilder (med min nya logik för extra kontroll)
        if num_blobs > 1:
            st.warning(f"⚠️ Bilden innehåller {num_blobs} separata delar. MNIST-modeller fungerar bäst med en siffra.")
            
        if pred == 9 and aspect_ratio < 0.35:
            st.info("💡 Den här bilden är ovanligt smal för att vara en 9:a. Kan vara en etta med serif.")
            
        st.markdown(f"### Modellen gissar: **{pred}** &nbsp;&nbsp; <span style='color:green; font-size:1.2rem;'>({conf:.0%} säkerhet)</span>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(10), probs, color=['#3498db']*10)
        ax.patches[pred].set_color('#f1c40f')
        ax.set_xticks(range(10))
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)