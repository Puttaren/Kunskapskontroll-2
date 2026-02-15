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
    # MIN KOMMENTAR: Vi laddar den tunga modellen som matchar din 99.32% träning
    return joblib.load("mnist_svc_deskew_agument_model.joblib")

model = load_model()

# TTA-motor som skapar 20 varianter med små geometriska transformationer. 
# Därefter får modellen analysera dem och fatta ett majoritetsbeslut.
def tta_predict(features, model, n_variants=20):
    # MIN KOMMENTAR: Vi skapar variationer för att se om t.ex. en 5:a blir en 0:a i vissa vinklar
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
    # 1. Hämta all data från din nya preprocess
    features, img_28, num_blobs, aspect_ratio, holes = preprocess.preprocess_image(img_input)
    
    # 2. Få modellens gissning via TTA
    pred, conf, probs = tta_predict(features, model, n_variants=20)
    
    # 3. STORS LÄGGAN: Logisk korrigering (Heuristik)
    original_pred = pred
    is_corrected = False
    
    # Fall 1: Uppenbar sexa som tolkas som femma
    if holes == 1 and pred == 5:
        pred = 6
        is_corrected = True
        
    # Fall 2: Åtta som tolkas som något annat
    elif holes >= 2 and pred != 8:
        pred = 8
        is_corrected = True
        
    # Fall 3: Nolla/Sexa/Nia som tolkas som en etta
    elif holes == 1 and pred == 1:
        possible_with_holes = [0, 4, 6, 8, 9]
        # Välj den siffra med hål som juryn röstade mest på
        pred = possible_with_holes[np.argmax(probs[possible_with_holes])]
        is_corrected = True

    return pred, conf, img_28, probs, num_blobs, aspect_ratio, is_corrected, original_pred

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
            fill_color="white", stroke_width=12, stroke_color="black",
            background_color="white", height=280, width=280,
            drawing_mode="freedraw", key=st.session_state.canvas_key
        )

        if st.button("Töm ritytan"):
            st.session_state.canvas_key = f"canvas_{np.random.randint(0, 1000)}"
            st.rerun()
    
    has_drawing = canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0
    
    if has_drawing:
        img_draw = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        st.session_state.last_draw = perform_analysis(img_draw)

    if "last_draw" in st.session_state and st.session_state.last_draw:
        # MIN KOMMENTAR: Packar upp 8 värden för att hantera korrigeringen
        pred, conf, img_28, probs, num_blobs, aspect_ratio, is_corrected, original_pred = st.session_state.last_draw
        
        with col_machine:
            st.caption("2. Maskinens vy (28x28)")
            st.image(img_28, width=280)

        # Feedback-logik
        if num_blobs > 1:
            st.warning(f"⚠️ Jag hittade {num_blobs} figurer. Rita bara en siffra.")
        
        if is_corrected:
            st.success(f"✅ **Logisk korrigering:** Juryn trodde {original_pred}, men topologin (hål i siffran) bekräftar att det är en **{pred}**.")
        
        st.markdown(f"### Resultat: **{pred}** &nbsp;&nbsp; <span style='color:green; font-size:1.2rem;'>({conf:.0%} jury-enighet)</span>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(10), probs, color=['#3498db']*10)
        ax.patches[pred].set_color('#f1c40f')
        ax.set_xticks(range(10))
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)

# Uppladdning
else:
    st.session_state.last_draw = None 
    uploaded_file = st.file_uploader("Välj bild", type=["jpg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        img_upload = Image.open(uploaded_file)
        st.session_state.last_upload = (perform_analysis(img_upload), img_upload)

    if "last_upload" in st.session_state and st.session_state.last_upload:
        # MIN KOMMENTAR: Samma 8 värden här för konsekvens
        (pred, conf, img_28, probs, num_blobs, aspect_ratio, is_corrected, original_pred), original_img = st.session_state.last_upload
        
        col_orig, col_mach_up = st.columns(2)
        with col_orig:
            st.caption("Original")
            st.image(original_img, width=280)
        with col_mach_up:
            st.caption("Maskinens vy")
            st.image(img_28, width=280)

        if is_corrected:
            st.success(f"✅ **Logisk korrigering:** Bilden analyserades som {original_pred}, men topologin tvingade fram en **{pred}**.")
            
        st.markdown(f"### Resultat: **{pred}** &nbsp;&nbsp; <span style='color:green; font-size:1.2rem;'>({conf:.0%} jury-enighet)</span>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(10), probs, color=['#3498db']*10)
        ax.patches[pred].set_color('#f1c40f')
        ax.set_xticks(range(10))
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)