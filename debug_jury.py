import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import predict
import preprocess
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="TTA Jury Debugger", layout="wide")
st.title("🕵️‍♂️ Juryns vittnesmål")
st.write("Använd denna sida för att se de 20 varianter som modellen faktiskt ser.")

# Skapa ritytan
canvas_result = st_canvas(
    fill_color="white", stroke_width=18, stroke_color="black",
    background_color="white", height=280, width=280, drawing_mode="freedraw"
)

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    
    # Kör din befintliga preprocess
    features, img_28, blobs, ratio = preprocess.preprocess_image(img)
    
    # Replikera TTA-logiken från din predict.py manuellt för att få ut bilderna
    img_2d = features.reshape(28, 28)
    variants = [img_2d]
    
    import scipy.ndimage as ndimage
    for _ in range(19):
        angle = np.random.uniform(-4, 4)
        dx, dy = np.random.uniform(-0.8, 0.8, size=2)
        v = ndimage.rotate(img_2d, angle, reshape=False, order=1, mode='constant', cval=0)
        v = ndimage.shift(v, [dy, dx], mode='constant', cval=0)
        variants.append(v)

    # Visa resultatet
    st.subheader("Resultat")
    cols = st.columns(10)
    for i, var in enumerate(variants):
        # Gör prediktion på just denna variant
        p = predict.model.predict(var.reshape(1, -1))[0]
        var_clipped = np.clip(var, 0, 1)
        
        cols[i % 10].image(var_clipped, caption=f"Gissning: {p}", use_container_width=True)

    # Sammanfattning
    all_preds = predict.model.predict(np.array([v.flatten() for v in variants]))
    counts = np.bincount(all_preds, minlength=10)
    st.write(f"### Slutgiltigt beslut: {np.argmax(counts)}")
    st.write(f"Röstfördelning: {counts}")
