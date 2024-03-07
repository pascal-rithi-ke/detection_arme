import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Images - Détection d'armes",
    page_icon="📸",
)

st.write("# Images")

uploaded_image = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, channels="BGR", caption="Image téléchargée")