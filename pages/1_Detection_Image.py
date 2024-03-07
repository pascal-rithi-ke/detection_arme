import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

model = load_model("./Notebook/model_inception.h5")

st.set_page_config(
    page_title="Images - Détection d'armes",
    page_icon="📸",
)

st.title("Images")

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img):
    processed_image = preprocess_image(img)
    prediction = model.predict(processed_image)
    st.image(img, caption="Image à analyser", width=300)
    st.write("Classification de l'image...")
    st.write("")
    predicted_class = np.argmax(prediction)
    return predicted_class

uploaded_image = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = np.array(image)
    predicted_class = predict_image(image)
    st.write(f"predicted_class : {predicted_class}")