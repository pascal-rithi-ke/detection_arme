import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Images - Détection d'armes",
    page_icon="📸",
)

st.title("Images")

# Charger le modèle CNN pré-entraîné
@st.cache_resource
def load_pred_model():
    model = load_model('Notebook/model.h5')
    return model

model = load_pred_model()

# Définir les catégories de classe
classes = ['Gun', 'Knife', 'Rifle']

uploaded_image = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Prétraitement de l'image
    resized_image = cv2.resize(image, (150, 150))  # Redimensionner l'image pour correspondre à la taille d'entrée du modèle
    normalized_image = resized_image / 255.0  # Normaliser les valeurs des pixels dans la plage [0, 1]
    
    # Classification de l'image
    prediction = model.predict(np.expand_dims(normalized_image, axis=0))[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    confidence = prediction[predicted_class_index]
    
    # Affichage des résultats
    st.image(image, channels="BGR", caption="Image téléchargée")
    st.write(f"Prédiction : {predicted_class} avec une confiance de {confidence:.2f}")
