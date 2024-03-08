import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

net = cv2.dnn.readNet("dataset/yolov3.weights", "dataset/yolov3_t.cfg")

# Définir les catégories de classe
classes = ['Gun', 'Knife', 'Rifle']

model = load_model("Notebook/inceptionv3_model.h5")

# Définir le titre et l'icône de la page
st.set_page_config(
    page_title="Vidéos - Détection d'armes",
    page_icon="🎥",
)

# Afficher le titre
st.write("# Vidéos")

# Ajouter le composant d'upload de fichier
upload_video = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi", "mov"])

# Définir les coordonnées du rectangle (x, y, largeur, hauteur)
#rectangle_coordinates = st.text_input("Coordonnées du rectangle (x, y, largeur, hauteur)", "100, 100, 100, 100")
rectangle_coordinates = ("100, 100, 100, 100")
rectangle_coordinates = [int(coord) for coord in rectangle_coordinates.split(",")]

frame_container = st.empty()

# Si une vidéo est uploadée
if upload_video is not None:
    # Créer une copie temporaire de la vidéo
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    with open(temp_video_path, "wb") as temp_video:
        temp_video.write(upload_video.read())

    # Utiliser MoviePy pour obtenir les frames de la vidéo
    video_clip = VideoFileClip(temp_video_path)

    # Afficher le nombre total de frames
    total_frames = int(video_clip.fps * video_clip.duration)
    st.write(f"Nombre total de frames : {total_frames}")

    # Lire la vidéo frame par frame
    for i in range(1, total_frames + 1):
        # Récupérer la frame
        frame = video_clip.get_frame(i / video_clip.fps)

        # Convertir la frame en UMat
        frame = cv2.UMat(np.array(frame))

        # Dessiner le rectangle sur la frame
        x, y, w, h = rectangle_coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Weapon", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"Frame {i}", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_container.image(frame.get(), channels="BGR")

    if video_clip:
        video_clip.close()

    # Supprimer la copie temporaire avec gestion d'erreur de permission
    try:
        os.remove(temp_video_path)
    except PermissionError:
        st.warning("Erreur de suppression du fichier temporaire. Il peut être utilisé par un autre processus.")
else:
    st.warning("Veuillez choisir une vidéo")
