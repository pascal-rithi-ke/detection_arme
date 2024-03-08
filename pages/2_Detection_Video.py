import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import os
import cv2
import numpy as np

net = cv2.dnn.readNet("dataset/yolov3.weights", "dataset/yolov3_t.cfg")
classes = ["Weapons"]  # Mettez vos classes dans une liste

# Définir le titre et l'icône de la page
st.set_page_config(
    page_title="Vidéos - Détection d'armes",
    page_icon="🎥",
)

# Afficher le titre
st.write("# Vidéos")

# Ajouter le composant d'upload de fichier
upload_video = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi", "mov"])

# Définir la confiance minimale pour l'affichage de la détection
min_confidence = st.slider("Confiance minimale", 0.0, 1.0, 0.5)

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

    # Ajouter un slider pour choisir le nombre de frames à afficher
    #num_frames = st.slider("Nombre de frames à afficher", 1, total_frames, 1)

    # Créer un élément Streamlit pour afficher la vidéo
    video_placeholder = st.empty()

    for i in range(total_frames):
        # Récupérer la frame
        frame = video_clip.get_frame(i / video_clip.fps)

        # Effectuer la détection d'objet
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Process YOLO output
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > min_confidence:
                    # Extract detection details
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Draw a box and label on the frame
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    # Get the class name from the list
                    class_name = classes[class_id]

                    # Add class name to label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Afficher la vidéo dans Streamlit avec le rectangle superposé
        video_placeholder.image(frame, channels="RGB", caption=f"Frame {i + 1}")

    if video_clip:
        video_clip.close()

    # Supprimer la copie temporaire avec gestion d'erreur de permission
    try:
        os.remove(temp_video_path)
    except PermissionError:
        st.warning("Erreur de suppression du fichier temporaire. Il peut être utilisé par un autre processus.")
else:
    st.warning("Veuillez choisir une vidéo")
