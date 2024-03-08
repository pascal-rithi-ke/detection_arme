import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Charger le modèle YOLO
net = cv2.dnn.readNet("dataset_yolo/yolov3.weights", "dataset_yolo/yolov3.cfg")

# Configurer les paramètres d'entraînement
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Charger les classes
classes = []
with open("dataset_yolo/classes.txt", "r") as f:
    classes = [line.strip() for line in f]

# Définir le titre et l'icône de la page
st.set_page_config(
    page_title="Vidéos - Détection d'armes",
    page_icon="🎥",
)

# Afficher le titre
st.write("# Vidéos")

# Ajouter le composant d'upload de fichier
upload_video = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi", "mov"])

# Si un fichier a été uploadé
if upload_video is not None:
    # Enregistrer le fichier uploader dans un répertoire temporaire
    temp_dir = tempfile.TemporaryDirectory()
    video_path = os.path.join(temp_dir.name, "uploaded_video.mp4")
    with open(video_path, "wb") as video_file:
        video_file.write(upload_video.read())

    # Lire la vidéo à l'aide d'OpenCV
    video_capture = cv2.VideoCapture(video_path)

    # Définir la taille de la fenêtre de la vidéo
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))

    # Créer un élément Streamlit pour afficher la vidéo
    video_placeholder = st.empty()

    # Lire la vidéo frame par frame
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Créer un blob à partir de l'image
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        layer_names = net.getUnconnectedOutLayersNames()
        detections = net.forward(layer_names)

        # Post-traitement des détections
        for obj in detections[0]:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                # Dessiner un rectangle autour de l'objet détecté
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[class_id]}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Afficher la vidéo dans Streamlit avec le rectangle superposé
        video_placeholder.image(frame, channels="BGR")

    # Libérer la ressource vidéo
    video_capture.release()
    # Supprimer le répertoire temporaire
    temp_dir.cleanup()