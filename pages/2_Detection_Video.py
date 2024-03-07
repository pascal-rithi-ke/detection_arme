import streamlit as st
import cv2
import numpy as np

net = cv2.dnn.readNet("dataset/yolov3.weights", "dataset/yolov3.cfg")

classes = ["Knife", "Pistol", "Rifle"]

# Définir le titre et l'icône de la page
st.set_page_config(
    page_title="Vidéos - Détection d'armes",
    page_icon="🎥",
)

# Afficher le titre
st.write("# Vidéos")

# Ajouter le composant d'upload de fichier
upload_video = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi", "mov"])

# Fonction pour effectuer la détection sur chaque frame
def perform_detection(frame):
    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process YOLO output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Extract detection details
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Draw a box and label on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Afficher la vidéo avec les détections
if upload_video is not None:
    video_file = st.video(upload_video)
    video_capture = cv2.VideoCapture(upload_video.name)  # Utiliser l'attribut filename
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = perform_detection(frame)
        st.image(frame, channels="BGR")
    video_capture.release()
    
# Afficher un message d'erreur si aucune vidéo n'est chargée
else:
    st.warning("Veuillez choisir une vidéo")