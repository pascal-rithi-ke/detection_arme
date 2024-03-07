import streamlit as st
import numpy as np 
import cv2

st.set_page_config(
    page_title="Accueil - Détection d'armes",
    page_icon="🔫",
)

st.title("Détection d'armes")

st.write("Bienvenue sur l'application de détection d'armes. Vous pouvez choisir de détecter des armes dans des images ou dans des vidéos.")
st.write("Pour commencer, veuillez choisir une option dans le menu de gauche.")

st.write("Pour plus d'informations, veuillez consulter le [GitHub](https://github.com/pascal-rithi-ke/detection_arme).")