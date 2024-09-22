# Détection d'armes

Ce projet a pour but de détecter des armes dans des images et des vidéos. Il utilise un réseau de neurones convolutif (CNN) pour la classification d'images et YOLO pour la détection d'objets en temps réel.

![Exemple](./assets/video_detect_gun.gif)

## Fonctionnalités

- Classification d'images
- Détection d'objets en temps réel
- Interface utilisateur avec Streamlit

## Installation

Avant de lancer l'application, assurez-vous que vous avez Python (minimum 3.8) d'installé.

Suivez ces étapes pour installer et configurer votre environnement :

```bash
git clone https://github.com/pascal-rithi-ke/detection_arme.git
cd votre_projet
```

Créez et activez un environnement virtuel (optionnel) :

```bash
python -m venv env
env\Scripts\activate
```

Installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## Démarrage de l'application

Pour lancer l'application Streamlit, utilisez la commande suivante :

```bash
streamlit run Accueil.py
```

Ouvrez votre navigateur et allez à l'adresse http://localhost:8501 pour voir l'application.

Pour quitter, appuyez sur `Ctrl + C` dans votre terminal. Vous pouvez quitter l'environnement virtuel avec la commande `deactivate`.

PS: Le dossier " Weapons.v2i.multiclass " est sur teams en format .ZIP
