# Détection d'armes

Cette application permet de détecter les armes dans une image ou une vidéo.
Elle utilise un modèle de détection d'objets pré-entraîné sur un jeu de données.
L'application est développée avec [Streamlit](https://streamlit.io/).

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
