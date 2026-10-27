# ==============================================================
# Application Streamlit : reconnaissance de fleurs en temps réel
# Modèle : MobileNetV2 + tête personnalisée (modele_fleurs.h5)
# Lancement : python -m streamlit run streamlit_app.py
# ==============================================================

import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

TAILLE = 160
EMOJIS = {"marguerite": "🌼", "pissenlit": "🌻", "rose": "🌹",
          "tournesol": "🌻", "tulipe": "🌷"}

st.set_page_config(page_title="Reconnaissance de fleurs", page_icon="🌷")


# --------------------------------------------------------------
# Chargement du modèle (une seule fois, gardé en mémoire)
# --------------------------------------------------------------
@st.cache_resource
def charger_modele():
    modele = keras.models.load_model("modele_fleurs.h5")
    with open("classes_fleurs.json", encoding="utf-8") as f:
        classes = json.load(f)
    return modele, classes


def pretraiter(image):
    """Prépare une image téléversée exactement comme à l'entraînement."""
    image = image.convert("RGB")                       # PNG avec transparence, noir et blanc...
    image = image.resize((TAILLE, TAILLE))             # 160 x 160 pixels
    tableau = np.array(image, dtype="float32")
    tableau = keras.applications.mobilenet_v2.preprocess_input(tableau)   # pixels entre -1 et 1
    return np.expand_dims(tableau, axis=0)            # lot d'une seule image : (1, 160, 160, 3)


# --------------------------------------------------------------
# Interface
# --------------------------------------------------------------
st.title("🌷 Reconnaissance de fleurs")
st.write("Téléversez la photo d'une fleur : le modèle indique s'il s'agit d'une "
         "**marguerite**, d'un **pissenlit**, d'une **rose**, d'un **tournesol** "
         "ou d'une **tulipe**.")

try:
    modele, classes = charger_modele()
except (OSError, FileNotFoundError):
    st.error("Modèle introuvable. Lancez d'abord `sauvegarder_modele_fleurs.py` "
             "pour créer modele_fleurs.h5 dans ce dossier.")
    st.stop()

fichier = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png", "webp"])

if fichier is None:
    st.info("En attente d'une image.")
    st.stop()

image = Image.open(fichier)
colonne_image, colonne_resultat = st.columns(2)

with colonne_image:
    st.image(image, caption="Image téléversée", width=300)

with colonne_resultat:
    with st.spinner("Analyse en cours..."):
        probabilites = modele.predict(pretraiter(image), verbose=0)[0]

    indice = int(np.argmax(probabilites))
    classe = classes[indice]
    confiance = float(probabilites[indice])

    st.subheader("Prédiction")
    st.markdown(f"## {EMOJIS.get(classe, '')} {classe.capitalize()}")
    st.metric("Confiance", f"{confiance:.1%}")
    st.progress(confiance)

    if confiance < 0.5:
        st.warning("Confiance faible : l'image est peut-être floue, ou ce n'est "
                   "pas l'une des 5 fleurs connues du modèle.")

st.subheader("Probabilités par classe")
tableau = pd.DataFrame({"Probabilité": probabilites}, index=classes)
st.bar_chart(tableau)

with st.expander("Détail des probabilités"):
    st.dataframe(tableau.sort_values("Probabilité", ascending=False)
                 .style.format({"Probabilité": "{:.2%}"}))
