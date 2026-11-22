"""Application Streamlit : segmentation avec 4 modèles (K-means / CAH)."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Segmentation : 4 modèles", page_icon="📊", layout="centered")

DOSSIER = Path(__file__).parent / "models"

MODELES = {
    "Clients télécom": {
        "fichier": "customer_clustering.joblib",
        "titre": "Segmentation des clients",
        "intro": "Saisis les caractéristiques d'un client pour connaître son groupe.",
        "pluriel": "ce client",
    },
    "Graines de blé": {
        "fichier": "wheat_clustering.joblib",
        "titre": "Segmentation des graines de blé",
        "intro": "Saisis les mesures d'une graine pour connaître son groupe.",
        "pluriel": "cette graine",
    },
    "Qualité du vin": {
        "fichier": "wine_clustering.joblib",
        "titre": "Segmentation des vins (Wine Quality)",
        "intro": "Saisis les mesures physico-chimiques d'un vin pour connaître son groupe.",
        "pluriel": "ce vin",
    },
    "Patients (Heart Disease)": {
        "fichier": "heart_clustering.joblib",
        "titre": "Segmentation des patients (Heart Disease)",
        "intro": "Saisis les informations cliniques d'un patient pour connaître son groupe.",
        "pluriel": "ce patient",
    },
}

# Libellés lisibles pour les variables du jeu Heart Disease
LIBELLES_HEART = {
    "age": "Âge (années)",
    "sex": "Sexe",
    "cp": "Type de douleur thoracique (1 à 4)",
    "trestbps": "Pression artérielle au repos (mm Hg)",
    "chol": "Cholestérol (mg/dl)",
    "fbs": "Glycémie à jeun > 120 mg/dl",
    "restecg": "ECG au repos (0 à 2)",
    "thalach": "Fréquence cardiaque maximale",
    "exang": "Angine induite par l'effort",
    "oldpeak": "Dépression du segment ST",
    "slope": "Pente du segment ST (1 à 3)",
    "ca": "Vaisseaux colorés (0 à 3)",
    "thal": "Test au thallium (3 normal, 6 défaut fixe, 7 défaut réversible)",
}
FORMAT_BINAIRE = {
    "sex": {0: "Femme", 1: "Homme"},
    "fbs": {0: "Non", 1: "Oui"},
    "exang": {0: "Non", 1: "Oui"},
}


@st.cache_resource
def charger_modele(fichier):
    return joblib.load(DOSSIER / fichier)


def predire(modele, valeurs):
    """Renvoie (classe, distances) pour un dictionnaire {colonne: valeur}."""
    colonnes = modele["colonnes"]
    ligne = pd.DataFrame([[valeurs[c] for c in colonnes]], columns=colonnes)
    z = modele["preprocess"].transform(ligne)
    distances = cdist(z, modele["centroides"])[0]
    return modele["classes"][int(distances.argmin())], distances


def formulaire_numerique(modele):
    """Un champ par variable, pré-rempli avec la médiane."""
    valeurs = {}
    cols = st.columns(2)
    for i, c in enumerate(modele["colonnes"]):
        mini, maxi, med = modele["minimum"][c], modele["maximum"][c], modele["mediane"][c]
        with cols[i % 2]:
            valeurs[c] = st.number_input(
                c, value=float(med), help=f"Valeurs observées : de {mini:.4g} à {maxi:.4g}", key=f"num_{c}"
            )
        if not (mini <= valeurs[c] <= maxi):
            st.caption(f"⚠️ {c} : valeur hors de la plage observée ({mini:.4g} à {maxi:.4g}).")
    return valeurs


def formulaire_heart(modele):
    valeurs = {}
    cols = st.columns(2)
    for i, c in enumerate(modele["colonnes"]):
        label = LIBELLES_HEART.get(c, c)
        with cols[i % 2]:
            if c in modele["numeriques"]:
                mini, maxi, med = modele["numeriques"][c]
                valeurs[c] = st.number_input(
                    label, value=float(med), help=f"Valeurs observées : de {mini:.4g} à {maxi:.4g}", key=f"h_{c}"
                )
            elif c in modele["binaires"]:
                fmt = FORMAT_BINAIRE.get(c, {0: "Non", 1: "Oui"})
                valeurs[c] = st.selectbox(label, [0, 1], format_func=lambda v, f=fmt: f[v], key=f"h_{c}")
            else:
                options = modele["categories"][c]
                valeurs[c] = st.selectbox(label, options, format_func=lambda v: f"{v:g}", key=f"h_{c}")
    return valeurs


# ----------------------------------------------------------------- interface
st.sidebar.title("📊 Segmentation")
choix = st.sidebar.radio("Choisis un modèle", list(MODELES))
info = MODELES[choix]

st.title(info["titre"])
st.write(info["intro"])

if not (DOSSIER / info["fichier"]).exists():
    st.error(f"Fichier modèle introuvable : models/{info['fichier']}. Copie-le dans le dossier models/.")
    st.stop()

modele = charger_modele(info["fichier"])
st.caption(f"Modèle utilisé : **{modele['nom_modele']}** · Source : {modele['source']}")

if choix.startswith("Patients"):
    st.warning("Projet pédagogique : ce n'est pas un outil de diagnostic médical.")

with st.form(f"form_{choix}"):
    if "categories" in modele:
        valeurs = formulaire_heart(modele)
    else:
        valeurs = formulaire_numerique(modele)
    envoye = st.form_submit_button("Prédire le groupe")

if envoye:
    classe, distances = predire(modele, valeurs)
    nom = modele["noms_classes"][int(classe)]
    st.success(f"Résultat : {info['pluriel']} appartient au groupe **{nom}**.")
    with st.expander("Voir les distances aux groupes"):
        tableau = pd.DataFrame(
            {"Groupe": [modele["noms_classes"][int(c)] for c in modele["classes"]], "Distance": distances}
        ).sort_values("Distance")
        st.dataframe(tableau, hide_index=True)
