"""Tableau de bord AeroGuard (interface en anglais) : état de la flotte et historique par moteur."""

import altair as alt
import pandas as pd
import streamlit as st

from aeroguard.decision import SEUIL_AE, SEUIL_XGB
from aeroguard.tableau import (
    COULEURS,
    LIBELLES,
    MESSAGES,
    ClientFlotte,
    compter_niveaux,
    tableau_flotte,
    tableau_historique,
)

st.set_page_config(page_title="AeroGuard — Fleet", page_icon="✈️", layout="wide")
st.title("AeroGuard — Fleet predictive maintenance")

url = st.sidebar.text_input("API address", "http://127.0.0.1:8000")
automatique = st.sidebar.toggle("Auto-refresh (5 s)", value=False)
st.sidebar.caption("Status: green = OK, orange = watch, red = urgent.")


@st.cache_resource
def obtenir_client(adresse):
    return ClientFlotte(adresse)


def couleur_statut(valeur):
    niveau = {libelle: cle for cle, libelle in LIBELLES.items()}.get(valeur)
    return f"background-color: {COULEURS[niveau]}; color: white" if niveau else ""


def graphique(donnees, colonne, seuil, titre):
    """Courbe d'un indicateur, points colorés par statut et seuil en pointillés."""
    base = alt.Chart(donnees).encode(x=alt.X("cycle:Q", title="Cycle (flight)"))
    ligne = base.mark_line(color="#999999").encode(y=alt.Y(f"{colonne}:Q", title=titre))
    points = base.mark_circle(size=45).encode(
        y=f"{colonne}:Q",
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=[LIBELLES[n] for n in COULEURS], range=list(COULEURS.values())),
            legend=alt.Legend(title="Status"),
        ),
        tooltip=["cycle", colonne, "status"],
    )
    regle = (
        alt.Chart(pd.DataFrame({"seuil": [seuil]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y="seuil:Q")
    )
    return (ligne + points + regle).properties(height=230)


@st.fragment(run_every="5s" if automatique else None)
def afficher():
    client = obtenir_client(url)
    if not client.disponible():
        st.error(f"The API is not responding at {url}. Start it with uvicorn, then refresh.")
        return
    flotte = client.flotte()
    if not flotte:
        st.info("No flight received yet: run scripts/remplir_flotte.py.")
        return

    comptes = compter_niveaux(flotte)
    cases = st.columns(4)
    cases[0].metric("Engines monitored", len(flotte))
    for case, niveau in zip(cases[1:], ("URGENT", "A_SURVEILLER", "OK"), strict=True):
        case.metric(LIBELLES[niveau], comptes[niveau])

    gauche, droite = st.columns([2, 3])
    tableau = tableau_flotte(flotte)
    with gauche:
        st.subheader("Fleet")
        st.caption("Most urgent engines first.")
        vue = tableau.drop(columns="niveau").style.map(couleur_statut, subset=["Status"])
        st.dataframe(vue, hide_index=True)
    with droite:
        moteur = st.selectbox("Engine to inspect", tableau["Engine"], key="moteur")
        brut = client.historique(moteur)
        if not brut:
            st.warning("No history available for this engine.")
            return
        derniere = brut[-1]
        st.markdown(f"**{LIBELLES[derniere['niveau']]}** — {MESSAGES[derniere['niveau']]}")
        if derniere.get("capteurs"):
            st.caption("Sensors involved: " + ", ".join(derniere["capteurs"]))
        historique = tableau_historique(brut)
        st.altair_chart(
            graphique(historique, "proba_usure", SEUIL_XGB, "Wear probability (XGBoost)")
        )
        st.altair_chart(
            graphique(historique, "erreur_reconstruction", SEUIL_AE, "Reconstruction error (AE)")
        )


afficher()
