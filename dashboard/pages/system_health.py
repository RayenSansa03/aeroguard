"""System Health: is the API alive, how fast is it, and do recent flights look like training?"""

import os

import httpx
import streamlit as st

from aeroguard.sante_systeme import ClientMonitoring, lecture_statut, lignes_derive

st.set_page_config(page_title="AeroGuard · System Health", page_icon="🩺", layout="wide")
st.title("System Health")

url = st.sidebar.text_input("API URL", os.environ.get("AEROGUARD_API_URL", "http://127.0.0.1:8000"))
client = ClientMonitoring(url, timeout=120)

try:
    sante = client.sante()
except httpx.HTTPError as erreur:
    st.error(f"API unreachable: {erreur}")
    st.stop()

colonnes = st.columns(4)
colonnes[0].metric("Status", sante["statut"].upper())
colonnes[1].metric("Version", sante["version"])
colonnes[2].metric("Inference", sante.get("inference", "unknown"))
colonnes[3].metric(
    "Response time",
    f"{sante['round_trip_ms']:.0f} ms",
    help=f"Server-side time: {sante['server_ms']:.1f} ms. The rest is the network.",
)
if sante.get("modeles_charges"):
    st.caption("Models loaded.")
else:
    st.caption("Models not loaded yet: they load with the first flight or drift report.")

st.subheader("Data drift")
limite = st.slider("Recent flights analysed", min_value=50, max_value=20000, value=5000, step=50)
try:
    rapport = client.derive(limite)
except httpx.HTTPError as erreur:
    st.warning(f"Drift report unavailable: {erreur}")
    st.stop()

titre, icone, explication = lecture_statut(rapport["status"])
st.markdown(f"### {icone} {titre}")
st.write(explication)
indicateurs = st.columns(3)
indicateurs[0].metric("Flights", rapport.get("n_flights", 0))
indicateurs[1].metric("Engines", rapport.get("n_engines", 0))
indicateurs[2].metric("Features in major drift", rapport.get("n_major", 0))
if rapport.get("features"):
    st.dataframe(lignes_derive(rapport, n=15), hide_index=True)
st.caption(
    "PSI per feature against training deciles: < 0.10 stable, 0.10–0.25 moderate, ≥ 0.25 major."
)
