import streamlit as st
import requests

st.title("🏸 Badminton Talent Identifier")

age = st.slider("Age", 10, 30)
agility = st.slider("Agility", 1, 10)

if st.button("Predict"):
    payload = {
        "age": age,
        "height": 170,
        "weight": 65,
        "agility": agility,
        "endurance": 7,
        "hand_eye_coord": 8,
        "flexibility": 7
    }
    res = requests.post("http://localhost:8000/predict", json=payload)
    st.success(res.json())
