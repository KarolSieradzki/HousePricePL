import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from util_functions.functions import *
from util_functions.form import *
from util_functions.analysis import display_data_analysis
from pathlib import Path
from PIL import Image

use_model = 'HistGradientBoosting'
best_model = joblib.load(f"../3_train/best_results/{use_model}.pkl")
scaler = joblib.load(f"../3_train/best_results/scaler.pkl")
model_scores = get_model_scores(use_model)
mae = model_scores['MAE']

def display_training_stats():
    st.header("📊 Statystyki trenowania modelu")
    
    with open("../3_train/best_results/results.json", 'r') as file:
        results = json.load(file)
    
    st.write("### Wyniki modeli:")
    st.json(results)

    st.write("### Porównanie modeli:")
    st.image("../3_train/best_results/actual_vs_predicted.png", caption="Rzeczywiste vs Przewidywane ceny", use_container_width=True)
    st.image("../3_train/best_results/mae_comparison.png", caption="Porównanie MAE", use_container_width=True)
    st.image("../3_train/best_results/rmse_comparison.png", caption="Porównanie RMSE", use_container_width=True)
   


# def display_data_analysis():
#     st.header("🔍 Statystyki oczyszczonych danych")
    
#     df = pd.read_csv('../2_clean_data/results/otodom_houses_cleaned.csv', delimiter=';')
    
#     st.write("### Podstawowe statystyki:")
#     st.write(df.describe())
    
#     st.write("### Przykładowe dane:")
#     st.dataframe(df.head(10))

page = st.sidebar.radio(
    "Nawigacja",
    ("Predykcja ceny", "Statystyki trenowania modelu", "Statystyki danych wejściowych")
)

if page == "Predykcja ceny":
    st.title("🏠 Predykcja ceny nieruchomości")

    input_df = generate_input_form()

    st.write("##### Użyty model: " + use_model)
    st.write("### Wprowadzone dane:")
    st.dataframe(input_df)

    if st.button("Przewiduj cenę"):
        input_scaled = scaler.transform(input_df)

        prediction_log = best_model.predict(input_scaled)
        prediction = np.expm1(prediction_log)

        lower_bound = prediction[0] - (mae / 2)
        upper_bound = prediction[0] + (mae / 2)

        st.write(f"### Szacowana cena: **{prediction[0]:,.2f} PLN**")
        st.write(f"### Przedział cenowy: {lower_bound:,.2f} PLN - {upper_bound:,.2f} PLN")

elif page == "Statystyki trenowania modelu":
    display_training_stats()

elif page == "Statystyki danych wejściowych":
    display_data_analysis()
