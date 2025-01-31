import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from util_functions.functions import *
from util_functions.form import *
from util_functions.analysis import display_data_analysis

page = st.sidebar.radio(
    "Nawigacja",
    ("Predykcja ceny", "Statystyki trenowania modelu", "Statystyki danych wejściowych")
)

def get_available_models(base_dir="3_train/results"):
    available_models = {}
    for folder in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, folder, "models")
        results_file = os.path.join(base_dir, folder, "results.json")
        if os.path.isdir(model_dir) and os.path.isfile(results_file):
            for file in os.listdir(model_dir):
                if file.endswith(".pkl"):
                    model_name = file.replace(".pkl", "")
                    available_models[f"{folder}/{model_name}"] = {
                        "model_path": os.path.join(model_dir, file),
                        "scaler_path": os.path.join(base_dir, folder, "scaler.pkl"),
                        "results_path": results_file
                    }
    return available_models

def get_mae_from_results(results_path, model_name):
    with open(results_path, 'r') as file:
        results = json.load(file)

    for result in results:
        if result["Model"] == model_name:
            return result["MAE"]
    return None

def display_training_stats():
    st.header("📊 Statystyki trenowania modelu")
    with open("3_train/best_results/results.json", 'r') as file:
        results = json.load(file)
    
    st.write("### Wyniki modeli:")
    st.json(results)
    st.write("### Porównanie modeli:")
    st.image("3_train/best_results/actual_vs_predicted.png", caption="Rzeczywiste vs Przewidywane ceny", use_container_width=True)
    st.image("3_train/best_results/mae_comparison.png", caption="Porównanie MAE", use_container_width=True)
    st.image("3_train/best_results/rmse_comparison.png", caption="Porównanie RMSE", use_container_width=True)







if page == "Predykcja ceny":

    models_dict = get_available_models()
    default_model = {
        "model_path": "3_train/best_results/HistGradientBoosting.pkl",
        "scaler_path": "3_train/best_results/scaler.pkl",
        "results_path": "3_train/best_results/results.json"
    }

    selected_model_key = st.sidebar.selectbox(
        "Wybierz model:",
        options=["best_results/HistGradientBoosting"] + list(models_dict.keys()),
        format_func=lambda x: f"Domyślny: {x}" if x == "best_results/HistGradientBoosting" else x,
        index=0
    )

    selected_model = default_model if selected_model_key == "best_results/HistGradientBoosting" else models_dict[selected_model_key]

    best_model = joblib.load(selected_model["model_path"])
    scaler = joblib.load(selected_model["scaler_path"])

    model_name = selected_model_key.split('/')[-1].replace("_", " ")
    mae = get_mae_from_results(selected_model["results_path"], model_name)

    if mae is None:
        st.warning(f"Nie udało się znaleźć MAE dla modelu: {selected_model_key}. Używana wartość domyślna MAE: 0.")
        mae = 0 


    st.title("🏠 Predykcja ceny nieruchomości")

    input_df = generate_input_form()
    st.write("##### Użyty model: " + selected_model_key)
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
