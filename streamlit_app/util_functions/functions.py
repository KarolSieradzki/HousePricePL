import streamlit as st
import json

def get_model_scores(model_name):
    with open("3_train/best_results/results.json", "r") as f:
        model_results = json.load(f)
    
    for result in model_results:
        if result['Model'] == model_name:
            return {
                "MAE": result['MAE'],
                "RMSE": result['RMSE'],
                "R2": result['R2']
            }
    
    st.error(f"Nie znaleziono wyników dla modelu: {model_name}")
    return {"MAE": None, "RMSE": None, "R2": None}