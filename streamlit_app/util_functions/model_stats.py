from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance
import json

def plot_actual_vs_predicted_price(selected_model, model_name):
    df = pd.read_csv('2_clean_data/results/otodom_houses_cleaned.csv', delimiter=';')
    scaler = joblib.load(selected_model["scaler_path"])
    model = joblib.load(selected_model["model_path"])

    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_test_scaled = scaler.transform(X_test)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=np.expm1(model.predict(X_test_scaled)), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted Price ({model_name})")
    st.pyplot(plt)

def plot_mae_comparison(results_path, trainig_number):

    results = pd.read_json(results_path)
    exclude = st.checkbox("Exclude MLPRegressor", value=True)

    if exclude:
        results = results[results["Model"]!="MLPRegressor"]

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Model", y="MAE", data=pd.DataFrame(results), palette="coolwarm")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Model Comparison - MAE ({trainig_number})")
    st.pyplot(plt)

def plot_feature_importances(trainig_path, model_name):

    with open(f"{trainig_path}/models/{model_name}_future_importances.json", "r") as f:
        data = json.load(f)

    features = np.array(data["features"])
    importances = np.array(data["importances"])

    sorted_indices = np.argsort(importances)[::-1][:10]
    top_features = features[sorted_indices]
    top_importances = importances[sorted_indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features, top_importances, color="mediumaquamarine")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top 10 Feature Importances - {model_name}")
    ax.invert_yaxis()

    st.pyplot(fig)
