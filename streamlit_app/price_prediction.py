import streamlit as st
import pandas as pd
import numpy as np
import joblib
from util_functions.functions import *
from util_functions.form import *
from util_functions import get_data


models_dict = get_data.get_available_models()

trainigs = get_data.get_trainings()
trainigs.sort(key=lambda x: int(x[1:]), reverse=True)

selected_trainig = st.selectbox(
    "Select training:",
    options=trainigs,
    format_func=lambda x: x.replace("t", "trainig ")
)

selected_model = st.selectbox(
    "Select model:",
    options=get_data.get_models_from_trainig(f'3_train/results/{selected_trainig}'),
    index=3
)


model = joblib.load(models_dict[f"{selected_trainig}/{selected_model}"]["model_path"])
scaler = joblib.load(models_dict[f"{selected_trainig}/{selected_model}"]["scaler_path"])

model_name = selected_model
mae = get_data.get_mae_from_results(models_dict[f"{selected_trainig}/{selected_model}"]["results_path"], model_name)

if mae is None:
    st.warning(f"MAE could not be found for model: {selected_model}. Used MAE default value: 0")
    mae = 0 


st.title("🏠 Property price prediction")

input_df = generate_input_form()
st.write("##### Model used: :green[" + selected_model + "] from :green["+selected_trainig.replace('t', 'training ')+']')
st.write("### Data entered(as dataframe):")
st.dataframe(input_df)

if st.button("Predict price"):
    input_scaled = scaler.transform(input_df)
    prediction_log = model.predict(input_scaled)
    prediction = np.expm1(prediction_log)

    lower_bound = prediction[0] - (mae / 2)
    upper_bound = prediction[0] + (mae / 2)

    st.write(f"### Estimated Price: :blue[{prediction[0]:,.2f} PLN]")
    st.write(f"#### Price range(estimated price +- MAE/2):")
    st.write(f"#### :orange[{lower_bound:,.2f} PLN] - :green[{upper_bound:,.2f} PLN]")
