import streamlit as st
from util_functions import get_data, model_stats
import pandas as pd

st.write("### Models & statistics")

training_results_path = '3_train/results'
best_results_path = '3_train/best_results'

models_dict = get_data.get_available_models()

trainigs = get_data.get_trainings()
trainigs.sort(key=lambda x: int(x[1:]), reverse=True)

selected_trainig = st.selectbox(
    "Select training:",
    options=trainigs,
    format_func=lambda x: x.replace("t", "trainig ")
)

trainig_results = pd.read_json(f'{training_results_path}/{selected_trainig}/results.json')
trainig_results.set_index('Model', inplace=True)
trainig_results.sort_values(by='MAE', ascending=True, inplace=True)
st.write(f"#### Trainig results for {selected_trainig.replace('t', 'training ')}")
st.dataframe(trainig_results, height=200, width=700)


slected_model = st.selectbox(
    "Select model:",
    options=get_data.get_models_from_trainig(f'3_train/results/{selected_trainig}'),
    index=3
)

st.write("#### Actual vs Predicted Price")
model_stats.plot_actual_vs_predicted_price(models_dict[f"{selected_trainig}/{slected_model}"], slected_model)

st.write("#### MAE Comparison")
model_stats.plot_mae_comparison(models_dict[f"{selected_trainig}/{slected_model}"]["results_path"],
                                 selected_trainig.replace('t', 'training '))

st.write("#### Feature Importances")
model_stats.plot_feature_importances(f'3_train/results/{selected_trainig}', slected_model)