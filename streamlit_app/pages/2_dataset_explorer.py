from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from util_functions import data_stats 


df_raw = pd.read_json('1_data_scraping/results/otodom_houses.json')
df_cleaned = pd.read_csv('2_clean_data/results/otodom_houses_cleaned.csv', delimiter=';')

st.header("🔍 Podstawowe statystyki")

st.write(f"#### Before cleaning({df_raw.shape[0]} offers):")
st.write(df_raw.head(50))

st.write(f"#### After cleaning({df_cleaned.shape[0]} offers):")
st.write(df_cleaned.head(50))
st.write(f"##### Describe:")
st.write(df_cleaned.describe())


st.header("Correlation heatmap")
target_column = st.selectbox("Select the target column:", df_cleaned.select_dtypes(include=[np.number]).columns, index=2)
data_stats.plot_correlation_heatmap_streamlit(df_cleaned, target_column)

st.header("Distribution of Price:")
data_stats.plot_price_distribution_streamlit(df_cleaned, target_column='Price')

st.header("Number of offers and average price by region:")
data_stats.plot_offers_and_price_by_region(df_cleaned)

st.header("Number of listings of each feature:")
binary_feature_prefix = st.selectbox(
    "Select the prefix of the binary features:",
    [
        'Zabudowa', 'Okna', 'Dach', 'Stan', 'Rynek',
        'Położenie', 'Liczba pięter', 'Ogłoszenie', 'Województwo', 'Okolica',
        'Pokrycie dachu', 'Materiał budynku', 'Media', 'Dojazd', 'Zabezpieczenia',
        'Dodatkowo', 'Ogrzewanie'
    ])
data_stats.plot_binary_feature_counts_streamlit(df_cleaned, binary_feature_prefix)

st.header("Average price by feature")
feature_to_compare = st.selectbox("Select a feature for comparison:", [
        'Zabudowa', 'Okna', 'Dach', 'Stan', 'Rynek',
        'Położenie', 'Liczba pięter', 'Ogłoszenie', 'Województwo', 'Okolica',
        'Pokrycie dachu', 'Materiał budynku', 'Media', 'Dojazd', 'Zabezpieczenia',
        'Dodatkowo', 'Ogrzewanie'
    ])
data_stats.plot_price_by_feature_streamlit(df_cleaned, feature_to_compare)


