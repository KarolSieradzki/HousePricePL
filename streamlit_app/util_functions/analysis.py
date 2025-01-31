import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def display_data_analysis():
    st.header("🔍 Statystyki oczyszczonych danych")
    
    file_path = '../2_clean_data/results/otodom_houses_cleaned.csv'
    df = pd.read_csv(file_path, delimiter=';')

    st.write("### Podstawowe statystyki:")
    st.write(df.describe())

    st.write("### Przykładowe dane:")
    st.dataframe(df.head(10))

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    st.write("### 🔗 Heatmapa korelacji")
    target_column = st.selectbox("Wybierz kolumnę celu:", df.select_dtypes(include=[np.number]).columns)
    plot_correlation_heatmap_streamlit(df, target_column)

    st.write("### 📊 Rozkład cen nieruchomości")
    plot_price_distribution_streamlit(df, target_column='Price')

    st.write("### 🏠 Analiza najczęstszych cech binarnych")
    binary_feature_prefix = st.selectbox("Wybierz prefiks cech binarnych:", ['Zabudowa', 'Ogrzewanie', 'Dodatkowo', 'Media', 'Dojazd'])
    plot_binary_feature_counts_streamlit(df, binary_feature_prefix)

    st.write("### 📈 Średnia cena względem wybranej cechy")
    feature_to_compare = st.selectbox("Wybierz cechę do porównania:", df.columns)
    plot_price_by_feature_streamlit(df, feature_to_compare)


def plot_correlation_heatmap_streamlit(data, target_column, top_n=10):
    numeric_data = data.select_dtypes(include=[np.number])
    correlations = numeric_data.corr()[target_column].abs().sort_values(ascending=False)
    top_features = correlations.index[:top_n+1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_data[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title(f'Top {top_n} Most Correlated Features Including {target_column}')
    st.pyplot(fig)

def plot_price_distribution_streamlit(data, target_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[target_column] / 1_000_000, kde=True, bins=30, color='skyblue', ax=ax)
    ax.set_title('Distribution of Price (in million PLN)')
    ax.set_xlabel('Price (million PLN)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def plot_binary_feature_counts_streamlit(data, feature_prefix):
    binary_features = [col for col in data.columns if col.startswith(feature_prefix)]
    counts = data[binary_features].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, palette='viridis', ax=ax)
    ax.set_title(f'Number of Listings for Each {feature_prefix} Feature')
    ax.set_xlabel('Count')
    st.pyplot(fig)

def plot_price_by_feature_streamlit(data, feature):
    grouped = data.groupby(feature)['Price'].mean().reset_index().sort_values(by=feature)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature, y='Price', data=grouped, palette='coolwarm', ax=ax)
    ax.set_title(f'Average Price by {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Average Price (PLN)')
    st.pyplot(fig)
