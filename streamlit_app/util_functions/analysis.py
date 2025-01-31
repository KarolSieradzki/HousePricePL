import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def display_data_analysis():
    st.sidebar.title("Wybierz analizę danych")
    option = st.sidebar.radio(
        "Co chcesz zobaczyć?",
        (
            "Podstawowe statystyki",
            "Przykładowe dane",
            "Heatmapa korelacji",
            "Rozkład cen nieruchomości",
            "Analiza najczęstszych cech binarnych",
            "Średnia cena względem cechy",
            "Cena względem regionu"
        )
    )
    
    file_path = '../2_clean_data/results/otodom_houses_cleaned.csv'
    df = pd.read_csv(file_path, delimiter=';')
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if option == "Podstawowe statystyki":
        st.header("🔍 Podstawowe statystyki")
        st.write(df.describe())

    elif option == "Przykładowe dane":
        st.header("Przykładowe dane")
        st.dataframe(df.head(10))

    elif option == "Heatmapa korelacji":
        st.header("Heatmapa korelacji")
        target_column = st.selectbox("Wybierz kolumnę celu:", df.select_dtypes(include=[np.number]).columns)
        plot_correlation_heatmap_streamlit(df, target_column)

    elif option == "Rozkład cen nieruchomości":
        st.header("Rozkład cen nieruchomości")
        plot_price_distribution_streamlit(df, target_column='Price')

    elif option == "Cena względem regionu":
        st.header("Cena względem regionu")
        plot_offers_and_price_by_region(df)

    elif option == "Analiza najczęstszych cech binarnych":
        st.header("Analiza najczęstszych cech binarnych")
        binary_feature_prefix = st.selectbox(
            "Wybierz prefiks cech binarnych:",
            [
             'Zabudowa', 'Okna', 'Dach', 'Stan', 'Rynek',
             'Położenie', 'Liczba pięter', 'Ogłoszenie', 'Województwo', 'Okolica',
             'Pokrycie dachu', 'Materiał budynku', 'Media', 'Dojazd', 'Zabezpieczenia',
             'Dodatkowo', 'Ogrzewanie'
            ])
        plot_binary_feature_counts_streamlit(df, binary_feature_prefix)

    elif option == "Średnia cena względem cechy":
        st.header("Średnia cena względem wybranej cechy")
        feature_to_compare = st.selectbox("Wybierz cechę do porównania:", [
             'Zabudowa', 'Okna', 'Dach', 'Stan', 'Rynek',
             'Położenie', 'Liczba pięter', 'Ogłoszenie', 'Województwo', 'Okolica',
             'Pokrycie dachu', 'Materiał budynku', 'Media', 'Dojazd', 'Zabezpieczenia',
             'Dodatkowo', 'Ogrzewanie'
            ])
        plot_price_by_feature_streamlit(df, feature_to_compare)




def plot_correlation_heatmap_streamlit(data, target_column, top_n=10):
    numeric_data = data.select_dtypes(include=[np.number])
    correlations = numeric_data.corr()[target_column].abs().sort_values(ascending=False)
    top_features = correlations.index[:top_n+1]

    fig, ax = plt.subplots(figsize=(15, 15))
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

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(x=counts.values, y=counts.index, palette='viridis', ax=ax)
    ax.set_title(f'Number of Listings for Each {feature_prefix} Feature')
    ax.set_xlabel('Count')
    st.pyplot(fig)

def plot_price_by_feature_streamlit(data, feature_prefix):
    relevant_columns = [col for col in data.columns if col.startswith(feature_prefix)]
    
    results = []
    for col in relevant_columns:
        avg_price = data[data[col] == 1]['Price'].mean() 
        results.append({'Feature': col, 'Average Price': avg_price})

    results_df = pd.DataFrame(results).sort_values(by='Average Price', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Average Price', y='Feature', data=results_df, palette='coolwarm', ax=ax)
    ax.set_title(f'Average Price by Binary Features ({feature_prefix})')
    ax.set_xlabel('Average Price (PLN)')
    ax.set_ylabel('Feature')
    st.pyplot(fig)


def plot_offers_and_price_by_region(data):
    regions = [col for col in data.columns if col.startswith('Województwo')]
    region_stats = []

    for region in regions:
        region_name = region.replace('Województwo ', '')
        region_data = data[data[region] == 1]
        num_offers = len(region_data)
        avg_price = region_data['Price'].mean() if num_offers > 0 else 0
        region_stats.append([region_name, num_offers, avg_price])

    df = pd.DataFrame(region_stats, columns=['Region', 'Number of Offers', 'Average Price']).sort_values(by='Number of Offers', ascending=False)

    fig, ax1 = plt.subplots(figsize=(15, 6))
    sns.barplot(x='Region', y='Number of Offers', data=df, palette='viridis', ax=ax1)
    ax1.set_ylabel('Number of Offers')
    ax1.set_xlabel('Region')
    plt.xticks(rotation=90)

    ax2 = ax1.twinx()
    sns.lineplot(x='Region', y='Average Price', data=df, marker='o', color='red', ax=ax2)
    ax2.set_ylabel('Average Price (PLN)')
    plt.title('Number of Offers and Average Price by Region')
    plt.tight_layout()

    st.pyplot(fig)
