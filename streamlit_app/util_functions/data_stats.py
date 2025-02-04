from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
    ax.set_xlabel('Price (Million PLN)')
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
    ax.set_xlabel('Average Price (Million PLN)')
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
    ax2.set_ylabel('Average Price (Million PLN)')
    plt.title('Number of Offers and Average Price by Region')
    plt.tight_layout()

    st.pyplot(fig)
