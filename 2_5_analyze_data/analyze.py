import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = '../2_clean_data/results/otodom_houses_cleaned.csv'
data = pd.read_csv(file_path, delimiter=';')

def save_plot(plt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, format='png', dpi=300)
    plt.close()

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

def plot_correlation_heatmap(data, target_column, top_n=10):
    numeric_data = data.select_dtypes(include=[np.number])

    correlations = numeric_data.corr()[target_column].abs().sort_values(ascending=False)
    top_features = correlations.index[:top_n+1]  # Uwzględniamy target_column w top_n

    plt.figure(figsize=(25, 25))
    sns.heatmap(numeric_data[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Top {top_n} Most Correlated Features Including {target_column}')
    save_plot(plt, 'result/features_correlation.png')


def plot_price_distribution(data, target_column):
    plt.figure(figsize=(10, 10))
    sns.histplot(data[target_column] / 1000000, kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {target_column} (in million PLN)')
    plt.xlabel('Price (million PLN)')
    plt.ylabel('Frequency')
    save_plot(plt, 'result/price_distribution.png')


def plot_binary_feature_counts(data, feature_prefix):
    binary_features = [col for col in data.columns if col.startswith(feature_prefix)]
    counts = data[binary_features].sum().sort_values(ascending=False)

    plt.figure(figsize=(16, 6))
    sns.barplot(x=counts.values, y=counts.index, palette='viridis')
    plt.title(f'Number of Listings for Each {feature_prefix} Feature')
    plt.xlabel('Count')
    save_plot(plt, f'result/binary_count/{feature_prefix}.png')


def save_summary_table_as_image(data, numeric_features, binary_feature_prefixes, output_path='result/typical_features.png'):
    numeric_summary = data[numeric_features].mean().sort_values(ascending=False).round(2)  # Zaokrąglamy do 2 miejsc po przecinku

    binary_summaries = []
    for prefix in binary_feature_prefixes:
        binary_features = [col for col in data.columns if col.startswith(prefix)]
        if len(binary_features) > 0:
            most_common_feature = data[binary_features].sum().idxmax()
            most_common_value = most_common_feature.replace(prefix + ' ', '')
            binary_summaries.append((prefix.strip(), most_common_value.strip()))

    summary_data = {'Feature': list(numeric_summary.index) + [item[0] for item in binary_summaries],
                    'Most Common': list(numeric_summary.apply(lambda x: f'{x:,.2f}')) + [item[1] for item in binary_summaries]}  # Formatowanie liczb
    summary_df = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(8, len(summary_df) * 0.5))
    ax.axis('off')

    table = plt.table(cellText=summary_df.values,
                      colLabels=summary_df.columns,
                      cellLoc='center',
                      loc='center',
                      colColours=['#f2f2f2', '#f2f2f2'])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    save_plot(plt, output_path)

def plot_offers_and_price_by_region(data, output_path='result/offers_by_region.png'):
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
    save_plot(plt, output_path)

def plot_price_by_feature(data, feature, output_path='result/price_by_feature.png'):
    grouped = data.groupby(feature)['Price'].mean().reset_index()
    grouped = grouped.sort_values(by=feature)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature, y='Price', data=grouped, palette='coolwarm')
    plt.title(f'Average Price by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Average Price (PLN)')
    plt.tight_layout()
    save_plot(plt, output_path)

if __name__ == '__main__':
    target_column = 'Price'
    plot_correlation_heatmap(data, target_column, top_n=15)
    plot_price_distribution(data, target_column)

    plot_binary_feature_counts(data, 'Zabudowa')
    plot_binary_feature_counts(data, 'Okna')
    plot_binary_feature_counts(data, 'Dach')
    plot_binary_feature_counts(data, 'Stan')
    plot_binary_feature_counts(data, 'Rynek')
    plot_binary_feature_counts(data, 'Położenie')
    plot_binary_feature_counts(data, 'Liczba pięter')
    plot_binary_feature_counts(data, 'Ogłoszenie')
    plot_binary_feature_counts(data, 'Województwo')
    plot_binary_feature_counts(data, 'Okolica')
    plot_binary_feature_counts(data, 'Pokrycie dachu')
    plot_binary_feature_counts(data, 'Ogrodzenie')
    plot_binary_feature_counts(data, 'Materiał budynku')
    plot_binary_feature_counts(data, 'Media')
    plot_binary_feature_counts(data, 'Dojazd')
    plot_binary_feature_counts(data, 'Zabezpieczenia')
    plot_binary_feature_counts(data, 'Dodatkowo')
    plot_binary_feature_counts(data, 'Ogrzewanie')

    numeric_features = ['Price', 'Area', 'Rooms count', 'Land area']
    binary_feature_prefixes = ['Zabudowa', 'Okna', 'Dach', 'Stan', 'Rynek', 'Położenie',
                               'Województwo', 'Materiał budynku', 'Ogrzewanie']  # Prefiksy cech binarnych
    save_summary_table_as_image(data, numeric_features, binary_feature_prefixes)

    plot_price_by_feature(data, feature='Media', output_path='result/price_by_feature/price_by_media.png')
    
    plot_offers_and_price_by_region(data)
