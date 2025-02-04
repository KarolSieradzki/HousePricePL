from cleaning.io import read_json_to_df, save_to_csv
from cleaning.basic_cleaning import *
from cleaning.encoding import encode_multilabel_column
from cleaning.clustering import cluster_locations_dbscan
from cleaning.statistics import dataframe_statistics

def main():
    otodom_houses = read_json_to_df('../1_data_scraping/results/otodom_houses.json')

    # rename columns
    column_mapping = {
        "Rok budowy": "Year of construction",
        "Powierzchnia działki": "Land area"
    }
    otodom_houses.rename(columns=column_mapping, inplace=True)

    #basic cleaning
    otodom_houses = clean_price(otodom_houses, "Price")
    otodom_houses = clean_price_per_m(otodom_houses, "Price per sqm")
    otodom_houses = clean_area(otodom_houses, "Area")
    otodom_houses = clean_area(otodom_houses, "Land area")
    otodom_houses = validate_and_fix_price_per_sqm(otodom_houses, "Price", "Price per sqm", "Area")
    otodom_houses = clear_room_count(otodom_houses, "Rooms count")
    otodom_houses = get_voivodeship_from_localization(otodom_houses, 'Address')
    otodom_houses = coordinates_to_numeric(otodom_houses, 'Latitude', 'Longitude')
    otodom_houses = clean_year_of_construction(otodom_houses, 'Year of construction')
    otodom_houses = clear_date(otodom_houses, 'Date')
   
    
    otodom_houses = treat_custom_nulls(otodom_houses, ['Brak informacji', 'brak informacji'])
    otodom_houses = remove_null_rows(otodom_houses, ['Price', 'Area', 'Land area', 'voivodeship', 'Latitude', 'Longitude'])

    #column coding
    otodom_houses = encode_multilabel_column(otodom_houses, 'Rodzaj zabudowy', 'Zabudowa')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Okna', 'Okna')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Dach', 'Dach')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Stan wykończenia', 'Stan')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Rynek', 'Rynek')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Położenie', 'Położenie')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Liczba pięter', 'Liczba pięter')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Typ ogłoszeniodawcy', 'Ogłoszenie')
    otodom_houses = encode_multilabel_column(otodom_houses, 'voivodeship', 'Województwo')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Okolica', 'Okolica')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Pokrycie dachu', 'Pokrycie dachu')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Ogrodzenie', 'Ogrodzenie')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Materiał budynku', 'Materiał budynku')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Media', 'Media')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Dojazd', 'Dojazd')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Zabezpieczenia', 'Zabezpieczenia')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Informacje dodatkowe', 'Dodatkowo')
    otodom_houses = encode_multilabel_column(otodom_houses, 'Ogrzewanie', 'Ogrzewanie')

    #location clustering 
    otodom_houses = cluster_locations_dbscan(otodom_houses, eps=0.05, min_samples=2)

    # drop unused/unimportant columns
    columns_to_drop=[
        'link', 'page_number', 'Poddasze', 'Certyfikat energetyczny', 'Dostępne od', 'Czynsz', 'Address',
        'Zabudowa', 'Zabudowa ', 'Okna ', 'Rodzaj zabudowy', 'Okna', 'Okolica', 'Okolica ', 'Dach', 'Dach ', 'Pokrycie dachu', 'Pokrycie dachu ',
        'Stan wykończenia', 'Stan ', 'Rynek', 'Rynek ', 'Ogrodzenie', 'Ogrodzenie ', 'Materiał budynku', 'Materiał budynku ',
        'Liczba pięter', 'Liczba pięter ', 'Położenie', 'Położenie ', 'Media', 'Media ', 'Dojazd', 'Dojazd ', 'Zabezpieczenia', 'Zabezpieczenia ',
        'Informacje dodatkowe', 'Dodatkowo ', 'Ogrzewanie', 'Ogrzewanie ', 'Typ ogłoszeniodawcy', 'Ogłoszenie ', 'Real estate office name',
        'Date', 'Price per sqm', 'Year of construction', 'voivodeship', 'Województwo '
    ]
    otodom_houses = drop_columns_by_name(otodom_houses, columns_to_drop)

    # delete duplicates
    otodom_houses = otodom_houses.loc[:, ~otodom_houses.columns.duplicated()]

    # remove outliers
    upper_limit = otodom_houses['Price'].quantile(0.98)
    lower_limit = otodom_houses['Price'].quantile(0.02)
    otodom_houses = otodom_houses[(otodom_houses['Price'] >= lower_limit) & (otodom_houses['Price'] <= upper_limit)]

    # add new features
    otodom_houses["Area_per_room"] = otodom_houses["Area"] / otodom_houses["Rooms count"]
    otodom_houses["Building_density"] = otodom_houses["Area"] / otodom_houses["Land area"]
    otodom_houses["Area_to_rooms_ratio"] = otodom_houses["Land area"] / otodom_houses["Rooms count"]


    dataframe_statistics(otodom_houses, exclude_columns=[])

    save_to_csv(otodom_houses, file_paths=[
        'results/otodom_houses_cleaned.csv'
    ])

if __name__ == "__main__":
    main()
