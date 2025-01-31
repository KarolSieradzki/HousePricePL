import pandas as pd
import numpy as np
import unicodedata

# price should be numeric only and greater than 10 000PLN
def clean_price(df, price_column, delete_price_below = 10000):
    if price_column not in df:
        raise ValueError(f"Column '{price_column}' not found in DataFrame")

    # delete "m 2" for domiporta
    df[price_column] = df[price_column].str.replace(r"m\s2", "", regex=True)
    # replace , to . for decimal numbers
    df[price_column] = df[price_column].str.replace(",", ".")
    # delete non numeric characters
    df[price_column] = df[price_column].str.replace(r"[^\d]", "", regex=True)
    #  convert to numeric
    df[price_column] = pd.to_numeric(df[price_column], errors="coerce")
    # delete values smaller than 10 000PLN
    df = df[df[price_column] >= delete_price_below]
    # delete rows with no price
    df = df[df[price_column].notnull()]

    return df

def clean_price_per_m(df, column_name):
    return clean_price(df, column_name, delete_price_below=100)
    

def clean_area(df, column_name):
    if column_name not in df:
        raise ValueError(f"Column '{column_name}', not found in DtaFrame")
    
    # # delete "m 2" for domiporta
    # df[column_name] = df[column_name].str.replace(r"m\s2", "", regex=True)
    # # replace , to . for decimal numbers
    # df[column_name] = df[column_name].str.replace(",", ".")
    # delete non numeric characters except dot
    df[column_name] = df[column_name].str.replace(r"[^\d.]", "", regex=True)
    # convert to numeric
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    #delete null values
    df = df[df[column_name].notnull()]
    
    return df

def clean_year_of_construction(df, col_name):
    if col_name not in df:
        raise ValueError(f"Column '{col_name}', not found in DtaFrame")
    
    # get onlu 4 digits
    df[col_name] = df[col_name].astype(str).str.extract(r'(\d{4})')
    # convert to numeric
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    # delete unrealistic values
    df[col_name] = df[col_name].where((df[col_name] >= 1200) & (df[col_name] <= 2040), np.nan)
    # convert to int
    df[col_name] = df[col_name].dropna().astype(int)

    return df

def get_voivodeship_from_localization(df, localization_col, name_for_voivodeship_col="voivodeship"):
    if localization_col not in df:
        raise ValueError(f"Column '{localization_col}', not found in DtaFrame")
    
    all_voivodeships=[
        'Dolnośląskie',
        'Kujawsko-pomorskie',
        'Lubelskie',
        'Lubuskie',
        'Łódzkie',
        'Małopolskie',
        'Mazowieckie',
        'Opolskie',
        'Podkarpackie',
        'Podlaskie',
        'Pomorskie',
        'Śląskie',
        'Świętokrzyskie',
        'Warmińsko-mazurskie',
        'Wielkopolskie',
        'Zachodniopomorskie',
    ]

    def normalize_text(text):
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
        return text.upper()
    
    normalized_voivodeships = [normalize_text(voivodeship) for voivodeship in all_voivodeships]

    def extract_voivodeship(localization):
        if not isinstance(localization, str):
            return None
        normalized_localization = normalize_text(localization)
        for voivodeship, normalized_voivodeship in zip(all_voivodeships, normalized_voivodeships):
            if normalized_voivodeship in normalized_localization:
                return voivodeship
        return None
    
    df[name_for_voivodeship_col] = df[localization_col].apply(extract_voivodeship)

    return df

def coordinates_to_numeric(df, lat_col, long_col):
    if lat_col not in df: raise ValueError(f"Column '{lat_col}' not found in DataFrame")
    if long_col not in df: raise ValueError(f"Column '{long_col}' not found in DataFrame")

    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[long_col] = pd.to_numeric(df[long_col], errors="coerce")

    return df

def validate_and_fix_price_per_sqm(df, price_col, price_per_sqm_col, area_col):
    corrected_count = 0

    for idx, row in df.iterrows():
        price = row[price_col]
        area = row[area_col]
        price_per_sqm = row[price_per_sqm_col]

        if not np.isnan(price) and not np.isnan(area) and area > 0:
            calculated_price_per_sqm = price/area
            if price_per_sqm is None or abs(calculated_price_per_sqm - price_per_sqm)>1:
                df.at[idx, price_per_sqm_col] = calculated_price_per_sqm
                corrected_count+=1

    print(f"Count of corrected prices per square meter: {corrected_count}")
    return df

def clear_room_count(df, column_name):
    if column_name not in df:
        raise ValueError(f"Column '{column_name}', not found in DataFarme")
    
     # delete non numeric characters
    df[column_name] = df[column_name].str.replace(r"[^\d]", "", regex=True)
    #  convert to numeric
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    # delete nulls
    df = df[df[column_name].notnull()]
    df[column_name] = df[column_name].astype(int)
    
    return df

def clear_date(df, date_col):
    if date_col not in df:
        raise ValueError(f"Column '{date_col}', not found in DataFarme")
    
    df[date_col] = df[date_col].str.extract(r'(\d{1,2}\.\d{1,2}\.\d{4})')
    df[date_col] = pd.to_datetime(df[date_col], format='%d.%m.%Y', errors='coerce')

    return df


def remove_null_rows(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_columns}")
    
    cleaned_df = df.dropna(subset=required_columns)
    
    return cleaned_df

def treat_custom_nulls(df, custom_null_values):
    return df.replace(custom_null_values, np.nan)

def drop_columns_by_name(df, columns_to_drop=[]):
    missing_columns = [col for col in columns_to_drop if col not in df.columns]
    if missing_columns:
        print(f"Columns not found: {missing_columns}")
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")
    return df