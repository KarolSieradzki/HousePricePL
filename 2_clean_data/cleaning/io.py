import pandas as pd
import json

def read_json_to_df(file_path):
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
            return pd.DataFrame(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found. Check the path.")
    except json.JSONDecodeError:
        raise ValueError(f"File '{file_path}' is not a valid JSON format.")


def save_to_csv(df, file_paths=['result/otodom_houses_cleaned.csv'], separator=';'):
    for file_path in file_paths:
        try:
            df.to_csv(file_path, sep=separator, index=False)
        except Exception as e:
            print(f"Błąd podczas zapisywania pliku {file_path}: {e}")