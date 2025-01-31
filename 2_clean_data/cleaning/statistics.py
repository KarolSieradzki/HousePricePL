import pandas as pd

def dataframe_statistics(df, exclude_columns=[]):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input mus be a pandas DataFrame")
    
    for column in df.columns:
        if column in exclude_columns:
            continue

        print(f'Column: {column}')
        print("-" * 40)

        null_count = df[column].isnull().sum()
        print(f"Null values: {null_count}")

        # for numeric values
        if pd.api.types.is_numeric_dtype(df[column]):
            mean = df[column].mean()
            median = df[column].median()
            min_val = df[column].min()
            max_val = df[column].max()

            print(f"mean: {mean}")
            print(f"median: {median}")
            print(f"min: {min_val}")
            print(f"max: {max_val}")
        
        elif pd.api.types.is_object_dtype(df[column]):
            unique_values = df[column].unique()
            unique_count = len(unique_values)

            print(f'unique values: {unique_values}')
            print(f'unique values count : {unique_count}')
        else:
            print("Other data type")

        print("-" * 40)