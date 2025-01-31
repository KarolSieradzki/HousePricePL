import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def encode_multilabel_column(df, col_name, prefix):
    if col_name not in df:
        raise ValueError(f"Column '{col_name}' not found in DataFrame")

    df[col_name] = df[col_name].replace('brak informacji', None)
    df[col_name] = df[col_name].fillna('').str.split(', ')

    mlb = MultiLabelBinarizer()
    binarized = mlb.fit_transform(df[col_name])
    binarized_df = pd.DataFrame(binarized, columns=[f"{prefix} {label}" for label in mlb.classes_], index=df.index)

    return pd.concat([df, binarized_df], axis=1)