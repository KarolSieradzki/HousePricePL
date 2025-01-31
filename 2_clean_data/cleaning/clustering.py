from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

def cluster_locations_dbscan(df, eps=0.1, min_samples=5):
    features = ['Latitude', 'Longitude']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['location_cluster'] = dbscan.fit_predict(scaled_data)

    return df