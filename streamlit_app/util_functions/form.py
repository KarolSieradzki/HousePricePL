import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

ALL_FEATURES = {
    "Zabudowa": ["bliźniak", "dworek/pałac", "gospodarstwo", "kamienica", "szeregowiec", "wolnostojący"],
    "Okna": ["aluminiowe", "brak", "drewniane", "plastikowe"],
    "Dach": ["brak", "płaski", "skośny"],
    "Stan": ["do remontu", "do wykończenia", "do zamieszkania", "stan surowy otwarty", "stan surowy zamknięty"],
    "Rynek": ["pierwotny", "wtórny"],
    "Położenie": ["miasto", "pod miastem", "wieś"],
    "Liczba pięter": ["1 piętro", "2 piętra", "3 piętra lub więcej", "parterowy"],
    "Ogłoszenie": ["biuro nieruchomości", "deweloper", "prywatny"],
    "Województwo": [
        "Dolnośląskie", "Kujawsko-pomorskie", "Lubelskie", "Lubuskie", "Mazowieckie",
        "Małopolskie", "Opolskie", "Podkarpackie", "Podlaskie", "Pomorskie",
        "Warmińsko-mazurskie", "Łódzkie", "Śląskie", "Świętokrzyskie"
    ],
    "Okolica": ["góry", "jezioro", "las", "morze"],
    "Pokrycie dachu": ["blacha", "dachówka", "eternit", "gont", "inne", "papa", "strzecha", "łupek"],
    "Ogrodzenie": ["betonowe", "drewniane", "inne", "metalowe", "murowane", "siatka", "żywopłoty"],
    "Materiał budynku": ["beton", "beton komórkowy", "cegła", "drewno", "inny", "keramzyt", "pustak", "silikat", "wielka płyta"],
    "Media": ["gaz", "internet", "kanalizacja", "oczyszczalnia", "prąd", "szambo", "telefon", "telewizja kablowa", "woda"],
    "Dojazd": ["asfaltowy", "polny", "utwardzony"],
    "Zabezpieczenia": ["domofon / wideofon", "drzwi / okna antywłamaniowe", "rolety antywłamaniowe", "system alarmowy"],
    "Dodatkowo": ["basen", "garaż", "klimatyzacja", "piwnica", "strych"],
    "Ogrzewanie": ["biomasa", "elektryczne", "gazowe", "geotermika", "kolektor słoneczny", "kominkowe", "miejskie", "olejowe", "piece kaflowe", "pompa ciepła", "węglowe"]
}


def cluster_location_for_input(latitude, longitude, csv_path="2_clean_data/results/otodom_houses_cleaned.csv", eps=0.1, min_samples=5):
   
    df = pd.read_csv(csv_path, delimiter=";")
    existing_coordinates = df[["Latitude", "Longitude"]].dropna().to_numpy()

    new_point = np.array([[latitude, longitude]])
    coordinates = np.vstack((existing_coordinates, new_point))

    scaler = StandardScaler()
    scaled_coordinates = scaler.fit_transform(coordinates)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_coordinates)

    return clusters[-1]

def generate_input_form():
    st.sidebar.header("Podaj podstawowe cechy nieruchomości")

    # Pola numeryczne
    latitude = st.sidebar.number_input("Szerokość geograficzna (Latitude)", value=52.2297)
    longitude = st.sidebar.number_input("Długość geograficzna (Longitude)", value=21.0122)
    area = st.sidebar.number_input("Powierzchnia budynku (m²)", min_value=20, max_value=500, value=100)
    rooms_count = st.sidebar.number_input("Liczba pokoi", min_value=1, max_value=10, value=3)
    land_area = st.sidebar.number_input("Powierzchnia działki (m²)", min_value=50, max_value=2000, value=500)

    # jednokrotny wybór
    zabudowa = st.sidebar.selectbox("Rodzaj zabudowy", options=["bliźniak", "dworek/pałac", "gospodarstwo", "kamienica", "szeregowiec", "wolnostojący"])
    stan = st.sidebar.selectbox("Stan wykończenia", options=["do remontu", "do wykończenia", "do zamieszkania", "stan surowy otwarty", "stan surowy zamknięty"])
    rynek = st.sidebar.selectbox("Typ rynku", options=["pierwotny", "wtórny"])
    okna = st.sidebar.selectbox("Okna", options=["aluminiowe", "brak", "drewniane", "plastikowe"])
    dach = st.sidebar.selectbox("Pokrycie dachu", options=["brak", "płaski", "skośny"])
    polozenie = st.sidebar.selectbox("Położenie", options=["miasto", "pod miastem", "wieś"])
    pietra = st.sidebar.selectbox("Liczba pięter", options=["1 piętro", "2 piętra", "3 piętra lub więcej", "parterowy"])
    ogloszenie = st.sidebar.selectbox("Typ ogłoszenia", options=["biuro nieruchomości", "deweloper", "prywatny"])
    wojewodztwo = st.sidebar.selectbox("Województwo", options=[
        "Dolnośląskie", "Kujawsko-pomorskie", "Lubelskie", "Lubuskie", "Mazowieckie",
        "Małopolskie", "Opolskie", "Podkarpackie", "Podlaskie", "Pomorskie",
        "Warmińsko-mazurskie", "Łódzkie", "Śląskie", "Świętokrzyskie"
    ])

    # wielokrotny wybór
    media = st.sidebar.multiselect("Media", options=["gaz", "internet", "kanalizacja", "oczyszczalnia", "prąd", "szambo", "telefon", "telewizja kablowa", "woda"])
    dodatkowo = st.sidebar.multiselect("Dodatkowe udogodnienia", options=["basen", "garaż", "klimatyzacja", "piwnica", "strych"])
    okolica = st.sidebar.multiselect("Okolica", options=["góry", "jezioro", "las", "morze"])
    pokrycie_dachu = st.sidebar.multiselect("Pokrycie dachu", options=["blacha", "dachówka", "eternit", "gont", "inne", "papa", "strzecha", "łupek"])
    ogrodzenie = st.sidebar.multiselect("Ogrodzenie", options=["betonowe", "drewniane", "inne", "metalowe", "murowane", "siatka", "żywopłoty"])
    material_budynku = st.sidebar.multiselect("Materiał budynku", options=["beton", "beton komórkowy", "cegła", "drewno", "inny", "keramzyt", "pustak", "silikat", "wielka płyta"])
    dojazd = st.sidebar.multiselect("Dojazd", options=["asfaltowy", "polny", "utwardzony"])
    zabezpieczenia = st.sidebar.multiselect("Zabezpieczenia", options=["domofon / wideofon", "drzwi / okna antywłamaniowe", "rolety antywłamaniowe", "system alarmowy"])
    ogrzewanie = st.sidebar.multiselect("Ogrzewanie", options=["biomasa", "elektryczne", "gazowe", "geotermika", "kolektor słoneczny", "kominkowe", "miejskie", "olejowe", "piece kaflowe", "pompa ciepła", "węglowe"])

    input_data = {
        "Latitude": [latitude],
        "Longitude": [longitude],
        "Area": [area],
        "Rooms count": [rooms_count],
        "Land area": [land_area]
    }

    for value in ALL_FEATURES["Zabudowa"]:
        input_data[f"Zabudowa {value}"] = [1 if value == zabudowa else 0]

    for value in ALL_FEATURES["Okna"]:
        input_data[f"Okna {value}"] = [1 if value in okna else 0]

    for value in ALL_FEATURES["Dach"]:
        input_data[f"Dach {value}"] = [1 if value in dach else 0] 

    for value in ALL_FEATURES["Stan"]:
        input_data[f"Stan {value}"] = [1 if value == stan else 0]

    for value in ALL_FEATURES["Rynek"]:
        input_data[f"Rynek {value}"] = [1 if value == rynek else 0]

    for value in ALL_FEATURES["Położenie"]:
        input_data[f"Położenie {value}"] = [1 if value in polozenie else 0]

    for value in ALL_FEATURES["Liczba pięter"]:
        input_data[f"Liczba pięter {value}"] = [1 if value in pietra else 0]

    for value in ALL_FEATURES["Ogłoszenie"]:
        input_data[f"Ogłoszenie {value}"] = [1 if value in ogloszenie else 0]

    for value in ALL_FEATURES["Województwo"]:
        input_data[f"Województwo {value}"] = [1 if value in wojewodztwo else 0]

    for value in ALL_FEATURES["Okolica"]:
        input_data[f"Okolica {value}"] = [1 if value in okolica else 0]

    for value in ALL_FEATURES["Pokrycie dachu"]:
        input_data[f"Pokrycie dachu {value}"] = [1 if value in pokrycie_dachu else 0]

    for value in ALL_FEATURES["Ogrodzenie"]:
        input_data[f"Ogrodzenie {value}"] = [1 if value in ogrodzenie else 0]

    for value in ALL_FEATURES["Materiał budynku"]:
        input_data[f"Materiał budynku {value}"] = [1 if value in material_budynku else 0]

    for value in ALL_FEATURES["Media"]:
        input_data[f"Media {value}"] = [1 if value in media else 0]

    for value in ALL_FEATURES["Dojazd"]:
        input_data[f"Dojazd {value}"] = [1 if value in dojazd else 0] 

    for value in ALL_FEATURES["Zabezpieczenia"]:
        input_data[f"Zabezpieczenia {value}"] = [1 if value in zabezpieczenia else 0]

    for value in ALL_FEATURES["Dodatkowo"]:
        input_data[f"Dodatkowo {value}"] = [1 if value in dodatkowo else 0]

    for value in ALL_FEATURES["Ogrzewanie"]:
        input_data[f"Ogrzewanie {value}"] = [1 if value in ogrzewanie else 0]


    location_cluster = cluster_location_for_input(latitude, longitude)
    input_data["location_cluster"] = location_cluster
    input_data["Area_per_room"] = input_data["Area"][0] / input_data["Rooms count"][0]
    input_data["Building_density"] = input_data["Area"][0] / input_data["Land area"][0]
    input_data["Area_to_rooms_ratio"] = input_data["Land area"][0] / input_data["Rooms count"][0]

    return pd.DataFrame(input_data)
