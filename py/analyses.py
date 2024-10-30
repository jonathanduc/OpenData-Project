import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# URL de base de l'API OMS
BASE_URL = "https://ghoapi.azureedge.net/api/"

# Fonction pour récupérer la liste des indicateurs
def get_indicators():
    indicators_url = f"{BASE_URL}Indicator"
    response = requests.get(indicators_url)
    if response.status_code == 200:
        indicators_data = response.json().get('value', [])
        return pd.DataFrame(indicators_data)
    else:
        st.error("Impossible de récupérer la liste des indicateurs.")
        return pd.DataFrame()

# Fonction pour récupérer les données d'un indicateur spécifique
def get_indicator_data(indicator_code):
    data_url = f"{BASE_URL}{indicator_code}"
    response = requests.get(data_url)
    if response.status_code == 200:
        data = response.json().get('value', [])
        return pd.DataFrame(data)
    else:
        st.error("Impossible de récupérer les données de l'indicateur sélectionné.")
        return pd.DataFrame()

# Fonction pour vérifier la recherche par mot-clé
def check_search_term_exists(indicators_df, search_term):
    return search_term and 'IndicatorName' in indicators_df.columns

# Fonction pour récupérer le code de l'indicateur sélectionné
def get_filtered_indicator_code(filtered_indicators_df, selected_indicator):
    return filtered_indicators_df.loc[filtered_indicators_df['IndicatorName'] == selected_indicator, 'IndicatorCode'].values[0]

# Fonction pour tracer la série temporelle pour un pays sélectionné
def plot_country_time_series(data_df, selected_country, selected_indicator):
    country_data = filter_data_by_country(data_df, selected_country).sort_values(by='TimeDim')
    if not country_data.empty and 'TimeDim' in country_data.columns and 'NumericValue' in country_data.columns:
        st.write(f"Évolution de {selected_indicator} pour {selected_country}")
        fig_line = px.line(country_data, x='TimeDim', y='NumericValue', 
                           title=f"Évolution de {selected_indicator} dans {selected_country}",
                           labels={'TimeDim': 'Année', 'NumericValue': 'Valeur'})
        fig_line.update_traces(mode="lines+markers")
        st.plotly_chart(fig_line)
        return True
    else:
        st.write("Aucune donnée temporelle disponible pour ce pays.")
        return False

# Fonction pour tracer la comparaison entre pays pour une année sélectionnée
def plot_year_comparison(data_df, selected_year, selected_indicator):
    year_data = filter_data_by_year(data_df, selected_year).sort_values(by='NumericValue', ascending=False)
    if not year_data.empty and 'SpatialDim' in year_data.columns and 'NumericValue' in year_data.columns:
        st.write(f"Comparaison de {selected_indicator} entre pays pour l'année {selected_year}")
        fig_bar = px.bar(year_data, x='SpatialDim', y='NumericValue', 
                         title=f"Comparaison de {selected_indicator} entre pays en {selected_year}",
                         labels={'SpatialDim': 'Pays', 'NumericValue': 'Valeur'})
        fig_bar.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig_bar)
        return True
    else:
        st.write("Aucune donnée disponible pour cette année.")
        return False

# Filtre les données par pays
def filter_data_by_country(data_df, country):
    return data_df[data_df['SpatialDim'] == country]

# Filtre les données par année
def filter_data_by_year(data_df, year):
    return data_df[data_df['TimeDim'] == year]

# En-tête de l'application
st.title("Analyse des Indicateurs de Santé")

# Récupération des indicateurs et recherche par mot-clé
indicators_df = get_indicators()
search_term = st.text_input("Recherchez un indicateur par mot-clé")
if check_search_term_exists(indicators_df, search_term):
    filtered_indicators_df = indicators_df[indicators_df['IndicatorName'].str.contains(search_term, case=False, na=False)]
    
    # Sélection de l'indicateur
    selected_indicator = st.selectbox("Sélectionnez un indicateur pour voir les données détaillées", 
                                       filtered_indicators_df['IndicatorName'].tolist())
    indicator_code = get_filtered_indicator_code(filtered_indicators_df, selected_indicator)

    # Récupération des données de l'indicateur sélectionné
    data_df = get_indicator_data(indicator_code)
    
    if not data_df.empty:
        # Sélection du pays et affichage de la série temporelle
        countries = data_df['SpatialDim'].unique()
        selected_country = st.selectbox("Sélectionnez un pays pour voir l'évolution dans le temps", countries)
        
        # Tracer la série temporelle pour le pays
        plot_country_time_series(data_df, selected_country, selected_indicator)
        
        # Sélection de l'année et comparaison entre pays
        years = data_df['TimeDim'].dropna().unique()
        selected_year = st.selectbox("Sélectionnez une année pour la comparaison entre pays", years)

        # Tracer la comparaison entre pays pour l'année sélectionnée
        plot_year_comparison(data_df, selected_year, selected_indicator)
    else:
        st.write("Aucune donnée disponible pour cet indicateur.")
else:
    st.write("Entrez un mot-clé pour rechercher un indicateur.")