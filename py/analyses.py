import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# URL de base de l'API OMS
BASE_URL = "https://ghoapi.azureedge.net/api/"

# En-tête de l'application
st.title("Analyse des Indicateurs de Santé")

# Utiliser l'API pour obtenir la liste des indicateurs
indicators_url = f"{BASE_URL}Indicator"
response = requests.get(indicators_url)
if response.status_code == 200:
    indicators_data = response.json().get('value', [])
    indicators_df = pd.DataFrame(indicators_data)

    # Interface utilisateur pour chercher des indicateurs par mot-clé
    search_term = st.text_input("Recherchez un indicateur par mot-clé")
    if search_term and 'IndicatorName' in indicators_df.columns:
        filtered_indicators_df = indicators_df[indicators_df['IndicatorName'].str.contains(search_term, case=False, na=False)]

        # Sélection de l'indicateur pour analyse
        selected_indicator = st.selectbox("Sélectionnez un indicateur pour voir les données détaillées", 
                                           filtered_indicators_df['IndicatorName'].tolist())
        indicator_code = filtered_indicators_df.loc[filtered_indicators_df['IndicatorName'] == selected_indicator, 'IndicatorCode'].values[0]

        # Récupération des données pour l'indicateur sélectionné
        data_url = f"{BASE_URL}{indicator_code}"
        data_response = requests.get(data_url)
        if data_response.status_code == 200:
            data = data_response.json().get('value', [])
            data_df = pd.DataFrame(data)
            
            if not data_df.empty:
                # Sélection du pays pour la visualisation temporelle
                countries = data_df['SpatialDim'].unique()
                selected_country = st.selectbox("Sélectionnez un pays pour voir l'évolution dans le temps", countries)

                # Filtrer les données pour le pays sélectionné et afficher la série temporelle
                country_data = data_df[data_df['SpatialDim'] == selected_country]
                country_data = country_data.sort_values(by='TimeDim')  # Trier par année

                if not country_data.empty and 'TimeDim' in country_data.columns and 'NumericValue' in country_data.columns:
                    # Graphique de série temporelle
                    fig = px.line(country_data, x='TimeDim', y='NumericValue', 
                                  title=f"Évolution de {selected_indicator} pour {selected_country}",
                                  labels={'TimeDim': 'Année', 'NumericValue': 'Valeur'})
                    fig.update_traces(mode="lines+markers")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Aucune donnée temporelle disponible pour ce pays.")

                # Comparaison entre pays pour une année spécifique
                years = data_df['TimeDim'].dropna().unique()
                selected_year = st.selectbox("Sélectionnez une année pour la comparaison entre pays", years)

                # Filtrer les données pour l'année sélectionnée
                year_data = data_df[data_df['TimeDim'] == selected_year]

                if not year_data.empty and 'SpatialDim' in year_data.columns and 'NumericValue' in year_data.columns:
                    # Trier par 'NumericValue' en ordre décroissant
                    year_data = year_data.sort_values(by='NumericValue', ascending=False)

                    # Graphique de comparaison par pays pour l'année choisie
                    fig = px.bar(year_data, x='SpatialDim', y='NumericValue', 
                                 title=f"Comparaison de {selected_indicator} entre pays pour l'année {selected_year}",
                                 labels={'SpatialDim': 'Pays', 'NumericValue': 'Valeur'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Aucune donnée disponible pour cette année.")
            else:
                st.write("Aucune donnée disponible pour cet indicateur.")
        else:
            st.error("Impossible de récupérer les données de l'indicateur sélectionné.")
    else:
        st.write("Entrez un mot-clé pour rechercher un indicateur.")
else:
    st.error("Impossible de récupérer la liste des indicateurs.")