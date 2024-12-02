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

# Fonction pour vérifier si les valeurs sont numériques
def has_numeric_values(data_df):
    return pd.to_numeric(data_df['NumericValue'], errors='coerce').notna().any()

# Fonction pour tracer la série temporelle pour un pays sélectionné
def plot_country_time_series(data_df, selected_country, selected_indicator, chart_type):
    # Filtrer les données pour le pays sélectionné
    country_data = filter_data_by_country(data_df, selected_country)

    # Agréger les données pour gérer les doublons (par exemple, moyenne pour chaque année)
    if 'TimeDim' in country_data.columns and 'NumericValue' in country_data.columns:
        aggregated_data = (
            country_data.groupby('TimeDim', as_index=False)
                        .agg({'NumericValue': 'mean'})  # Utilisez 'mean', 'median', ou autre selon vos besoins
        )
        
        # Trier par année
        aggregated_data = aggregated_data.sort_values(by='TimeDim')

        if not aggregated_data.empty:
            # Choisir le type de graphique
            if chart_type == "Ligne":
                fig = px.line(aggregated_data, x='TimeDim', y='NumericValue', 
                              title=f"Évolution de {selected_indicator} dans {selected_country}",
                              labels={'TimeDim': 'Année', 'NumericValue': 'Valeur'})
                fig.update_traces(mode="lines+markers")
            else:  # Graphique en barres
                fig = px.bar(aggregated_data, x='TimeDim', y='NumericValue',
                             title=f"Évolution de {selected_indicator} dans {selected_country}",
                             labels={'TimeDim': 'Année', 'NumericValue': 'Valeur'})

            st.plotly_chart(fig)
        else:
            st.write("Aucune donnée temporelle disponible après l'agrégation pour ce pays.")
    else:
        st.write("Données invalides ou manquantes pour tracer l'évolution temporelle.")

# Fonction pour tracer la comparaison entre pays pour une année sélectionnée
def plot_year_comparison(data_df, selected_year, selected_indicator, chart_type):
    year_data = filter_data_by_year(data_df, selected_year)

    if not year_data.empty and 'SpatialDim' in year_data.columns and 'NumericValue' in year_data.columns:
        # Agréger les données pour gérer les doublons (par exemple, moyenne pour chaque pays)
        aggregated_data = (
            year_data.groupby('SpatialDim', as_index=False)
                     .agg({'NumericValue': 'mean'})  # Choisissez la méthode d'agrégation appropriée
        )

        # Ajouter un filtre interactif pour les plages de valeurs
        min_value = int(aggregated_data['NumericValue'].min())
        max_value = int(aggregated_data['NumericValue'].max())

        if min_value == max_value:
            st.write(f"Pour l'année {selected_year}, toutes les valeurs de l'indicateur sont identiques : {min_value}. Aucun filtrage par plage de valeurs n'est possible.")
            filtered_data = aggregated_data  # Pas de filtrage
        else:
            st.write(f"Valeurs disponibles pour {selected_year} : entre {min_value} et {max_value}")
            
            # Filtre interactif via un slider
            value_range = st.slider(
                "Filtrer les pays par plage de valeurs", 
                min_value=min_value, 
                max_value=max_value, 
                value=(min_value, max_value)
            )

            # Appliquer le filtre
            filtered_data = aggregated_data[(aggregated_data['NumericValue'] >= value_range[0]) & 
                                            (aggregated_data['NumericValue'] <= value_range[1])]

        # Ajouter une limite aux N premiers pays
        top_n = st.number_input(
            "Nombre maximum de pays à afficher",
            min_value=1,
            max_value=filtered_data.shape[0],
            value=min(20, filtered_data.shape[0])
        )
        
        # Limiter aux N premiers pays triés par NumericValue
        filtered_data = filtered_data.nlargest(top_n, 'NumericValue')

        if not filtered_data.empty:
            if chart_type == "Ligne":
                fig = px.line(filtered_data, x='SpatialDim', y='NumericValue', 
                              title=f"Comparaison de {selected_indicator} entre pays en {selected_year}",
                              labels={'SpatialDim': 'Pays', 'NumericValue': 'Valeur'})
                fig.update_traces(mode="lines+markers")
            else:  # Bar chart
                fig = px.bar(filtered_data, x='SpatialDim', y='NumericValue',
                             title=f"Comparaison de {selected_indicator} entre pays en {selected_year}",
                             labels={'SpatialDim': 'Pays', 'NumericValue': 'Valeur'})
                fig.update_layout(
                    xaxis=dict(
                        tickmode='linear',
                        automargin=True
                    ),
                    height=600,  # Ajuste la hauteur
                    width=1200   # Largeur suffisante pour l'ascenseur
                )
            
            st.plotly_chart(fig)
        else:
            st.write("Aucun pays ne correspond aux critères de filtre.")
    else:
        st.write(f"Aucune donnée disponible pour l'année {selected_year}.")        
              
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
if search_term and 'IndicatorName' in indicators_df.columns:
    filtered_indicators_df = indicators_df[indicators_df['IndicatorName'].str.contains(search_term, case=False, na=False)]
    
    # Sélection de l'indicateur
    selected_indicator = st.selectbox("Sélectionnez un indicateur pour voir les données détaillées", 
                                       filtered_indicators_df['IndicatorName'].tolist())
    indicator_code = filtered_indicators_df.loc[filtered_indicators_df['IndicatorName'] == selected_indicator, 'IndicatorCode'].values[0]

    # Récupération des données de l'indicateur sélectionné
    data_df = get_indicator_data(indicator_code)
    
    if not data_df.empty:
        # Vérification des valeurs numériques
        if has_numeric_values(data_df):
            # Choix du type de graphique à afficher
            graph_choice = st.radio("Choisissez le type de graphique à afficher", 
                                     ["Comparaison Entre Pays", "Évolution Temporelle"], key="graph_choice")

            if graph_choice == "Comparaison Entre Pays":
                st.subheader("Comparaison Entre Pays")
                # Sélecteur pour l'année
                years = sorted(data_df['TimeDim'].dropna().unique())
                selected_year = st.selectbox("Sélectionnez une année", years, key="year_selection")
                # Sélecteur pour le type de graphique
                chart_type_comparison = st.radio("Type de graphique pour la comparaison entre pays", ["Ligne", "Barres"], key="comparison")
                # Tracer le graphique
                plot_year_comparison(data_df, selected_year, selected_indicator, chart_type_comparison)

            elif graph_choice == "Évolution Temporelle":
                st.subheader("Évolution Temporelle")
                # Sélecteur pour le pays
                countries = data_df['SpatialDim'].unique()
                selected_country = st.selectbox("Sélectionnez un pays", countries, key="country_selection")
                # Sélecteur pour le type de graphique
                chart_type_time_series = st.radio("Type de graphique pour l'évolution temporelle", ["Ligne", "Barres"], key="time_series")
                # Tracer le graphique
                plot_country_time_series(data_df, selected_country, selected_indicator, chart_type_time_series)
            
        else:
            st.write("Cet indicateur ne contient pas de valeurs numériques et ne peut pas être analysé.")
    else:
        st.write("Aucune donnée disponible pour cet indicateur.")
else:
    st.write("Entrez un mot-clé pour rechercher un indicateur.")
