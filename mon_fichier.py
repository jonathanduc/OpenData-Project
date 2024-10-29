#%%
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.set_page_config(page_title="OMS", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")
# Fonction pour récupérer la liste des indicateurs depuis l'API de l'OMS
def get_indicators():
        url = "https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,'Health')"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['value']
            indicators = {item['IndicatorName']: item['IndicatorCode'] for item in data}
            return indicators
        else:
            st.error("Erreur lors de la récupération des indicateurs.")
            return None

# Fonction pour récupérer les données de l'API de l'OMS pour un indicateur spécifique
def get_who_data(indicator_id):
    base_url = "https://ghoapi.azureedge.net/api/"
    url = f"{base_url}{indicator_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['value']  # Les données sont dans la clé 'value'
    else:
        st.error("Erreur lors de la récupération des données.")
        return None
# Fonction pour convertir les données en DataFrame
def convert_to_dataframe(data):
    records = []
    for entry in data:
        record = {
                'Country': entry.get('SpatialDim', 'N/A'),
                'Year': entry.get('TimeDim', 'null'),
                'Value': entry.get('Value', 'N/A')
            }
        records.append(record)
    return pd.DataFrame(records)
    
# Options de navigation
page = st.sidebar.selectbox("Sélectionnez une page", ["Accueil 🏠", "Analyse des données ⛑️📊 ","Machine Learning 📈📉", "À propos ℹ️"])

# Page d'accueil
if page == "Accueil 🏠":
    data = ''
    st.title("Accueil 🏠")
    st.title('Analyse des données de santé publique - API de l’OMS')
    st.write("Bienvenue sur l'application d'analyse de données de santé publique !")
    st.write("Cet outil récupère et affiche les données de santé publique de l'OMS.")
    st.image('https://upload.wikimedia.org/wikipedia/commons/3/3a/Logo_de_l%27Organisation_mondiale_de_la_santé.svg',caption='Logo de l\'OMS')
    video_url = "https://www.dailymotion.com/embed/video/x5npp4e"

    st.markdown(
        f"""
        <iframe src="{video_url}" width="1200" height="500" frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe>
        """,
        unsafe_allow_html=True
    )  

    st.markdown(
        """
        <div style="text-align: right;">
            <img src="https://portal-cdn.scnat.ch/asset/6fa448d9-8935-5622-aabf-a38072964841/2Personalisierte_Gesundheit.png?b=25632ea7-3d85-5198-a1f0-4ad8c9dcd6ac&v=df8f0af8-c032-5691-983d-84399f3dabd8_100&s=BemJi1u43IKhuBMmsD2ab6cA6zawwM6Nr_qD3i5tf6Aj4U4VLzDFpXxq4CBe-B-5Cfj0mHLCg1S89lSrv0ZhXRFcUUkGUGXlPIQrJePIWBtf3kwOZ1sHagPcP8_VmpKU8SOWNLpIKLcQUJrG66a27HSo7itvX6f0pdatcpVpnjc&t=fc13185f-cc70-4eb1-86f2-ea80914407bc&sc=2" width="400">
        </div>
        """,
        unsafe_allow_html=True
    )      
# Analyse des données
elif page == "Analyse des données ⛑️📊 ":
    st.title("Analyse des données de santé publique ⛑️📊")

    # Récupérer les indicateurs et les afficher dans une liste déroulante
    indicators = get_indicators()
    if indicators:
        indicator_name = st.selectbox('Sélectionnez un indicateur', list(indicators.keys()))
        indicator_id = indicators[indicator_name]
        
        st.write("Vous avez choisi l'indicateur :", indicator_name)

        # Bouton pour analyser les données
        if st.toggle("Analyser les données"):
            data = get_who_data(indicator_id)
            if data:
                st.write("Données brutes :")
                df = convert_to_dataframe(data)

                # Traitement de la colonne Year en format numérique et des valeurs 'No d'
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df['Value'] = df['Value'].replace('No d', 0).astype(float)

                # Affichage des données brutes et analyses descriptives
                with st.expander("Voir les données brutes"):
                    st.dataframe(df, use_container_width=True)
                
                with st.expander("Voir les analyses descriptives"):
                    st.dataframe(df.describe(), use_container_width=True)

                # Sélection du pays
                select_country = st.selectbox('Sélectionnez un pays', sorted(df['Country'].unique()))
                st.write(f"Vous avez choisi le pays : {select_country}")

                # Filtrer le DataFrame pour le pays sélectionné
                country_df = df[df['Country'] == select_country]

                if st.toggle('Voir les graphiques'):
                    # Distribution des valeurs
                    st.write('Distribution des données :')
                    st.bar_chart(country_df['Value'].value_counts())

                    # Sélection du type de graphique
                    select_graph = st.selectbox('Sélectionnez un type de graphique', ['Bar', 'Line'])
                    # Préparation des données pour le graphique par année
                    group = country_df.groupby('Year').sum('Value').reset_index()
                    st.write(f"Graphique de l'évolution de {indicator_name} par année pour {select_country}")

                    # Affichage du graphique sélectionné
                    if select_graph == 'Bar':
                        st.bar_chart(data=group, x='Year', y='Value')
                    elif select_graph == 'Line':
                        st.line_chart(data=group, x='Year', y='Value')
                else:
                    st.write("Aucune donnée trouvée pour cet indicateur.ERRRROR")
            else:
                st.write("Aucune donnée trouvée pour cet indicateur.")
#Machine Learning
elif page == "Machine Learning 📈📉":
    st.title("Machine Learning 📈📉")

     # Récupérer les indicateurs et les afficher dans une liste déroulante
    indicators = get_indicators()
    if indicators:
        indicator_name = st.selectbox('Sélectionnez un indicateur', list(indicators.keys()))
        indicator_id = indicators[indicator_name]
        
        st.write("Vous avez choisi l'indicateur :", indicator_name)

        # Bouton pour analyser les données
        if st.toggle("Analyses ML"):
            data = get_who_data(indicator_id)
            if data:
                st.write("Données brutes :")
                df = convert_to_dataframe(data)

                # Traitement de la colonne Year en format numérique et des valeurs 'No d'
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df['Value'] = df['Value'].replace('No d', 0).astype(float)

                # Affichage des données brutes et analyses descriptives
                with st.expander("Voir les données brutes"):
                    st.dataframe(df, use_container_width=True)

                if st.toggle('Voir la régression linéaire'):

                    # Préparation des données pour le graphique par année
                    data = df.groupby('Year').sum()
                    data = data.reset_index()
                    data = data[['Year','Value']]
                    

                    linear_model = LinearRegression()
                    linear_model.fit(data[['Year']], data['Value'])
                    score = linear_model.score(data[['Year']], data['Value'])
                    coeff = linear_model.coef_
                    intercept = linear_model.intercept_

                    st.write(f'#### Coefficient : {linear_model.coef_}')
                    st.write(f'#### Intercept : {linear_model.intercept_}')
                    st.write(f'#### Score R2 : {score}')


                    # Préparation des prédictions
                    years_future = np.array([[2020], [2021], [2022], [2023], [2024], [2025]])
                    predictions = linear_model.predict(years_future)

                    # Initialisation du graphique Plotly
                    fig = go.Figure()

                    # Ajout des données réelles
                    fig.add_trace(go.Scatter(x=data['Year'], y=data['Value'], mode='markers', name='Données réelles', marker=dict(color='blue')))

                    # Ajout de la ligne de régression
                    fig.add_trace(go.Scatter(x=data['Year'], y=linear_model.predict(data[['Year']]), mode='lines', name='Régression linéaire', line=dict(color='red')))

                    # Ajout des prédictions futures
                    fig.add_trace(go.Scatter(x=years_future.flatten(), y=predictions, mode='lines+markers', name='Prédictions futures', line=dict(dash='dash', color='green')))

                    # Mise en forme du graphique
                    fig.update_layout(
                        title="Régression linéaire et prédictions futures",
                        xaxis_title="Année",
                        yaxis_title="Valeur",
                        template="plotly_white"
                    )

                    # Interface utilisateur Streamlit
                    st.title("Application de Régression Linéaire et Prédictions")
                    st.write("Visualisation des données, de la régression linéaire et des prédictions futures.")
                    st.plotly_chart(fig, use_container_width=True)

            if st.toggle('Voir le Clustering :'): 
                st.title('Clustering')
                # Préparation des données
                scaler = StandardScaler()
                data2 = df.groupby('Country').sum()
                data2 = data2.reset_index()

                # Normalisation des valeurs
                data_scaled = scaler.fit_transform(data2[['Value']])

                # Clustering avec KMeans
                kmeans = KMeans(n_clusters=3, random_state=42)
                kmeans.fit(data_scaled)
                data2['Cluster'] = kmeans.labels_

                # Création de la figure Plotly
                fig = go.Figure()
                for cluster in data2['Cluster'].unique():
                    cluster_data = data2[data2['Cluster'] == cluster]
                    fig.add_trace(go.Scatter(
                        x=cluster_data['Country'],  # Nom des pays en axe x
                        y=cluster_data['Value'],    # Valeurs originales en axe y
                        mode='markers', 
                        name=f'Cluster {cluster}',
                        hovertext=cluster_data['Country'],  # Nom du pays pour le survol
                        hoverinfo="text"  # Affiche uniquement le nom du pays au survol
                    ))

                # Mise en forme de la figure
                fig.update_layout(
                    title="Clustering des données par pays",
                    xaxis_title="Pays",
                    yaxis_title="Valeur",
                    template="plotly_white"
                )

                # Affichage du graphique dans Streamlit
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.write("Aucune donnée trouvée pour cet indicateur.")
#A propos 
elif page == "À propos ℹ️":
    st.title("À propos ℹ️")
    st.write("Cet outil a été créé dans le cadre du cours d'Open data et Web des données.")
    st.page_link("https://github.com/jonathanduc/OpenData-Project", label = "Lien vers le projet sur GitHub", icon= "🔗")
    image = 'data:image/png;base64,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'
    st.page_link("https://www.who.int/data/gho", label = "Lien vers l'API de l'OMS", icon= "🔗")
    st.page_link('https://docs.streamlit.io', label='Lien vers la documentation de Streamlit', icon='🔗')
    st.write("## À propos des développeurs")

# Ajout des liens GitHub en utilisant Markdown
    st.markdown(
        """
        **[🔗 Jonathan Duckes](https://github.com/jonathanduc)**  
        **[🔗 Girondin Audric](https://github.com/aaudric)**
        """
    )
    texte = """ Nous sommes deux étudiants en Master 2 et avons entrepris ce projet dans le cadre de notre formation afin de renforcer nos compétences en développement d’applications et de visualisation de données. L'objectif de ce projet est de concevoir un dashboard interactif en utilisant Streamlit, une bibliothèque Python spécialisée dans la création d'interfaces web.

Le dashboard vise à fournir une visualisation claire et interactive des données sur un sujet spécifique, avec des indicateurs clés, des graphiques et une interactivité grâce à des widgets intégrés. Pour garantir des données à jour et pertinentes, nous utilisons une API pour récupérer dynamiquement le dataset nécessaire.

Notre travail est structuré autour de plusieurs éléments essentiels :
- Un texte explicatif qui clarifie l'objectif du dashboard et son utilité pour l'utilisateur.
- L'intégration d'une API pour importer les données en temps réel.
- Des indicateurs et des graphiques permettant une analyse visuelle des informations.
- Des fonctionnalités interactives qui permettent aux utilisateurs d'explorer les données en fonction de leurs besoins spécifiques.

Ce projet est une opportunité pour nous de mettre en pratique les compétences acquises en data science et en développement, tout en créant un outil utile et fonctionnel pour une meilleure compréhension des données.
"""
    st.write(texte)
