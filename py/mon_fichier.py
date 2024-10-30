#%%
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(page_title="OMS", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# Fonction pour récupérer la liste des indicateurs contenant "Health" dans le nom
def get_indicators():
    url = "https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,'Health')"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get('value', [])
        indicators = {item['IndicatorName']: item['IndicatorCode'] for item in data}
        return indicators
    else:
        print("Erreur lors de la récupération des indicateurs.")
        return None

# Fonction pour récupérer les données d'un indicateur spécifique
def get_who_data(indicator_id):
    base_url = "https://ghoapi.azureedge.net/api/"
    url = f"{base_url}{indicator_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json().get('value', [])
        
        # Filtrer les données pour ne garder que celles où NumericValue est non nul
        filtered_data = [entry for entry in data if entry.get('NumericValue') is not None]
        
        # Retourner les données filtrées si elles existent
        return filtered_data if filtered_data else None
    else:
        print("Erreur lors de la récupération des données.")
        return None

# Fonction pour obtenir les indicateurs avec valeurs NumericValue non vides
def get_indicators_with_numeric_value(limit=10):
    indicators = get_indicators()
    valid_indicators = {}
    
    if indicators:
        with st.spinner('Vérification des indicateurs pour NumericValue...'):
            for i, (name, code) in enumerate(indicators.items()):
                if i >= limit:  # Limiter le nombre d'indicateurs vérifiés
                    break
                data = get_who_data(code)
                if data:  # Si get_who_data retourne des données avec NumericValue non vide
                    valid_indicators[name] = code
    
    return valid_indicators
# Fonction pour convertir les données en DataFrame
def convert_to_dataframe(data):
    records = []
    for entry in data:
        record = {
                'Country': entry.get('SpatialDim', 'N/A'),
                'Continent': entry.get('ParentLocation', 'N/A'),
                'Year': entry.get('TimeDimensionValue', 'null'),
                'Value': entry.get('NumericValue', 'N/A')
            }
        records.append(record)
    return pd.DataFrame(records)
    
# Options de navigation
page = st.sidebar.selectbox("Sélectionnez une page", ["Accueil 🏠", "Analyse des données ⛑️📊 ","📉 Machine Learning 📈", "À propos ℹ️"])

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
    nb_indicators = st.number_input("Nombre d'indicateurs à afficher", 1, 30, 5)
     # Récupérer les indicateurs et les afficher dans une liste déroulante

    indicators = get_indicators_with_numeric_value(limit=nb_indicators)
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
                df['Year'] = pd.to_datetime(df['Year'], format= '%Y').dt.to_period('Y')
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
elif page == "📉 Machine Learning 📈":
    st.title("📉 Machine Learning 📈")
    nb_indicators = st.number_input("Nombre d'indicateurs à afficher", 1, 30, 5)
     # Récupérer les indicateurs et les afficher dans une liste déroulante
    indicators = get_indicators_with_numeric_value(limit=nb_indicators)
    if indicators:
        indicator_name = st.selectbox('Sélectionnez un indicateur', list(indicators.keys()))
        indicator_id = indicators[indicator_name]
        st.write("Vous avez choisi l'indicateur :", indicator_id)
        
        st.write("Vous avez choisi l'indicateur :", indicator_name)

        # Bouton pour analyser les données
        if st.toggle("Analyses ML"):
            data = get_who_data(indicator_id)
            if data:
                st.write("Données brutes :")
                df = convert_to_dataframe(data)

                # Traitement de la colonne Year en format numérique et des valeurs 'No d'
                df['Year'] = pd.to_datetime(df['Year'], format= '%Y').dt.to_period('Y')
                df['Value'] = df['Value'].replace('No d', 0).astype(float)

                # Affichage des données brutes et analyses descriptives
                with st.expander("Voir les données brutes"):
                    st.dataframe(df, use_container_width=True)

                df['Year'] = df['Year'].dt.year

                if st.toggle('Régression Linéaire'):

                    st.subheader(':blue[Régression Linéaire]')

                    # Préparation des données pour le graphique par année
                    data = df.groupby('Year').sum()
                    data = data.reset_index()
                    data = data[['Year','Value']]
                    
    
                    linear_model = LinearRegression()

                    linear_model.fit(data[['Year']], data['Value'])
                    score = round(linear_model.score(data[['Year']], data['Value']),4)
                    rounded_coefficient = round(linear_model.coef_[0], 4)
                    intercept = round(linear_model.intercept_,4)

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
                         title={
                            "text": "Régression linéaire et prédictions futures",
                            "x": 0.5,  # Centre le titre horizontalement
                            "xanchor": "center",  # Ancre le texte du titre au centre
                            "font": {"size": 20}
                        },
                        xaxis={
                            "title": {
                                "text": "Année",
                                "font": {"size": 16}  # Taille du titre de l'axe X
                            }
                        },
                        yaxis={
                            "title": {
                                "text": "Valeur",
                                "font": {"size": 16}  # Taille du titre de l'axe Y
                            }
                        },
                        template="plotly_white"
                    )

                    # Interface utilisateur Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric('#### Coefficient', rounded_coefficient)
                    col2.metric('#### Intercept', intercept)
                    col3.metric('#### Score R2', score)
                   

                if st.toggle('Clustering'): 

                    st.subheader(':blue[Clustering]')

                    data2 = df.copy()
                    data2 = data2.groupby(['Country','Continent']).sum('Value')
                    data2 = data2.reset_index()

                    # Encodage des continents
                    label_encoder = LabelEncoder()
                    data2['Continent_encoded'] = label_encoder.fit_transform(data2['Continent'])

                    # Conservez les colonnes nécessaires pour le clustering
                    data2_for_clustering = data2[['Value', 'Continent_encoded']]

                    # Normalisation des données
                    scaler = StandardScaler()
                    data2_scaled = scaler.fit_transform(data2_for_clustering)

                    # Réduction de dimension avec PCA (1 composante principale)
                    pca = PCA(n_components=2)
                    data2_pca = pca.fit_transform(data2_scaled)

                    explained_variance = pca.explained_variance_ratio_ # Variance expliquée par chaque composante principale


                    # Méthode du coude
                    inertia = []
                    cluster_range = range(1, 11)  # Tester jusqu'à 10 clusters
                    for k in cluster_range:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(data2_pca)
                        inertia.append(kmeans.inertia_)

                    # Créez la figure de la méthode du coude
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(cluster_range, inertia, marker='o', linestyle='-')
                    ax.set_xlabel("Nombre de clusters", fontsize=16)
                    ax.set_ylabel("Inertie (somme des distances au carré)", fontsize=16)
                    ax.set_title("Méthode du coude pour le choix du nombre de clusters", fontsize=20, loc='center')
                    ax.grid(True)

                    # Affichez la figure avec Streamlit
                    st.plotly_chart(fig, clear_figure=True)
                    

                    n_cluster = st.select_slider('Choisir le nombre de clusters', options=[2, 3, 4, 5, 6, 7, 8, 9, 10], value=3)

                    # Clustering avec KMeans
                    kmeans = KMeans(n_clusters= n_cluster, random_state=42)
                    kmeans.fit(data2_pca)
                    data2['Cluster'] = kmeans.labels_
                    data2 = data2[['Country', 'Value', 'Cluster', 'Continent', 'Continent_encoded']]
                    
                    
                    # Création de la figure Plotly
                    fig = go.Figure()
                    for cluster in sorted(data2['Cluster'].unique()):
                        cluster_data = data2[data2['Cluster'] == cluster]
                        # Utilisez les valeurs de PCA pour les axes x et y
                        fig.add_trace(go.Scatter(
                            x=data2_pca[data2['Cluster'] == cluster][:, 0],  # Première composante principale
                            y=data2_pca[data2['Cluster'] == cluster][:, 1],  # Deuxième composante principale
                            mode='markers',
                            name=f'Cluster n°{cluster +1}',
                            marker=dict(symbol='circle', size=10),
                            text=cluster_data['Country'],  # Nom du pays pour le survol
                            hoverinfo="text"  # Affiche uniquement le nom du pays au survol
                        ))

                    # Mise en forme de la figure
                    fig.update_layout(
                        title={
                            "text": "Clustering des pays en fonction des deux premières composantes principales",
                            "x": 0.5,  # Centre le titre horizontalement
                            "xanchor": "center",  # Ancre le texte du titre au centre
                            "font": {"size": 20}
                        },
                        xaxis={
                            "title": {
                                "text": f"Première composante principale ({explained_variance[0]:.2%} de variance expliquée)",
                                "font": {"size": 16}  # Taille du titre de l'axe X
                            }
                        },
                        yaxis={
                            "title": {
                                "text": f"Deuxième composante principale ({explained_variance[1]:.2%} de variance expliquée)",
                                "font": {"size": 16}  # Taille du titre de l'axe Y
                            }
                        },
                        template="plotly_white"
                    )

                    # Affichage du graphique dans Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("### Shape des clusters")
                    col = st.columns(n_cluster)
                    for i in range(n_cluster) : 
                        col[i].metric(f'#### Cluster n°{i+1}', data2[data2['Cluster'] == i].shape[0])
            else : 
                st.write("Aucune donnée trouvée pour cet indicateur.")
#A propos 
elif page == "À propos ℹ️":
    st.title("À propos ℹ️")
    st.write("Cet outil a été créé dans le cadre du cours d'Open data et Web des données.")
    st.page_link("https://github.com/jonathanduc/OpenData-Project", label = "Lien vers le projet sur GitHub", icon= "🔗")
    st.page_link("https://www.who.int/data/gho", label = "Lien vers l'API de l'OMS", icon= "🔗")
    st.page_link('https://docs.streamlit.io', label='Lien vers la documentation de Streamlit', icon='🔗')
    st.write("## À propos des développeurs")

# Ajout des liens GitHub en utilisant Markdown
    st.markdown(
        """
        **[🔗 Duckes Jonathan](https://github.com/jonathanduc)**  
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
