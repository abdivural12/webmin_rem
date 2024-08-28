import streamlit as st

st.set_page_config(page_title="Analyse ANOVA et Modèles", page_icon=":bar_chart:", layout="wide")

# Style de CSS
st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;  /* Pour que le texte par défaut soit blanc sur fond noir */
    }
    .title {
        font-size: 2.5rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 1rem;
        color: white;
        letter-spacing: 0.03rem;
    }
    .subtitle {
        font-size: 2.2rem;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 2rem;
        color: gray;
    }
    .logos-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: -20px;
    }
    .logos-container img {
        margin: 0 20px;
    }
    div.block-container {
        padding: 1rem;
    }
    .block {
        margin-bottom: 2rem;
    }
    .stColumn > div:first-child {
        margin-right: 0.5rem;
        margin-left: 0.5rem;
    }
    .stDataFrame {
        width: 100%;
    }
    h2 {
        font-size: 1.8rem !important;
        margin-bottom: 1rem;
        margin-top: 1rem;
        text-align: center;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre
st.markdown('<div class="title">Dashboard de webmining </div>', unsafe_allow_html=True)

# logos
st.markdown('<div class="logos-container"><img src="https://moodle.msengineering.ch/pluginfile.php/1/core_admin/logo/0x150/1643104191/logo-mse.png" width="250"><img src="https://www.hes-so.ch/typo3conf/ext/wng_site/Resources/Public/HES-SO/img/logo_hesso_master_tablet.svg" width="250"></div>', unsafe_allow_html=True)

# introduction
st.markdown('<div class="subtitle">Cette interface a été créée pour visualiser les résultats de l analyse des données collectées sur le site Airbnb. Pour une meilleure expérience de visualisation, veuillez ajuster le zoom de votre écran à 50 %.</div>', unsafe_allow_html=True)

import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.read_csv("data/total_out_clean.csv")


# Table ANOVA pour le prix
formula_price = 'price ~ old_price + C(Type) + C(pays) + C(region) + voyageurs + rooms + bed + bathroom'
model_price = ols(formula_price, data=data).fit()
anova_table_price = sm.stats.anova_lm(model_price, typ=2)

# Coefficients de facteurs pour le prix
coefficients_price = model_price.params

# Table ANOVA pour new_rating
formula_rating = 'new_rating ~ price + old_price + C(Type) + C(region) + voyageurs + rooms + bed + bathroom'
model_rating = ols(formula_rating, data=data).fit()
anova_table_rating = sm.stats.anova_lm(model_rating, typ=2)

# Première rangée 
col1, col2, col3 = st.columns(3, gap="small")

with col1:
    st.markdown("<h2>Table d'ANOVA pour le prix</h2>", unsafe_allow_html=True)
    st.dataframe(anova_table_price.style.format(precision=2), height=300)

with col2:
    st.markdown("<h2>Coefficients des facteurs pour le prix</h2>", unsafe_allow_html=True)
    st.dataframe(coefficients_price, height=300)

with col3:
    st.markdown("<h2>Table d'ANOVA pour new_rating</h2>", unsafe_allow_html=True)
    st.dataframe(anova_table_rating.style.format(precision=2), height=300)

# réduction de prix 
data['price_reduction'] = data['old_price'] - data['price']

#  réduction moyenne de prix par région 
region_reduction = data.groupby('region')['price_reduction'].mean().reset_index()

# Normalisation des données de réduction de prix
scaler = StandardScaler()
region_reduction_normalized = scaler.fit_transform(region_reduction[['price_reduction']])

# Nous appliquon K-means 
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
region_clusters = kmeans.fit_predict(region_reduction_normalized)

# Nous ajoutons les informations de cluster au dataframe
region_reduction['cluster'] = region_clusters

# Création  d'un graphique interactif avec bleu, rouge, et vert pour les clusters

fig1 = px.scatter(region_reduction, 
                 x=region_reduction.index, 
                 y='price_reduction', 
                 color=region_reduction['cluster'].astype(str),  
                 title='Clustering des régions par réduction de prix',
                 labels={'x': 'Region Index', 'price_reduction': 'Price Reduction'},
                 hover_data=['region'],
                 color_discrete_map={
                     '0': 'blue',
                     '1': 'green',
                     '2': 'red'
                 })

# Mettre à jour la mise en page pour ajuster le titre et les axes
fig1.update_layout(
    title={
        'text': 'Clustering des régions par réduction de prix',
        'font': {'size': 24},  # Agrandir la taille du titre
        'x': 0  # Centrer le titre
    },
    showlegend=True,
    xaxis_title='Region Index',
    yaxis_title='Price Reduction'
)


# Préparation des données pour la comparaison des modèles (après calculs et groupby)
data = pd.get_dummies(data, columns=['Type', 'pays', 'region'], drop_first=True)
X = data[['old_price', 'rooms', 'bed', 'bathroom'] + 
         [col for col in data.columns if 'Type_' in col or 'pays_' in col or 'region_' in col]]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Model 2: Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Model 3: XGBoost Regression
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Comparaison des modèles
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "Mean Squared Error": [mse_lr, mse_rf, mse_xgb],
    "R^2": [r2_lr, r2_rf, r2_xgb]
})

# Création des jauges pour R^2
fig = go.Figure()

# Jauge pour Linear Regression - R^2
fig.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=r2_lr,
    title={"text": "Linear Regression R²"},
    domain={'x': [0.0, 0.3], 'y': [0, 1]},
    gauge={'axis': {'range': [0, 1]}},
    #delta={'reference': 1, 'position': "top"}  # Optionnel, ajoute un delta (différence) visuel
))

# Jauge pour Random Forest - R^2
fig.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=r2_rf,
    title={"text": "Random Forest R²"},
    domain={'x': [0.35, 0.65], 'y': [0, 1]},
    gauge={'axis': {'range': [0, 1]}},
    #delta={'reference': 1, 'position': "top"}  # Optionnel, ajoute un delta (différence) visuel
))

# Jauge pour XGBoost - R^2
fig.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=r2_xgb,
    title={"text": "XGBoost R²"},
    domain={'x': [0.7, 1.0], 'y': [0, 1]},
    gauge={'axis': {'range': [0, 1]}},
    #delta={'reference': 1, 'position': "top"}  # Optionnel, ajoute un delta (différence) visuel
))

# Mise à jour de la mise en page pour les jauges, y compris la réduction de la taille du texte du titre
fig.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=50, b=20),
    title={'font': {'size': 1}}  # Réduire la taille du titre ici
)

# Deuxième rangée : Comparaison des modèles et jauges côte à côte avec largeur réduite
col4, col5 = st.columns([1, 2], gap="small")

with col4:
    st.markdown("<h2>Comparaison des Modèles</h2>", unsafe_allow_html=True)
    st.dataframe(comparison, height=250)

with col5:
    st.plotly_chart(fig, use_container_width=True)


# Création de scatter plot pour Predicted vs Actual Prices (fig2)
result_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred_rf})
fig2 = px.scatter(result_df, 
                  x='Actual Price', 
                  y='Predicted Price', 
                  title='Random Forest: Predicted vs Actual Prices',
                  labels={'Actual Price': 'Actual Price', 'Predicted Price': 'Predicted Price'},
                  trendline='ols')

# Agrandir le titre du graphique
fig2.update_layout(
    title={
        'text': 'Random Forest: Predicted vs Actual Prices',
        'font': {'size': 24},  # Agrandir la taille du titre
        'x': 0  # Centrer le titre
    }
)

# Troisième rangée : Visualisation de Clustering et Résultats du Modèle côte à côte
col6, col7 = st.columns(2, gap="small")

with col6:
    st.plotly_chart(fig1, use_container_width=True)

with col7:
    st.plotly_chart(fig2, use_container_width=True)

# Quatrième rangée : Carte de Visualisation de Clustering et Application de Prédiction de Prix côte à côte
col8, col9 = st.columns([1, 1], gap="small")  #  Nous ajustons la tailles de colonnes

# Carte de Clustering
with col8:
    st.title('Visualisation des Clusters Géographiques')
    data_map = pd.read_csv("data/region_coordinates.csv")
    reductions = pd.read_csv("data/total_out_clean.csv")


    # Calculer le pourcentage de réduction
    reductions['price_reduction'] = (reductions['old_price'] - reductions['price']) / reductions['old_price'] * 100

    # Ici nous regroupons les données par region en prenant la mayonne de la réduction pour chaque region
    region_reductions = reductions.groupby('region')['price_reduction'].mean().reset_index()

    # Appliquer le clustering K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    region_reductions['cluster'] = kmeans.fit_predict(region_reductions[['price_reduction']])

    # Fusionner avec les coordonnées géographiques
    merged_data = pd.merge(region_reductions, data_map, on='region', how='inner')

    # Création d'une carte Folium avec une taille ajustée
    map_clusters = folium.Map(location=[48.8566, 2.3522], zoom_start=5, width='100%', height='100%')  # Ajustement de la taille

    # Ajouter les points à la carte avec le montant de réduction dans les popups
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    for idx, row in merged_data.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=8,
            popup=f"Région: {row['region']}<br>Réduction: {row['price_reduction']:.2f}%",
            color=colors[row['cluster']],
            fill=True,
            fill_color=colors[row['cluster']]
        ).add_to(map_clusters)

    # Afficher la carte dans Streamlit
    folium_static(map_clusters, width=1200, height=700)  

# Application de Prédiction de Prix
with col9:
    st.title('Application de Prédiction de Prix')

    # Nous définisiion le mappage pays et region
    country_region_map = {
        'Allemagne': ['Munich', 'Berlin', 'Hamburg', 'Frankfurt', 'Cologne', 'Stuttgart', 'Düsseldorf'],
        'Autriche': ['Vienna', 'Salzburg', 'Graz', 'Innsbruck', 'Linz'],
        'Belgique': ['Brussels', 'Antwerp', 'Ghent', 'Charleroi', 'Liège'],
        'Espagne': ['Madrid', 'Barcelona', 'Valencia', 'Seville', 'Malaga'],
        'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice'],
        'Italie': ['Rome', 'Milan', 'Venice', 'Naples', 'Florence'],
        'Portugal': ['Lisbon', 'Porto', 'Braga', 'Faro', 'Coimbra'],
        'Suisse': ['Zurich', 'Geneva', 'Basel', 'Lausanne', 'Bern'],
    }

    # Champs d'entrée pour l'utilisateur
    old_price = st.number_input("Old Price", min_value=100.0, max_value=10000.0, value=3000.0)
    rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
    bed = st.number_input("Number of Beds", min_value=1, max_value=10, value=2)
    bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

    # Sélection pour les données catégorielles (Type, pays, région)
    Type = st.selectbox("Property Type", options=['apartment', 'loft', 'cottage'])
    pays = st.selectbox("Country", options=list(country_region_map.keys()))

    # Mise à jour dynamique des régions en fonction du pays sélectionné
    region = st.selectbox("Region", options=country_region_map[pays])

    # Convertir les entrées au format requis par le modèle
    input_data = pd.DataFrame({
        'old_price': [old_price],
        'rooms': [rooms],
        'bed': [bed],
        'bathroom': [bathroom],
        'Type_apartment': [1 if Type == 'apartment' else 0],
        'Type_loft': [1 if Type == 'loft' else 0],
        'Type_cottage': [1 if Type == 'cottage' else 0],
        'pays_Allemagne': [1 if pays == 'Allemagne' else 0],
        'pays_Autriche': [1 if pays == 'Autriche' else 0],
        'pays_Belgique': [1 if pays == 'Belgique' else 0],
        'pays_Espagne': [1 if pays == 'Espagne' else 0],
        'pays_France': [1 if pays == 'France' else 0],
        'pays_Italie': [1 if pays == 'Italie' else 0],
        'pays_Portugal': [1 if pays == 'Portugal' else 0],
        'pays_Suisse': [1 if pays == 'Suisse' else 0],
        f'region_{region}': [1]
    })

    # Remplissage des colonnes manquantes avec des zéros
    for col in X_train.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Réorganiser les colonnes pour correspondre aux données d'entraînement
    input_data = input_data[X_train.columns]

    # Faire des prédictions
    if st.button('Predict Price'):
        predicted_price = rf_model.predict(input_data)
        st.write(f"### Predicted Price: ${predicted_price[0]:,.2f}")
