from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .lagged_cross_impact_analysis import create_lagged_features
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np


def prepare_clustering_data(cross_impact_results):
    """
    Prépare les données pour le clustering à partir des coefficients d'impact croisé.

    Args:
        cross_impact_results (dict): Résultats de régression contenant les coefficients et les caractéristiques.

    Returns:
        pd.DataFrame: Matrice des coefficients d'impact croisé, où les lignes représentent les actions.
    """
    # Initialiser une matrice avec les actions comme lignes et coefficients comme colonnes
    stocks = list(cross_impact_results.keys())
    clustering_data = pd.DataFrame(index=stocks)

    for stock, results in cross_impact_results.items():
        for predictor, coefficient in zip(results['features'], results['coefficients']):
            clustering_data.loc[stock, predictor] = coefficient

    # Remplir les valeurs manquantes par 0
    clustering_data = clustering_data.fillna(0)

    return clustering_data


def perform_kmeans_clustering(data, n_clusters=3):
    """
    Effectue un clustering K-Means.

    Args:
        data (pd.DataFrame): Données pour le clustering.
        n_clusters (int): Nombre de clusters.

    Returns:
        tuple: Labels des clusters et l'objet KMeans.
    """
    # Standardiser les données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Appliquer K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)

    return cluster_labels, kmeans


def perform_hierarchical_clustering(data):
    """
    Effectue un clustering hiérarchique.

    Args:
        data (pd.DataFrame): Données pour le clustering.

    Returns:
        ndarray: Matrice de linkage pour le clustering hiérarchique.
    """
    # Standardiser les données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Calculer la matrice de linkage
    linkage_matrix = linkage(scaled_data, method='ward')

    return linkage_matrix


def plot_pca_clusters(data, cluster_labels):
    """
    Trace un scatter plot des clusters avec PCA pour réduction de dimension.

    Args:
        data (pd.DataFrame): Données originales pour le clustering.
        cluster_labels (array-like): Labels des clusters pour chaque action.
    """
    # Réduire la dimensionnalité avec PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Plot des clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=cluster_labels, palette="viridis", s=100)
    plt.title("Clustering Results (PCA Projection)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(title="Cluster")
    plt.show()


def plot_dendrogram(linkage_matrix, labels):
    """
    Trace un dendrogramme pour le clustering hiérarchique.

    Args:
        linkage_matrix (ndarray): Matrice de linkage.
        labels (list): Noms des actions.
    """
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=45, leaf_font_size=10)
    plt.title("Dendrogram of Stock Clustering", fontsize=14)
    plt.xlabel("Stocks", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.show()


def analyze_cross_impact_with_xgboost(full_data, lagged=False, lag_steps=1):
    """
    Analyse l'impact croisé en utilisant XGBoost pour une régression.

    Args:
        full_data (pd.DataFrame): Données contenant les colonnes 'symbol', 'minute',
                                  'Integrated_OFI', 'price_change'.
        lagged (bool): Si True, utilise les OFI décalés dans le temps comme prédicteurs.
        lag_steps (int): Nombre de décalages temporels à appliquer pour les OFI laggés.

    Returns:
        dict: Résultats des modèles XGBoost pour chaque symbole cible, incluant les métriques R².
    """
    results = {}
    symbols = full_data['symbol'].unique()

    # Créer un DataFrame pivoté des OFI
    pivoted_ofi = full_data.pivot(index='minute', columns='symbol', values='Integrated_OFI')

    # Si lagged est activé, créer les fonctionnalités laggées
    if lagged:
        lagged_ofi = create_lagged_features(pivoted_ofi, lag_steps=lag_steps)
        pivoted_ofi = pivoted_ofi.reindex(lagged_ofi.index)  # Aligner les indices des données laggées et normales
        combined_ofi = pd.concat([pivoted_ofi, lagged_ofi], axis=1)
    else:
        combined_ofi = pivoted_ofi

    for target_symbol in symbols:
        # Variables explicatives : OFI intégré des autres symboles
        X = combined_ofi.copy()

        # Variable cible : variation des prix pour le symbole cible
        y = full_data[full_data['symbol'] == target_symbol][['minute', 'price_change']].dropna()
        y = y.set_index('minute')['price_change']

        # Vérifier les indices
        common_index = X.index.intersection(y.index)  # Intersection des indices
        if common_index.empty:
            print(f"Pas de données communes pour {target_symbol}. Ignoré.")
            continue

        X = X.loc[common_index].fillna(0)
        y = y.loc[common_index]

        # Diviser les données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Configuration de XGBoost pour une régression
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'reg:squarederror',  # Régression
            'eval_metric': 'rmse',           # Root Mean Square Error comme métrique d'évaluation
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }

        # Entraîner le modèle
        model = xgb.train(params, dtrain, num_boost_round=100)

        # Prédictions
        y_pred = model.predict(dtest)

        # Calcul des métriques
        oos_r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_train.mean()) ** 2))  # OOS R²
        r2 = r2_score(y_test, y_pred)  # R² classique

        # Sauvegarder les résultats
        results[target_symbol] = {
            'model': model,
            'R^2': r2,
            'OOS R^2': oos_r2,
            'features': X.columns.tolist(),
            'y_test': y_test,
            'y_pred': y_pred
        }

    return results