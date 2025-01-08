import pandas as pd
from sklearn.linear_model import LinearRegression
from .lagged_cross_impact_analysis import create_lagged_features
from sklearn.model_selection import train_test_split
import numpy as np

import warnings
from pandas.errors import PerformanceWarning

# Ignorer les PerformanceWarning
warnings.simplefilter(action='ignore', category=PerformanceWarning)


def calculate_minute_price_changes_with_filters(full_data, integrated=True):
    """
    Calcule les variations de prix à la minute pour chaque action, en excluant les 30 premières et dernières minutes
    de la journée de trading, et en se limitant aux fenêtres de 30 minutes entre 10:00 et 15:30.

    Args:
        full_data (pd.DataFrame): Données contenant les colonnes 'ts_event', 'symbol', 'bid_px_00', 'ask_px_00', et 'Integrated_OFI'.

    Returns:
        pd.DataFrame: DataFrame filtré et avec les variations de prix par minute.
    """
    # Convertir les timestamps en datetime
    full_data['ts_event'] = pd.to_datetime(full_data['ts_event'])

    # Ajouter une colonne pour grouper par minute
    full_data['minute'] = full_data['ts_event'].dt.floor('min')

    # Calculer le prix moyen bid-ask par ligne
    full_data['mid_price'] = (full_data['bid_px_00'] + full_data['ask_px_00']) / 2

    # Exclure les 30 premières et dernières minutes de chaque journée
    full_data['time_only'] = full_data['ts_event'].dt.time
    trading_start = pd.to_datetime("10:00:00").time()
    trading_end = pd.to_datetime("15:30:00").time()

    filtered_data = full_data[
        (full_data['time_only'] >= trading_start) & (full_data['time_only'] <= trading_end)
    ]

    # Agréger les données par minute pour chaque symbole
    if integrated:
        # Utiliser 'Integrated_OFI'
        minute_price_df = (
            filtered_data.groupby(['symbol', 'minute'])
            .agg({
                'mid_price': 'last',  # Prendre le dernier prix moyen de la minute
                'Integrated_OFI': 'sum'  # Somme de l'OFI intégré sur la minute
            })
            .reset_index()
        )
    else:
        # Utiliser les colonnes 'nOFI_x'
        nOFI_columns = [col for col in full_data.columns if col.startswith('nOFI_')]
        minute_price_df = (
            filtered_data.groupby(['symbol', 'minute'])
            .agg({
                'mid_price': 'last',  # Prendre le dernier prix moyen de la minute
                **{col: 'sum' for col in nOFI_columns}  # Somme des nOFI_x sur la minute
            })
            .reset_index()
        )

    # Ajouter les variations de prix
    price_changes = (
        minute_price_df.groupby('symbol')
        .apply(
            lambda group: group.assign(price_change=np.log(group['mid_price']) - np.log(group['mid_price'].shift(1))))
        .reset_index(drop=True)
    )

    # Fusionner les variations de prix avec le DataFrame agrégé
    minute_price_df = minute_price_df.merge(
        price_changes[['symbol', 'minute', 'price_change']],
        on=['symbol', 'minute'],
        how='left'
    )

    return minute_price_df.dropna()


def analyze_impact_with_integrated_ofi(full_data, lagged=False, lag_steps=1, cross_impact=True):
    """
    Analyse l'impact croisé en utilisant l'OFI intégré et les variations de prix.

    Args:
        full_data (pd.DataFrame): Données contenant les colonnes 'symbol', 'minute',
                                  'Integrated_OFI', 'price_change'.
        lagged (bool): Si True, utilise les OFI décalés dans le temps comme prédicteurs.
        lag_steps (int): Nombre de décalages temporels à appliquer pour les OFI laggés.

    Returns:
        dict: Résultats des régressions pour chaque symbole cible.
    """
    results = {}
    symbols = full_data['symbol'].unique()

    # Créer un DataFrame pivoté des OFI normaux
    pivoted_ofi = full_data.pivot(index='minute', columns='symbol', values='Integrated_OFI').dropna()

    # Si lagged est activé, créer les fonctionnalités laggées
    if lagged:
        lagged_ofi = create_lagged_features(pivoted_ofi, lag_steps=lag_steps)
        pivoted_ofi = pivoted_ofi.reindex(lagged_ofi.index)  # Aligner les indices des données laggées et normales
        combined_ofi = pd.concat([pivoted_ofi, lagged_ofi], axis=1)
    else:
        combined_ofi = pivoted_ofi

    for target_symbol in symbols:
        # Variables explicatives : OFI intégré des autres symboles
        if cross_impact:
            if lagged :
                # Inclut uniquement les OFI laggés pour les autres symboles
                X = combined_ofi.loc[:, [col for col in combined_ofi.columns if f"_Lag" in col]]
            else :
                X = combined_ofi.copy()
        else:
            if lagged:
                # Inclut uniquement les OFI laggés pour le symbole cible
                X = combined_ofi.loc[:, [col for col in combined_ofi.columns if f"{target_symbol}_Lag" in col]]
            else:
                # Inclut uniquement l'OFI intégré pour le symbole cible
                X = combined_ofi.loc[:, [target_symbol]]

        # Variable cible : variation des prix pour le symbole cible
        y = full_data[full_data['symbol'] == target_symbol][['minute', 'price_change']].dropna()
        y = y.set_index('minute')['price_change']

        # Aligner les indices
        common_index = X.index.intersection(y.index)
        if common_index.empty:
            print(f"Pas de données communes pour {target_symbol}. Ignoré.")
            continue

        X = X.loc[common_index].fillna(X.mean())
        y = y.loc[common_index]
        y = y + np.random.normal(0, 1e-6, len(y)) # add noise

        if X.shape[0] == 0 or y.shape[0] == 0:
            print(f"Pas assez de données pour {target_symbol}. Ignoré.")
            continue

        # Diviser en données d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Régression linéaire
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        is_r2 = 0
        oos_r2 = 0

        if lagged:
            # Prédictions sur l'ensemble de test
            y_test_pred = reg.predict(X_test)

            oos_r2 = 1 - (np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - y_train.mean()) ** 2))
        else:
            # Prédictions sur l'ensemble d'entraînement
            y_train_pred = reg.predict(X_train)

            is_r2 = 1 - (np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - y_train.mean()) ** 2))

        # Prédictions sur l'ensemble de test pour évaluer le R² classique
        r2 = reg.score(X_test, y_test)

        # Sauvegarder les résultats
        results[target_symbol] = {
            'coefficients': reg.coef_,
            'intercept': reg.intercept_,
            'R^2': r2,
            'IS R^2': is_r2,
            'OOS R^2': oos_r2,
            'features': X.columns.tolist(),
        }

    return results


def analyze_impact_with_multi_level_ofi(full_data, lagged=False, lag_steps=1, cross_impact=True):
    """
    Analyse l'impact croisé en utilisant les niveaux de nOFI et les variations de prix.

    Args:
        full_data (pd.DataFrame): Données contenant les colonnes 'symbol', 'minute',
                                  'nOFI_0', 'nOFI_1', ..., 'nOFI_4', 'price_change'.
        lagged (bool): Si True, utilise les nOFI décalés dans le temps comme prédicteurs.
        lag_steps (int): Nombre de décalages temporels à appliquer pour les nOFI laggés.
        cross_impact (bool): Si True, inclut les impacts croisés entre les actifs.

    Returns:
        dict: Résultats des régressions pour chaque symbole cible.
    """
    results = {}
    symbols = full_data['symbol'].unique()

    # Créer un DataFrame pivoté pour chaque nOFI_x
    nOFI_columns = [col for col in full_data.columns if col.startswith('nOFI_')]
    pivoted_ofi = {}
    for col in nOFI_columns:
        pivoted_ofi[col] = full_data.pivot(index='minute', columns='symbol', values=col).dropna()

    # Si lagged est activé, créer les fonctionnalités laggées
    if lagged:
        lagged_ofi = {}
        for col, df in pivoted_ofi.items():
            lagged_ofi[col] = create_lagged_features(df, lag_steps=lag_steps)
            pivoted_ofi[col] = df.reindex(lagged_ofi[col].index)  # Aligner les indices des données laggées et normales
        combined_ofi = pd.concat([pivoted_ofi[col] for col in nOFI_columns] +
                                 [lagged_ofi[col] for col in nOFI_columns], axis=1)
    else:
        combined_ofi = pd.concat([pivoted_ofi[col] for col in nOFI_columns], axis=1)

    for target_symbol in symbols:
        # Variables explicatives : nOFI des autres symboles
        if cross_impact:
            if lagged:
                # Inclut uniquement les nOFI laggés pour les autres symboles
                X = combined_ofi.loc[:, [col for col in combined_ofi.columns if f"_Lag" in col]]
            else:
                X = combined_ofi.copy()
        else:
            if lagged:
                # Inclut uniquement les nOFI laggés pour le symbole cible
                X = combined_ofi.loc[:, [col for col in combined_ofi.columns if f"{target_symbol}_Lag" in col]]
            else:
                # Inclut uniquement les nOFI pour le symbole cible
                X = combined_ofi.loc[:, [col for col in combined_ofi.columns if f"{target_symbol}" in col]]

        # Variable cible : variation des prix pour le symbole cible
        y = full_data[full_data['symbol'] == target_symbol][['minute', 'price_change']].dropna()
        y = y.set_index('minute')['price_change']

        # Aligner les indices
        common_index = X.index.intersection(y.index)
        if common_index.empty:
            print(f"Pas de données communes pour {target_symbol}. Ignoré.")
            continue

        X = X.loc[common_index].fillna(X.mean())
        y = y.loc[common_index]
        y = y + np.random.normal(0, 1e-6, len(y))  # Ajouter un bruit aléatoire pour éviter les singularités

        if X.shape[0] == 0 or y.shape[0] == 0:
            print(f"Pas assez de données pour {target_symbol}. Ignoré.")
            continue

        # Diviser en données d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Régression linéaire
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        is_r2 = 0
        oos_r2 = 0

        if lagged:
            # Prédictions sur l'ensemble de test
            y_test_pred = reg.predict(X_test)

            oos_r2 = 1 - (np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - y_train.mean()) ** 2))
        else:
            # Prédictions sur l'ensemble d'entraînement
            y_train_pred = reg.predict(X_train)

            is_r2 = 1 - (np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - y_train.mean()) ** 2))

        # Prédictions sur l'ensemble de test pour évaluer le R² classique
        r2 = reg.score(X_test, y_test)

        # Sauvegarder les résultats
        results[target_symbol] = {
            'coefficients': reg.coef_,
            'intercept': reg.intercept_,
            'R^2': r2,
            'IS R^2': is_r2,
            'OOS R^2': oos_r2,
            'features': X.columns.tolist(),
        }

    return results

