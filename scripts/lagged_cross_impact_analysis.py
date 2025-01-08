import pandas as pd


def create_lagged_features(data, lag_steps=1):
    """
    Crée des colonnes laggées pour chaque stock.

    Args:
        data (pd.DataFrame): Données OFI pivotées par stock.
        lag_steps (int): Nombre de lags à créer.

    Returns:
        pd.DataFrame: Données avec les colonnes laggées.
    """
    lagged_data = pd.DataFrame(index=data.index)
    for lag in range(1, lag_steps + 1):
        for col in data.columns:
            lagged_data[f'{col}_Lag{lag}'] = data[col].shift(lag)
    return lagged_data.dropna()