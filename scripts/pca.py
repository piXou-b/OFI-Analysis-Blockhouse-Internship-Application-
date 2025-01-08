from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def integrate_ofi_with_pca(ofi_df):
    """
    Intègre les OFI multi-niveaux en une seule métrique via PCA.

    Args:
        ofi_df (pd.DataFrame): DataFrame contenant les colonnes nOFI_0 à nOFI_4.

    Returns:
        pd.Series: Série contenant l'Integrated OFI calculé pour chaque ligne.
    """
    # Sélectionner les colonnes des OFI normalisés
    ofi_columns = [f'nOFI_{level}' for level in range(5)]
    ofi_data = ofi_df[ofi_columns]

    # Vérifiez si toutes les colonnes existent
    if not all(col in ofi_df.columns for col in ofi_columns):
        raise ValueError("Toutes les colonnes nOFI_{level} nécessaires ne sont pas présentes dans le DataFrame.")

    # # Normaliser les données
    scaler = StandardScaler()
    ofi_normalized = scaler.fit_transform(ofi_data)

    # Appliquer le PCA
    pca = PCA(n_components=1)
    pca.fit(ofi_normalized)

    # Poids de la première composante principale
    first_component_weights = pca.components_[0]
    l1_norm = np.sum(np.abs(first_component_weights))  # Norme L1 des poids
    normalized_weights = first_component_weights / l1_norm  # Normalisation

    # Calculer l'Integrated OFI
    integrated_ofi = ofi_normalized @ normalized_weights

    # Afficher le pourcentage de variance expliquée
    explained_variance = pca.explained_variance_ratio_
    print(f"Variance expliquée par la première composante principale : {explained_variance[0]:.4f}")

    # Retourner l'Integrated OFI comme une série
    return pd.Series(integrated_ofi, index=ofi_df.index, name='Integrated_OFI')