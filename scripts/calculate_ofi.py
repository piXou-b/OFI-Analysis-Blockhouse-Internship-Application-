import pandas as pd
import numpy as np


def calculate_ofi(df, levels=5):
    """
    Calcule l'OFI normalisé pour chaque niveau et le multi-niveau intégré.

    Args:
        df (pd.DataFrame): Données contenant les colonnes bid_px_XX, bid_sz_XX, ask_px_XX, ask_sz_XX.
        levels (int): Nombre de niveaux du carnet d'ordres.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes OFI normalisées pour chaque niveau.
    """
    # Initialiser un dictionnaire pour les OFI normalisés
    df = df.copy()
    unsigned_columns = df.select_dtypes(include=['uint', 'uint8', 'uint16', 'uint32', 'uint64']).columns
    for col in unsigned_columns:
        df[col] = df[col].astype('int32')
    depths = []

    for level in range(levels):
        bid_price = f"bid_px_0{level}"
        ask_price = f"ask_px_0{level}"
        bid_size = f"bid_sz_0{level}"
        ask_size = f"ask_sz_0{level}"

        # Compute price and size changes for this level
        bid_price_change = df[bid_price].diff()
        ask_price_change = df[ask_price].diff()
        bid_size_change = df[bid_size].diff()
        ask_size_change = df[ask_size].diff()

        # Compute Bid OFI
        ofi_bid = np.where(
            bid_price_change > 0,
            df[bid_size],  # New size when price increases
            np.where(
                bid_price_change < 0,
                -df[bid_size],  # Negative size when price decreases
                bid_size_change  # Size change when price remains constant
            )
        )

        # Compute Ask OFI
        ofi_ask = np.where(
            ask_price_change > 0,
            -df[ask_size],  # Negative size when price increases
            np.where(
                ask_price_change < 0,
                df[ask_size],  # Positive size when price decreases
                ask_size_change  # Size change when price remains constant
            )
        )
        df[f"OFI_{level}"] = ofi_bid - ofi_ask

        # Calculer la profondeur moyenne pour ce niveau
        avg_depth = (df[bid_size] + df[ask_size])
        depths.append(avg_depth)

    avg_depth = sum(depths) / (2 * (levels + 1))

    for level in range(levels):
        df[f"nOFI_{level}"] = df[f"OFI_{level}"] / avg_depth

    return df