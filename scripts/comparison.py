import pandas as pd
from .cross_impact_analysis import analyze_impact_with_integrated_ofi


def compare_self_vs_cross_impact(full_data, target_stocks, lagged=False, lag_steps=1):
    """
    Compare self-impact vs. cross-impact in regression models.

    Args:
        full_data (pd.DataFrame): Data containing 'Integrated_OFI', 'price_change', and lagged columns if applicable.
        target_stocks (list): List of target stocks to analyze.
        lagged (bool): Whether to use lagged OFI or not.
        lag_steps (int): Number of lag steps if lagged is True.

    Returns:
        dict: Comparison results with self-impact and cross-impact R² values.
    """
    # Cross-impact analysis using the existing function
    cross_impact_results = analyze_impact_with_integrated_ofi(full_data, lagged=lagged, lag_steps=lag_steps)
    # Self-impact analysis using the existing function
    self_impact_results = analyze_impact_with_integrated_ofi(full_data, lagged=lagged, lag_steps=lag_steps, cross_impact=False)

    results = {}

    for target_stock in target_stocks:
        # Extract cross-impact R² value
        if lagged:
            isr2_cross = cross_impact_results[target_stock]['OOS R^2']
            isr2_self = self_impact_results[target_stock]['OOS R^2']
        else:
            isr2_cross = cross_impact_results[target_stock]['IS R^2']
            isr2_self = self_impact_results[target_stock]['IS R^2']

        # Store results
        results[target_stock] = {
            'self_r2': isr2_self,
            'cross_r2': isr2_cross,
        }

    return results