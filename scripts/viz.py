import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_heatmap_coefficients(results, target_stocks):
    """
    Plot a heatmap of regression coefficients in a 5x5 layout for all target stocks.
    """
    # Initialize a DataFrame to store coefficients
    coefficients = pd.DataFrame(index=target_stocks, columns=target_stocks)

    # Fill the DataFrame with regression coefficients
    for target_stock in results.keys():
        for i, predictor in enumerate(results[target_stock]['features']):
            # Extract the stock name from the feature (e.g., "AAPL_OFI_Lagged" -> "AAPL")
            predictor_stock = predictor.replace('_OFI_Lagged', '')
            coefficients.loc[predictor_stock, target_stock] = results[target_stock]['coefficients'][i]

    # Fill NaN values with 0 and handle future warnings
    coefficients = coefficients.fillna(0).infer_objects(copy=False)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(coefficients.astype(float), cmap='coolwarm', center=0, fmt=".1e")
    plt.xlabel('Target Stocks', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


def plot_scatter_ofi_vs_price_change(full_data, target_stocks):
    """
    Create scatter plots of OFI (or lagged OFI) vs price change for each stock.

    Args:
        full_data (pd.DataFrame): Data containing 'symbol', 'minute', 'Integrated_OFI', and 'price_change'.
        target_stocks (list): List of stocks to create scatter plots for.
        lagged (bool): Whether to use lagged OFI data or not.
        lag_steps (int): Number of steps to lag the data (if lagged is True).
    """
    plt.figure(figsize=(16, 10))
    num_stocks = len(target_stocks)
    rows = (num_stocks + 1) // 2  # Calculate the number of rows for subplots

    for i, stock in enumerate(target_stocks):
        plt.subplot(rows, 2, i + 1)

        # Filter data for the current stock
        stock_data = full_data[full_data['symbol'] == stock]

        x = stock_data['Integrated_OFI']
        y = stock_data['price_change']
        xlabel = "Integrated OFI"

        # Check for valid data
        if x.isnull().all() or y.isnull().all():
            print(f"No valid data for {stock}. Skipping plot.")
            continue

        # Scatter plot
        sns.scatterplot(
            x=x,
            y=y,
            alpha=0.6,
            edgecolor=None
        )

        plt.xlabel(xlabel)
        plt.ylabel("Price Change")
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.show()


def plot_self_vs_cross_comparison(results, lagged=False):
    """
    Plot comparison of self-impact vs. cross-impact R² values.

    Args:
        results (dict): Results from compare_self_vs_cross_impact.
    """
    # Prepare data for plotting
    stocks = results.keys()
    r2_self = [results[stock]['self_r2'] for stock in stocks]
    r2_cross = [results[stock]['cross_r2'] for stock in stocks]

    # Bar plot of R² values
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    x = range(len(stocks))
    plt.bar(x, r2_self, width=bar_width, label='Self-Impact R²', color='skyblue')
    plt.bar([p + bar_width for p in x], r2_cross, width=bar_width, label='Cross-Impact R²', color='salmon')
    plt.xticks([p + bar_width / 2 for p in x], stocks, rotation=45)
    plt.xlabel('Stocks', fontsize=12)
    plt.ylabel('R² Value', fontsize=12)
    plt.legend()
    plt.show()


def plot_ofi_levels(ofi_data):
    """
    Plot OFI metrics for individual stocks across different levels as a bar plot.

    Args:
        ofi_data (pd.DataFrame): DataFrame containing OFI metrics with stocks as rows
                                 and levels as columns.
    """
    # Set the figure size
    plt.figure(figsize=(12, 6))

    # Define the number of levels and stock names
    levels = ofi_data.columns
    stocks = ofi_data.index
    bar_width = 0.15  # Width of each bar
    x = np.arange(len(levels))  # X positions for the groups of bars

    # Plot bars for each stock
    for i, stock in enumerate(stocks):
        plt.bar(x + i * bar_width, ofi_data.loc[stock], width=bar_width, label=stock)

    # Add titles and labels
    plt.xlabel('OFI Levels', fontsize=12)
    plt.ylabel('OFI Value', fontsize=12)
    plt.xticks(x + (len(stocks) - 1) * bar_width / 2, levels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Stocks', fontsize=10)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_correlation_matrices_by_stock(data):
    """
    Compute and plot correlation matrices for each stock's OFI levels.

    Args:
        data (pd.DataFrame): Dataset with OFI levels and a `symbol` column.
    """
    # Group the data by stock
    grouped = data.groupby('symbol')

    # Plot correlation heatmap for each stock
    for stock, group in grouped:
        # Drop the symbol column to focus on OFI levels
        ofi_levels = group[[col for col in group.columns if 'nOFI_' in col]]

        # Compute correlation matrix
        corr_matrix = ofi_levels.corr()

        # Plot heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.xlabel('OFI Levels')
        plt.ylabel('OFI Levels')
        plt.show()