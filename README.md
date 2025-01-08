# Order Flow Imbalance (OFI) Analysis Project

## Description

This project analyzes the self- and cross-impact of normalized and integrated Order Flow Imbalances (OFIs) on asset returns. It uses regression models, clustering techniques, and predictive models to uncover relationships among stocks and their OFI dynamics. The main objectives include:

- Evaluating the contemporaneous price impact of OFIs.
- Investigating cross-impact relationships across multiple stocks.
- Exploring predictive power for future returns.
- Leveraging clustering for stock behavior insights.

The analysis is conducted on Nasdaq ITCH data for the stocks AAPL, AMGN, JPM, TSLA, and XOM, spanning from 2024-11-04 to 2024-11-08.

## Project Structure

The project includes the following key components:

1. **OFI Calculation**: Computation of best-level, deeper-level, and integrated OFIs.
2. **Regression Analysis**: Models such as PI^1, CI^1, PI^I, and CI^I examine self- and cross-impact dynamics.
3. **Predictive Modeling**: Forward-looking models FPI^1, FPI^I, FCI^1, and FCI^I test the forecasting power of OFIs.
4. **Clustering**: PCA and hierarchical clustering analyze relationships among stocks.

## Steps to Run the Analysis

### 1. Install Dependencies

Ensure Python (>= 3.8) is installed on your system. Install the required packages by running:

```bash
pip install -r requirements.txt
```
### 2. Set Up Data Bento API

The project uses Data Bento to fetch Nasdaq ITCH data. Follow these steps to configure the API:
1. Create a Data Bento Account:
Visit Data Bento and sign up.
2. Generate an API Key:
Log in and navigate to the “API Keys” section. Create a new API key.
3. Add Your API Key to fetch_data():
Replace the placeholder with your API key:

### 3. Data Preparation
The script fetches, cleans, and preprocesses Nasdaq ITCH data using your API key. It processes the dataset to calculate multi-level OFIs and returns.

### 4. Run the Analysis

Execute the analysis notebook

### 5. View Results

The analysis generates:
	•	Summary statistics of OFIs.
	•	Correlation heatmaps and PCA variance explained.
	•	IS and OS R² values for regression and predictive models.
	•	Clustering visualizations.

Figures and tables are provided in the respective sections of the code.

Customization
Adjust the stocks, time intervals, or hyperparameters in the code to tailor the analysis to your specific requirements.

Contact
For questions or support, please contact Benjamin EMILY at benji.emily@icloud.com.
