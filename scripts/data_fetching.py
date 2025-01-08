import databento as db


def fetch_data(symbol, api_key):
    client = db.Historical(api_key)

    df = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        schema="mbp-10",
        symbols=symbol,
        start="2024-11-04",
        end="2024-11-11",
    ).to_df()
    print(f"Fetched data for {symbol}: {df['ts_event'].min()} to {df['ts_event'].max()}")
    return df


def preprocess_data(df):
    required_colums = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00', 'bid_px_01', 'ask_px_01', 'bid_sz_01', 'ask_sz_01', 'bid_px_02', 'ask_px_02', 'bid_sz_02', 'ask_sz_02', 'bid_px_03', 'ask_px_03', 'bid_sz_03', 'ask_sz_03', 'bid_px_04', 'ask_px_04', 'bid_sz_04', 'ask_sz_04', 'bid_px_05', 'ask_px_05', 'bid_sz_05', 'ask_sz_05']
    # Traitement des NaN
    df[required_colums] = df[required_colums].ffill()
    df[required_colums] = df[required_colums].bfill()

    # Remplacer les 0 par la moyenne (ou supprimer les lignes)
    df = df[(df[required_colums] != 0).all(axis=1)]

    # Trier par timestamp
    df = df.sort_values(by='ts_event')

    return df