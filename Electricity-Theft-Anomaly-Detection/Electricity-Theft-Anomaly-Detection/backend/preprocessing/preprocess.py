import pandas as pd
import os

def full_preprocessing(filepath):

    data = pd.read_csv(
        filepath,
        sep=';',
        decimal=',',
        index_col=0
    )

    data.index = pd.to_datetime(data.index)

    data_filled = data.ffill().bfill()

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    cleaned_path = os.path.join(upload_dir, "cleaned_consumption.csv")
    data_filled.to_csv(cleaned_path)

    features = []

    for meter in data_filled.columns:

        meter_data = data_filled[meter]

        avg_consumption = meter_data.mean()
        max_consumption = meter_data.max()
        std_dev = meter_data.std()

        load_factor = avg_consumption / max_consumption if max_consumption != 0 else 0

        features.append({
            "consumer_id": meter,
            "avg_consumption": avg_consumption,
            "max_consumption": max_consumption,
            "variability": std_dev,
            "load_factor": load_factor
        })

    features_df = pd.DataFrame(features)

    features_df.to_csv(
        os.path.join(upload_dir, "explainable_anomalies.csv"),
        index=False
    )

    return features_df