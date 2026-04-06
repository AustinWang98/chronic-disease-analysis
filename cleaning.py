import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

_DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = _DATA_DIR / "chronic_disease_indicators.csv"


def clean_my_data():
    data = pd.read_csv(RAW_CSV)
    data.drop(columns=[
        'Response', 'StratificationCategory2', 'Stratification2',
        'StratificationCategory3', 'Stratification3', 'ResponseID',
        'StratificationCategoryID2', 'StratificationID2',
        'StratificationCategoryID3', 'StratificationID3'], inplace=True)

    numerical_cols = ['DataValue', 'DataValueAlt', 'LowConfidenceLimit', 'HighConfidenceLimit']
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

    data.dropna(subset=['Geolocation'], inplace=True)

    data.drop(columns=['DataValueFootnoteSymbol', 'DataValueFootnote'], inplace=True)

    z_scores = stats.zscore(data[numerical_cols])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]

    return data
