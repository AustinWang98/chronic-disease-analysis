import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

_DATA_DIR = Path(__file__).resolve().parent / "data"
train_df = pd.read_csv(_DATA_DIR / "final_training_data.csv")

categorical_preprocessing = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])
numeric_preprocessing = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_preprocessing, ['LocationAbbr', 'Stratification1']),
        ('num', numeric_preprocessing, ['YearStart'])
    ])

features = train_df[["YearStart", "LocationAbbr", "Stratification1"]]
features_transformed = preprocessor.fit_transform(features)
target = train_df['DataValue']

X_train, X_test, y_train, y_test = train_test_split(
    features_transformed, target, test_size=0.2, random_state=42)

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
