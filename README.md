# U.S. chronic disease indicators — EDA & prediction dashboard

Interactive [Dash](https://dash.plotly.com/) app for exploratory analysis and a small mortality prediction model, using CDC [U.S. Chronic Disease Indicators](https://www.cdc.gov/cdi/index.html)–style tabular data.

## Repository layout

| Path | Role |
|------|------|
| `app.py` | Dash application entrypoint (`server` for Gunicorn / App Engine) |
| `cleaning.py` | Load raw CSV, impute numerics, drop sparse rows, **z-score outlier screen** (all \|z\| < 3) |
| `visualizations.py` | Plotly figures (choropleths, linked dropdowns, scatter, etc.) |
| `model.py` | **sklearn** `ColumnTransformer` + **RandomForestRegressor**; training data in `data/` |
| `data/chronic_disease_indicators.csv` | Raw indicator extract (~85M; keep path stable for `cleaning.py`) |
| `data/final_training_data.csv` | Filtered training slice for the heart-disease mortality model |
| `notebooks/` | `data_pipeline.ipynb` (cleaning / export), `models.ipynb` (linear RF, NN experiments with PyTorch) |

## Techniques (methods summary)

### Data preparation (`cleaning.py`)

- **Median imputation** for `DataValue`, `DataValueAlt`, and confidence-limit columns.
- **Row filtering**: drop rows missing `Geolocation`; drop footnote columns when unused.
- **Outlier handling**: **z-scores** on the numeric value columns; retain rows where all |z| < 3 (standard univariate screen).

### Visualization (`visualizations.py` + `app.py`)

- **Plotly Express** for bar, line, density heatmap, choropleth, pie, and scatter plots.
- **Plotly Graph Objects** with **`updatemenus`** for year / state toggles (multi-trace figures without full app callbacks for every view).
- **Geo**: US choropleths with `locationmode='USA-states'`.
- **Merged analytics**: inner join of diabetes vs. obesity prevalence by `LocationAbbr` for a **bivariate scatter** (correlation-style EDA).

### Machine learning (`model.py`)

- **`ColumnTransformer`** combining:
  - **One-hot encoding** (`OneHotEncoder`) for `LocationAbbr` and `Stratification1` (`handle_unknown='ignore'` for robustness at prediction time).
  - **StandardScaler** on `YearStart`.
- **`RandomForestRegressor`** (100 trees, `random_state=42`) after an **80/20 `train_test_split`** on the preprocessed design matrix.
- The Dash UI calls **`preprocessor.transform`** on user-selected year / state / stratum, then **`forest_model.predict`**.

### Deployment

- **`app.yaml`** (Google App Engine): `gunicorn -b 0.0.0.0:8080 app:server`.

## Local run

Requires **Python 3.9+** (matches `app.yaml` runtime).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8050 (Dash dev server via `app.run`).

**Note:** `notebooks/models.ipynb` uses **PyTorch** for neural-network experiments; PyTorch is not listed in `requirements.txt` (install separately if you rerun that notebook).

## Data provenance

Place the CDC chronic disease indicators CSV at `data/chronic_disease_indicators.csv` (or adjust `RAW_CSV` in `cleaning.py`). Regenerate `data/final_training_data.csv` from `notebooks/data_pipeline.ipynb` if you change filters or outcomes.
