# BDC-Alptis
Project part of Ensae "Business Data Challenge" 2025-2026

# How to start:
Clone the repo and follow the following steps:

## Environement creation and activation
```bash
uv sync
source .venv/bin/activate
```

 **Attention !!:  Travailler toujours sur sa branche personnelle:**
```bash
git switch <branche_personnelle>
```
Avant chaque commit/push, checker qu'on est pas sur le main:
```bash
git branch
```
Si jamais on veut pousser sur main: cf Kelly


## Data Loading

Two loading modes are available depending on the variable `LOAD_FROM_S3` in `Config`

### Local Loading

1. Create a folder matching the `DATA_RAW_DIR` variable defined in `Config`
2. Organize your data files following this structure:
```
data/
├── training/
│   └── *.csv
├── validation/
│   └── *.csv
└── test/
    └── *.csv
```

### SSPCloud Loading (S3 Loading)

Data is loaded directly from the shared MinIO bucket on SSPCloud.

Create a `.env` file at the root of the project with the following variables:
```ini
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_S3_ENDPOINT='minio-simple.lab.groupe-genes.fr'
AWS_DEFAULT_REGION= 'us-east-1'
AWS_BUCKET_NAME='projet-bdc-data'
```
>  These credentials can be found in your SSPCloud account under **My Account → Storage Connection**. Note that `AWS_SESSION_TOKEN` expires periodically and must be refreshed.


## Data Preprocessing

Located in `src/data_processing/`, the preprocessing module handles data cleaning and preparation:

- **`data_load.py`**: Handles data loading from either local storage or S3 based on configuration
- **`data_processing.py`**: Orchestrates the complete preprocessing pipeline, coordinating cleaning and aggregation steps to produce a merged, client-level dataset
- **`cleaning.py`**: Cleans the portfolio data through feature engineering, missing value imputation, and categorical encoding
- **`aggregation.py`**: Aggregates secondary files (reclamations, consumptions, interactions, overdue payments) at the client level
- **`feature_engineering.py`**: Handles the creation of the new variables (courtier_anciennete_categories, client_age_categories, ...)
- **`data_external.py`**: Handles the integration of external variables (revenue of commune in 2021 from Insee, ...)

#### Demo (only for dev - to delete later)
Look at the demo_loading notebook to know how to load a file from SPPCloud (after creating the .env file) 

# TO DO 

>Performance du modèle:
créer un script a faire tourner sur mlflow pour chaque type de modèle:
- reg_logistique
-boosting
- random_forest

>Analyse quantitative:
analyser les mails (BERT TOpics) et les topics qui reviennet le +
analyser les avis nps (<3 ou avis nps des clients ayant résiliés) avec BERT Topic



--> **ATTENTION: QUAND CREE FONCTION FIT, VIRER CLIENT_CODE (Variable KEY_COLUMN dans Config)**
--> SINON RISQUE DOVERFIT SUR CA (On peut pas la drop au moment du data processing sinon perd info quand fait prédiction)

Pour cela: 
from Config import KEY_COLUMN, TARGET_COLUMN
X = df.drop(columns=[TARGET_COLUMN, KEY_COLUMN])
y = df[target_col]
model.fit(X, y)
predictions = model.predict(X)

**Quand on a les résultats de prediction(a la fin du training):**
 Recoller la colonne client_code via index — garanti aligné car même df
results = pd.DataFrame({
    "client_code":  df_ready["client_code"],  # depuis le même df
    "churn_predit": predictions
})
