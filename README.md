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

## Data preparation
## Data Preparation

- **`prepare_data_enc_norm.py`**, **`prepare_data_noenc_norm.py`**, and **`prepare_data_enc_nonorm.py`**: pipeline scripts that handle:
  - data preprocessing according to the selected parameters (e.g. encoding with normalization, encoding without normalization, etc.)
  - preparation of the train/test datasets
  - saving the preprocessed datasets to S3 storage (same environment as used in the Data Loading step)

**Note**: The destination S3 bucket can be modified through the variable `S3_DATA_PROCESSED_BUCKET` in `Config`.
## Training

## Model training
We've tested different models of training 


## To do a Training and track results on MLFLOW
Once you defined in your branch all the changes (on the data processing and on the model config):
0. Change the variable MLFLOW_EXPERIMENT_NAME with the name of your experiment
1. Open a MLFLOW service and copy-paste the password of the service somewhere
2. Add the following in your `.env `file: 
```ini
MLFLOW_TRACKING_USERNAME=projet-bdc-data
MLFLOW_TRACKING_URI=https://projet-bdc-data-mlflow.lab.groupe-genes.fr/
MLFLOW_TRACKING_PASSWORD=... 
MLFLOW_S3_ENDPOINT_URL=https://minio-simple.lab.groupe-genes.fr
```
>Paste the password saved in step 1
3. Run the model you want in terminal (*ex: uv run python train_xgboost.py*)
4. Open the link of URI to see the results
5. Can see the saved models in bucket bdc-alptis-g2/Artifacts_model_training

## To create an Argo-Workflow on Onyxia

⚠️ **Very important**: create a VSCode with "Admin" role selected.

Then, pass secret credentials to kubernetes:  

1. Create in a file, named `secret.yaml` with all your secrets variables, as followed: 
```
apiVersion: v1
kind: Secret
metadata:
  name: env-secrets 
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: ...
  AWS_SECRET_ACCESS_KEY: ... 
  AWS_SESSION_TOKEN: ....
  MLFLOW_TRACKING_PASSWORD: ....
```
2. Pass the secrets to the kubernetes cluster: 
```bash
kubectl apply -f ./secret.yaml
```
3. Run the workflow in argoworkflow
```bash
kubectl apply -f argo_workflows/train_pipeline.yaml
```
4. Open the server argoworkflow to visualise the workflow

#### Demo (only for dev - to delete later)
Look at the demo_loading notebook to know how to load a file from SPPCloud (after creating the .env file) 

# TO DO 

- Créer les variables quantitatives sur analyses textuelles

Attention; mlflow experience dans projet-bdc-alptis

FINIR ARGOWORKFLOW pour entrainement modeles




- écrire des script avec missing_values laissées dans data preparation: en faisant attention a option fill missing values --> tester sur modeles robuste a missing values et voir si diff en terme de score

- Check si yaml ok selon les specificités de chaque modele

EXPORT... les var mlflow pour tester rapidos script training

- modif src/model/config: 
si le temps, ajouter les variables max/min de chaque search_space dans un partie optuna propre a chaque modele
- tester les script de chaque modele: xgboost ok, tester log_reg, catboost, lightgbm, histgb et random forest 

- creer github workflow action pour updater les X_train/Y_train/... du bucket S3 automatiquement à chaque modif du dossier data_processing (ie workflow qui run chacun des prepare_data...)

>Puis créer argoworkflow qui regroupe modeles avec meme data processing 
ISSUE: ARGOWORKFLOW fonctionne que sur SSPCLOUD pas GENES

>Run plein d'argoworfklow en changeant le dataprocessing (ajout de variables, suppressions d'autres,...) + avec et sans clipping + avec sans smote, ... et tjrs en précisant dans le nom de l'expérience (mlflow/argoworkflow) changement faits sur dataprocessing

- Ajouter sélection meilleur model + l'étape de clipping finale: 0 si annulation résiliation/ 0 si date_debut_effet_garanti_mois<11.5 à la pipeline argoworkflow 


## Data processing
>Ajouter a data procesisng:
Impaye
Variables crées sur les mail

>Analyse quantitative:
identifier les mails des clients ayant annulés la résiliation
identifier thèmes qui reviennent le plus dans motif Résiliation

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

