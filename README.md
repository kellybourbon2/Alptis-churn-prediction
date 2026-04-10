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
- **`prepare_data.py`**: Pipeline script that handles the data processsing, the train/test datasets preparation and the saving of the pre-processed datasets in S3 Storage (same as the one from Data Loading). The bucket for the storing in S3 can be changes with the variable `S3_DATA_PROCESSED_BUCKET` in `Config`.

## Training

We've tested different models of training 
...

## To create an Argo-workflow locally

1. Create the namespace kubernetes, for example called "argo"
```ini
kubectl create namespace argo
```

2. Install argoworkflow on the created namespace: 
```ini
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.7.12/install.yaml
```

3. Share the variables from `.env` file to the namespace kubectl created, as a secret: 
```ini
kubectl create secret generic env-secrets --from-env-file=.env -n argo
```
4. Create the argo-workflow (from the file `argo_workflows/train_pipeline`) on the namespace:
```ini
kubectl create -f argo_workflows/train_pipeline.yaml -n argo
```
5. Log to argo-workflow ui in another terminal bash (disable authentification then display on port 2467):
First desactivate authentification:
```ini
kubectl patch deployment argo-server -n argo \
  --type='json' \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/args","value":["server","--auth-mode=server"]}]'
```
Then print on port 2746:
```ini
while true; do kubectl port-forward svc/argo-server -n argo 2746:2746 2>/dev/null; sleep 1; done
```
open:
https://localhost to see Argoworkflow UI


...or see how to setup kubernetes to S3 storage on Onyxia when available (Account > Onyxia)
Install argoworkflow on namespace:

kubectl apply -n test_namespace -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
>  Make sure to change your-namespace with the name you gave to the kubectl space you've created.
> Find credentials for access-key and secret-key in your SSPCloud account under **My Account → Storage**

## To create an Argo-Workflow on Onyxia
--> Write when Onyxia's back (issue with Onyxia: impossible to put secrets in kubectl, has to go through Vault i think but not sure)

First, Pass secret credentials to Vault
... ? 

Then, open an ArgoWorkflow server: 
1. Create a template Argoworflow
2. Paste the workflow in the template : argo_workflows\train_pipeline.yaml
>  make sure to change to namespace variable with your own namespace. *Ex: user-kbourbon* 
3. Create workflow

#### Demo (only for dev - to delete later)
Look at the demo_loading notebook to know how to load a file from SPPCloud (after creating the .env file) 

# TO DO 

## Model training
>Faire enfin marcher argoworkflow 
>Créer un script pour catboost, random forest : faire un autre DAG, avec un data_preparation_2 sans option encoding 
>Eventuellement script pour rég linéaire: nécessite encore un data_preparation_3 avec encoding + tri sur multicolinéarité
>Ajouter sélection meilleur model + l'étape de clipping finale: 0 si annulation résiliation/ 0 si date_debut_effet_garanti_mois<11.5 à argoworkflow 


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

