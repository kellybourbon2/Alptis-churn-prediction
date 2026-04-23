# BDC-Alptis
Project part of Ensae "Business Data Challenge" 2025-2026

# Data Preprocessing

Located in `src/data_processing/`, the preprocessing module handles data cleaning and preparation:

- **`data_load.py`**: Handles data loading from either local storage or S3 based on configuration
- **`data_processing.py`**: Orchestrates the complete preprocessing pipeline, coordinating cleaning and aggregation steps to produce a merged, client-level dataset
- **`cleaning.py`**: Cleans the portfolio data through feature engineering, missing value imputation, and categorical encoding
- **`aggregation.py`**: Aggregates secondary files (reclamations, consumptions, interactions, overdue payments) at the client level
- **`feature_engineering.py`**: Handles the creation of the new variables (courtier_anciennete_categories, client_age_categories, ...)
- **`data_external.py`**: Handles the integration of external variables (revenue of commune in 2021 from Insee, ...)
- **`text_processing.py`**: Handles the creation of features on the textual data (coming from nps reviews of year n or mails from interaction file)

The **`CODEBOOK.md`** presents a description of each variable created during features engineering, agregation, handling of textual data or external data integration.

# Data Preparation

Data preparation converts the processed features into training-ready datasets with different preprocessing options for different models. All scripts are located in `src/data_preparation/`.

## Preparation Scripts

### Available Variants

Four data preparation pipelines are available, each optimized for different models:

#### 1. **prepare_data_enc_norm.py** - Encoded + Normalized
- **Output suffix**: `enc_norm`
- **Use for**: Logistic Regression
- **Process**:
  - Encodes categorical features (target encoding, ordinal encoding)
  - Normalizes numerical features using StandardScaler
  - Prevents data leakage by fitting scaler only on training data


#### 2. **prepare_data_enc_nonorm.py** - Encoded, Not Normalized
- **Output suffix**: `enc_nonorm`
- **Use for**: HistGradientBoosting, Random Forest
- **Process**:
  - Encodes categorical features (target encoding, ordinal encoding)

#### 3. **prepare_data_noenc_nonorm_duplicatemissing.py** - Raw Features + Missing Indicators
- **Output suffix**: `noenc_nonorm`
- **Use for**: CatBoost (handles categorical features natively)
- **Process**:
  - No encoding or normalisation
  - Adds missing value indicator columns for columns with NaN values
  - Example: if column `client_age` has missing values that were imputed, creates `client_age_missing` with remaining missing values

#### 4. **prepare_data_enc_nonorm_duplicatemissing.py** - Encoded + Missing Indicators 
- **Output suffix**: `enc_nonorm_missing`
- **Use for**: Testing with missing value indicators on tree models
- **Process**:
  - Encodes categorical features
  - NO normalization
  - Adds missing value indicator columns


### How to Run Data Preparation

```bash
uv run python prepare_data_enc_nonorm.py             #to prepare data with encoding but without normalisation for example 
```
### Architecture: Avoiding Data Leakage

Each preparation script follows this strict pattern to prevent data leakage:

```python
# 1. Create FIT processor on TRAINING data only
train_processor = DataProcessor(mode="training")
df_train = train_processor.run(optional_encoding=True, optional_normalisation=True)

# 2. Create TRANSFORM processor for validation/test data
test_processor = DataProcessor(mode="validation")

# 3. Pass fitted transformers to test processor 
test_processor.scaler = train_processor.scaler          # Use training scaler
test_processor.target_encoding_maps = train_processor.target_encoding_maps  # Use training encoding

# 4. Apply same transformations to test data
df_test = test_processor.run_transform(optional_encoding=True, optional_normalisation=True)
```

**Key Principle**: All encoders, scalers, and transformation parameters are fit ONLY on training data, then applied to test data. This prevents information from test set leaking into training.

### Output Format

Each preparation script generates 4 files saved to S3 in Parquet format:

```
S3_DATA_PROCESSED_BUCKET/
├── X_train_{suffix}.parquet    # Training features (without target or key column)
├── y_train_{suffix}.parquet    # Training target variable
├── X_test_{suffix}.parquet     # Test features (without target or key column)
└── y_test_{suffix}.parquet     # Test target variable
```

**Example outputs**:
- `X_train_enc_norm.parquet`, `y_train_enc_norm.parquet`, etc. (for Logistic Regression)
- `X_train_noenc_nonorm.parquet`, `y_train_noenc_nonorm.parquet`, etc. (for CatBoost)

### Configuration

The destination S3 bucket can be modified in `Config.py`:

```python
S3_DATA_PROCESSED_BUCKET = "bdc-alptis-g2/processed_data"
```

# Model

## Final Model

|------|------|
| `model_final.py` | **Final prediction pipeline** — runs all models, aggregates outputs, applies a global threshold on decision (0.65) and business-rule clipping (NPS, mail churn mentions, contract commitment) to produce the final churn score |
| `model_stack.py` | **Stacking ensemble** — meta-learner combining Logistic Regression, XGBoost, and CatBoost; best-performing models are loaded from S3 |
| `model_baseline_majority_vote.py` | **Baseline model** — majority-vote classifier used as reference benchmark for the stacking ensemble |
| `model_bertopic.py` | **Topic modelling** — BERTopic pipeline applied to client email data to extract churn-related signals (e.g. price mentions, dissatisfaction) |
| `model_saving.py` | **Model I/O utility** — standardised helpers to load and save model artefacts to/from the S3 bucket |

The models on which the final model was stacked from  a logistic regression, a xgboost and a catboost model trained independantly. Next Section gives more details about their training.


## Models training

The model training pipeline supports multiple algorithms optimized for churn prediction. All training scripts are located in `src/models_training/` and are configured through `src/models_training/config/training_config.yaml`.

### Supported Models

- **Logistic Regression**: Interpretable baseline model with balanced class weights
- **Random Forest**: Ensemble method with feature importance analysis
- **XGBoost**: Gradient boosting with early stopping support
- **LightGBM**: Optimized gradient boosting for efficiency
- **HistGradientBoosting**: Native histogram-based gradient boosting
- **CatBoost**: Native support for categorical features, no encoding required

### Training Configuration

The `training_config.yaml` file defines:

### Global Training Parameters
- `random_state`: Seed for reproducibility (default: 42)
- `cv_folds`: Number of cross-validation folds (default: 5)
- `n_optuna_trials`: Number of hyperparameter optimization trials (default: 50)
- `primary_metric`: Primary evaluation metric for hyperparameter optimization(default: "pr_auc")
- `use_smote`: Whether to apply SMOTE for class imbalance (default: false)
- `early_stopping_round`: Early stopping patience for boosting models (default: 50)
- `shap_sample_size`: Number of samples for SHAP explanation (default: 200)

### Data Preparation specificiations for each model
Each model is configured with a `data_suffix` specifying its input format:
- `"enc_norm"`: Encoded and normalized features (for Logistic Regression)
- `"enc_nonorm"`: Encoded but not normalized features  (for XGBoost, LightGBM and HistGB )
- `"noenc_nonorm"`: Raw features without encoding (CatBoost for example)

### Running a Single Model Locally

To train and evaluate a specific model:

```bash
# Set up environment
uv sync
source .venv/bin/activate
# Run a specific model, for example xgboost
uv run python src/models_training/train_xgboost.py   
```

Models training are automatically dumped into `bdc-alptis-g2/Artifacts_model_training/`

# Experiment Reproducibility

To ensure full reproducibility of experiments across the Alptis team, we maintain a containerized infrastructure using Docker, MLflow, and Argo Workflows. This allows anyone to run the complete training pipeline on the processed dataset.

## Environment Setup & Credentials 

### S3 Storage Access

Create `.env` file in project root:
```bash
cp .env_template .env
```

Fill in your credentials from **Onyxia Genes** → **My Account** → **Storage Connection**:
```ini
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
```

⚠️ **Note**: `AWS_SESSION_TOKEN` expires periodically and must be refreshed.

### MLflow Tracking (Required for Model Training)

For tracking results on the shared MLflow server, add to `.env`:
```ini
MLFLOW_TRACKING_USERNAME=projet-bdc-data
MLFLOW_TRACKING_PASSWORD=...
MLFLOW_TRACKING_URI=https://projet-bdc-data-mlflow.lab.groupe-genes.fr/
```

**Get MLFLOW_TRACKING_PASSWORD from**: Onyxia Genes → **My Services** → **Project: projet-bdc-data** → Service **"Alptis-churn-mlflow-g2"** → copy password from service details.

## Directory Structure for Reproducibility

The following files and directories are essential for experiment reproducibility:

```
├── .env                         # Environment variables for S3 and MLFLOW access (git-ignored)
├── Config.py                    # Data loading & model configuration
├── secret.yaml                  # Kubernetes secrets for Argo (git-ignored)
├── docker/                      # Container configuration
│   └── Dockerfile               # Image with all dependencies + S3 access
├── src/data_preparation/        # Data transformation scripts for different model types
│   ├── prepare_data_enc_norm.py
│   │ 
│   ├── prepare_data_noenc_nonorm.py
│   └── prepare_data_enc_nonorm.py
├── src/models_training/config/           # Model configuration
│   └── training_config.yaml     # Hyperparameters for all 6 models
├── src/models_training/                  # Model training scripts
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── train_catboost.py
│   ├── train_random_forest.py
│   ├── train_logistic_regression.py
│   └── train_histgb.py
└── argo_workflows/              # Kubernetes workflow definitions
    └── train_pipeline.yaml      # Complete pipeline orchestration
    ```

    ### Data & Result Flow

```
S3 raw data (bdc-alptis-g2/raw_data/) 
  → src/data_preparation/ 
  → S3 processed data (bdc-alptis-g2/processed_data/) 
  → src/models_training/ (loads from S3)
  → S3 artifacts + MLflow (logs results)
```


## Docker - Containerized Environment

The `argo_workflows/train_pipeline.yaml` defines a complete, automated training pipeline that:

1. **Data Preparation** (parallel execution):
   - Prepares `enc_norm` data for Logistic Regression
   - Prepares `enc_nonorm` data for Random Forest and HistGradientBoosting
   - Prepares `enc_nonorm_duplicatemissing` for XGBoost and LightGBM
   - Prepares `noenc_nonorm_duplicatemissing` data for CatBoost 

2. **Model Training** (parallel execution with dependencies):
   - Trains 6 different models in parallel on their respective data formats
   - Each model runs hyperparameter optimization
   - Computes SHAP explanations
   - Logs results to MLflow

3. **Reproducible Execution**:
   - Runs in Kubernetes with fixed resource allocation
   - Uses Docker image for consistent environment
   - Passes secrets securely via Kubernetes secrets
   - All runs are version-controlled and logged



# How to Create an Argo Workflow Experiment on SSPCloud

This guide walks you through running the complete training pipeline on Kubernetes infrastructure, which trains all 6 models in parallel with full reproducibility.

## Step 1: Set Up SSPCloud Services

You'll need two services running on Onyxia Genes: Argo Workflows and VSCode.

### 2.1 Create Argo Workflow Service

1. Log in to **Onyxia Genes**
2. Navigate to **My Services** → **Create a new service**
3. Select **Argo Workflows** service
4. Deploy and note your **namespace** (shown at the top of the service)

### 2.2 Create VSCode Service

1. Navigate to **My Services** → **Create a new service**
2. Select **VSCode** service
3. ⚠️ **Important**: Set role to **"Admin"** (required for kubectl/Argo access)
4. Deploy and open the VSCode service

In the VSCode terminal:

```bash
# Clone the repository
git clone https://github.com/kellybourbon2/Alptis-churn-prediction.git

# Set up the environment
cd Alptis-churn-prediction
uv sync
```

## Step 3: Create Kubernetes Secrets File

Since Argo runs in Kubernetes, create `secret.yaml` using the credentials from `.env`:

```bash
cp secret_template.yaml secret.yaml
```

Then apply it to the cluster:
```bash
kubectl apply -f ./secret.yaml
```

See **"Environment Setup & Credentials"** section above for where to find these values.

## Step 4: Configure & Launch the Argo Workflow

Open `argo_workflows/train_pipeline.yaml` and update the metadata:

```yaml
metadata:
  name: my_experiment_v1              # Unique name for this run
  namespace: your-namespace           # Your Onyxia namespace (from Argo service)
```

Then submit the workflow:


```bash
# Apply the workflow to Kubernetes
kubectl apply -f argo_workflows/train_pipeline.yaml

# Verify the workflow was submitted
kubectl get workflows
# Should list your workflow with status "Running" or "Pending"
```

## Step 5: Monitor the Workflow Execution

### 5.1 Real-time Monitoring in Argo UI

1. Open your **Argo Workflows service** (created in Step 1.1)
2. You should see your workflow listed with the name you specified and the state of the workflow.


## Step 6: View Results in MLflow

Once the workflow completes, all training results are logged to MLflow.

### 6.1 Access MLflow

1. In **Onyxia Genes**, go to **My Services** → **Project: projet-bdc-alptis**
2. Open the shared service **"Alptis-churn-mlflow-g2"**
3. Or navigate directly to: `https://projet-bdc-data-mlflow.lab.groupe-genes.fr/`

