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
- **When to use**: Linear models that benefit from normalized features

#### 2. **prepare_data_enc_nonorm.py** - Encoded, Not Normalized
- **Output suffix**: `enc_nonorm`
- **Use for**: XGBoost, LightGBM, HistGradientBoosting, Random Forest
- **Process**:
  - Encodes categorical features (target encoding, ordinal encoding)
  - Keeps numerical features in original scale
  - Faster than normalized variant
- **When to use**: Tree-based models that are invariant to feature scaling

#### 3. **prepare_data_noenc_nonorm_duplicatemissing.py** - Raw Features + Missing Indicators
- **Output suffix**: `noenc_nonorm`
- **Use for**: CatBoost (handles categorical features natively)
- **Process**:
  - NO categorical encoding (keeps raw categorical values)
  - NO normalization of numerical features
  - Adds missing value indicator columns for columns with NaN values
  - Example: if column `age` has missing values, creates `age_missing` (binary flag)
- **When to use**: CatBoost which has native support for categorical features and benefits from missing value indicators

#### 4. **prepare_data_enc_nonorm_duplicatemissing.py** - Encoded + Missing Indicators (Experimental)
- **Output suffix**: `enc_nonorm_missing`
- **Use for**: Testing with missing value indicators on tree models
- **Process**:
  - Encodes categorical features
  - NO normalization
  - Adds missing value indicator columns
- **When to use**: Research/experimentation to compare impact of missing indicators

## How to Run Data Preparation

### Run All Preparation Variants

```bash
# Prepare all data variants in parallel
cd src/data_preparation
python prepare_data_enc_norm.py                    # Logistic Regression data
python prepare_data_enc_nonorm.py                  # Tree-based models data
python prepare_data_noenc_nonorm_duplicatemissing.py  # CatBoost data
```

### Run Single Variant

```bash
# Prepare only one variant
cd src/data_preparation
python prepare_data_enc_norm.py
```

### In Argo Workflow

The workflow automatically runs all three variants in parallel as the first step:

```yaml
# From argo_workflows/train_pipeline.yaml
tasks:
  - name: prepare-data-enc-norm
    template: prepare-data
    arguments:
      parameters:
        - name: script
          value: "src/data_preparation/prepare_data_enc_norm.py"
  # ... similar for other variants
```

## Data Preparation Pipeline Details

### Architecture: Avoiding Data Leakage

Each preparation script follows this strict pattern to prevent data leakage:

```python
# 1. Create FIT processor on TRAINING data only
train_processor = DataProcessor(mode="training")
df_train = train_processor.run(optional_encoding=True, optional_normalisation=True)

# 2. Create TRANSFORM processor for validation/test data
test_processor = DataProcessor(mode="validation")

# 3. Pass fitted transformers to test processor (CRITICAL!)
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

Data loading source is controlled by `LOAD_FROM_S3` in `Config.py`:
- `LOAD_FROM_S3 = True`: Loads raw data from S3 (default for Argo Workflow)
- `LOAD_FROM_S3 = False`: Loads raw data from local `data/` folder

## Model to Data Format Mapping

The training configuration automatically links each model to its data format:

| Model | Data Format | Script | Encoding | Normalization |
|-------|-------------|--------|----------|---------------|
| Logistic Regression | `enc_norm` | `prepare_data_enc_norm.py` | ✓ | ✓ |
| Random Forest | `enc_nonorm` | `prepare_data_enc_nonorm.py` | ✓ | ✗ |
| XGBoost | `enc_nonorm` | `prepare_data_enc_nonorm.py` | ✓ | ✗ |
| LightGBM | `enc_nonorm` | `prepare_data_enc_nonorm.py` | ✓ | ✗ |
| HistGradientBoosting | `enc_nonorm` | `prepare_data_enc_nonorm.py` | ✓ | ✗ |
| CatBoost | `noenc_nonorm` | `prepare_data_noenc_nonorm_duplicatemissing.py` | ✗ | ✗ |

See `src/models/config/training_config.yaml` for the mapping.

# Model Training

The model training pipeline supports multiple algorithms optimized for churn prediction. All training scripts are located in `src/models/` and are configured through `src/models/config/training_config.yaml`.

## Supported Models

- **Logistic Regression**: Interpretable baseline model with balanced class weights
- **Random Forest**: Ensemble method with feature importance analysis
- **XGBoost**: Gradient boosting with early stopping support
- **LightGBM**: Optimized gradient boosting for efficiency
- **HistGradientBoosting**: Native histogram-based gradient boosting
- **CatBoost**: Native support for categorical features, no encoding required

## Training Configuration

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
- `"enc_norm"`: Encoded and normalized features (Logistic Regression for example)
- `"enc_nonorm"`: Encoded but not normalized features  
- `"noenc_nonorm"`: Raw features without encoding (CatBoost for example)

## Running a Single Model Locally

To train and evaluate a specific model:

```bash
# Set up environment
uv sync
source .venv/bin/activate
# Run a specific model, for example xgboost
uv python src/models/train_xgboost.py   
```

Each training script will:
1. Load the preprocessed data from S3 (or local storage if configured)
2. Run Optuna hyperparameter optimization with cross-validation
3. Train the final model with best parameters
4. Compute SHAP feature importance
5. Evaluate on test set
6. Log all results to MLflow

## Output and Results

Training results are logged to MLflow including:
- Best hyperparameters
- Cross-validation and test metrics (ROC-AUC, PR-AUC, F1, Precision, Recall)
- Training time
- Top 20 SHAP feature importance values
- Trained model artifacts
- Classification report and confusion matrix

Results are automatically saved to S3 bucket `bdc-alptis-g2/Artifacts_model_training/`


# Experiment Reproducibility

To ensure full reproducibility of experiments across the Alptis team, we maintain a containerized infrastructure using Docker, MLflow, and Argo Workflows. This allows anyone to run the complete training pipeline on the processed dataset.

## Directory Structure for Reproducibility

```
├── docker/              # Container configuration
│   └── Dockerfile       # Image with all dependencies 
├── argo_workflows/      # Kubernetes workflow 
│   └── train_pipeline.yaml    # Complete training pipeline orchestration
├── src/models/config/   # Model configuration
│   └── training_config.yaml   # Hyperparameters and 
└── secret.yaml          # Secrets management 
```

## Docker - Containerized Environment

The `docker/Dockerfile` encapsulates the complete data processing and training environment:
- Python environment with all dependencies
- MLflow integration
- S3 connectivity

This ensures that experiments run consistently regardless of the local machine setup.

**Benefit**: Anyone with the Dockerfile can recreate the exact same environment that produced the results.

## MLflow - Experiment Tracking

MLflow tracks all training experiments with:
- Hyperparameters used for each model
- Performance metrics (ROC-AUC, PR-AUC, F1, etc.)
- Cross-validation scores
- Training time and computational resources
- SHAP feature importance
- Trained model artifacts

All experiments are logged to the shared MLflow server at: `https://projet-bdc-data-mlflow.lab.groupe-genes.fr/`


## Argo Workflows - Pipeline Orchestration

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

**Benefit**: The entire experiment pipeline is defined as code. Alptis team can re-run the exact same experiments on the processed dataset without manual intervention, ensuring consistency and reproducibility across runs.

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
3. ⚠️ **CRITICAL**: Set role to **"Admin"** (required for kubectl/Argo access)
4. Deploy and open the VSCode service

In the VSCode terminal:

```bash
# Clone the repository
git clone https://github.com/kellybourbon2/Alptis-churn-prediction.git

# Set up the environment
cd Alptis-churn-prediction
uv sync
```

## Step 3: Configure Kubernetes Secrets

The pipeline needs access to AWS S3 and MLflow. Store these credentials securely using Kubernetes secrets.

### 4.1 Create secret.yaml

Create a file `secret.yaml` in the project root:

```yaml
cp secret_template.yaml secret.yaml
```

### 4.2 Find Your Credentials

**AWS Credentials:**
- Log into **Onyxia Genes**
- Go to **My Account** → **Storage Connection**
- Copy the three values: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`

**MLflow Password:**
- In Onyxia Genes, go to **My Services** → **Project: projet-bdc-data**
- Find the shared service **"Alptis-churn-mlflow-g2"** (shared with your team)
- Open it and copy the password from the service details

### 4.3 Apply Secrets to Kubernetes

```bash
# Register the secrets with Kubernetes (you're already in your namespace)
kubectl apply -f ./secret.yaml

# Verify the secret was created
kubectl get secrets
```

## Step 5: Configure the Argo Workflow

### 5.1 Update Workflow Metadata

Open `argo_workflows/train_pipeline.yaml` and update the metadata section:

```yaml
metadata:
  name: my_experiment_v1              # Unique name for this run
  namespace: your-namespace           # Your Onyxia namespace (from Argo service)
```

**Important**: Each experiment must have a unique name. If you reuse a name, it will fail or overwrite the previous run.


## Step 5: Launch the Workflow

Submit the workflow to Kubernetes:

```bash
# Apply the workflow to Kubernetes
kubectl apply -f argo_workflows/train_pipeline.yaml

# Verify the workflow was submitted
kubectl get workflows
# Should list your workflow with status "Running" or "Pending"

# Get detailed status
kubectl describe workflow my_experiment_v1
```

## Step 6: Monitor the Workflow Execution

### 6.1 Real-time Monitoring in Argo UI

1. Open your **Argo Workflows service** (created in Step 2.1)
2. You should see your workflow listed with the name you specified 


### 6.2 Command Line Monitoring

```bash
# Watch workflow status continuously
kubectl describe workflow my_experiment_v1 -w

# View logs from a specific task (e.g., train-xgboost)
kubectl logs -l workflow=my_experiment_v1,task=train-xgboost -f

# Get all pod events
kubectl get events --sort-by='.lastTimestamp'
```

## Step 7: View Results in MLflow

Once the workflow completes, all training results are logged to MLflow.

### 7.1 Access MLflow

1. In **Onyxia Genes**, go to **My Services** → **Project: projet-bdc-alptis**
2. Open the shared service **"Alptis-churn-mlflow-g2"**
3. Or navigate directly to: `https://projet-bdc-data-mlflow.lab.groupe-genes.fr/`

