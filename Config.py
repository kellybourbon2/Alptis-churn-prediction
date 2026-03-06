
"""Configuration file for data paths and dataset specifications"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# To load locally or from S3
LOAD_FROM_S3 = False

#Local data path
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data_raw")

# S3 config
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT", "")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME", "projet-bdc-data")
S3_VERIFY_SSL = False

# Dataset file configurations

TRAINING_FILES = {
    "consommations": "apprentissage_consommations_2023_11.csv",
    "impayes": "apprentissage_impayes_2023_11.csv",
    "interactions": "apprentissage_interactions_2023_11.csv",
    "portefeuille": "apprentissage_portefeuille_2023_11.csv",
    "reclamations": "apprentissage_reclamations_2023_11.csv",
}

# Evaluation/Validation set files (2024-11)
EVALUATION_FILES = {
    "consommations": "evaluation_consommations_2024_11.csv",
    "impayes": "evaluation_impayes_2024_11.csv",
    "interactions": "evaluation_interactions_2024_11.csv",
    "portefeuille": "evaluation_portefeuille_2024_11.csv",
    "reclamations": "evaluation_reclamations_2024_11.csv",
}

# Map sets to their file configurations
DATASET_MAPPING = {
    "training": TRAINING_FILES,
    "evaluation": EVALUATION_FILES,
    "validation": EVALUATION_FILES,  
}