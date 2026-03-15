"""Configuration file for data paths and dataset specifications"""

import os
from dotenv import load_dotenv
import pandas as pd

#---------------DATA PROCESSING--------------

TARGET_COLUMN='target_resiliation_6mois'
KEY_COLUMN = "client_code"

REFERENCE_DATES = {
    "training":  pd.Timestamp("2023-11-30"),
    "validation": pd.Timestamp("2024-11-30"),  
    "evaluation":  pd.Timestamp("2024-11-30")
}

HIGH_CARDINALITY=10
EXCEPT_HIGH_CARDINALITY = ["courtier_segmentation_interne"]

FILES_TO_DROP= "impaye"
COLUMNS_TO_DROP = ["client_structure_familiale", 'annee_mois_paiement'] 
#annee mois paiement: liste des mois de paiement, trouver moyen de le process pour modèle

COLUMNS_TO_PROCESSED_WITH_NLP= ['interaction_historique_mail', "client_nps_verbatim_n", "client_nps_verbatim_n_moins1"]
COLUMNS_ORDINAL = ["age_categories", "courtier_anciennete_categories","client_nb_assures", "client"]

#categorical encoding
ANCIENNETE_COURTIER_COLUMN = 'courtier_anciennete_annees'
ANCIENNETE_BINS = [0, 2, 7, 12, 20, 40] #cf graph of Overview alptis
ANCIENNETE_LABELS= ["new", "recent", "stable", "old", "very old"]

AGE_COLUMN= "client_age_souscripteur"
AGE_BINS = [18, 25, 35, 50, 60, 80, 120]  #based on younger and older person in portefeuille
AGE_LABELS = ['young', 'adult', 'mature','middle aged', 'senior', 'senior plus'] 

RECLA_DELAIS_COURT = 3 #(<3: court)
RECLA_DELAIS_LONG= 15 #(>15: long, 3-15: moyen)

#----------ENVIRONNEMENT SETTING--------------------------

# Load environment variables
load_dotenv(override=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# To load locally or from S3
LOAD_FROM_S3 = True

#If LOAD_FROM_S3=False, put dataset in following dir: 
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data")

#Merged dataset directory in SPPCloud in format parquet
MERGED_PARQUET_S3= "projet-bdc-alptis-g2/data_merged.parquet"

# S3 config
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT", "")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME", "projet-bdc-data")
S3_VERIFY_SSL = False

#To load directly processed dataset (clean, aggregated and merged)
DATA_PROCESSED_DIR= os.path.join(PROJECT_ROOT, "processed_dataset.csv")

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