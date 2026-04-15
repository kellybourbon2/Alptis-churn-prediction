"""Configuration file for data paths and dataset specifications"""

import os
from dotenv import load_dotenv
import pandas as pd

# Load variables from secret file
load_dotenv(override=True)

#------MODEL EXPERIMENT --------------
#------------------------------------
MLFLOW_EXPERIMENT_NAME= os.getenv("MLFLOW_EXPERIMENT_NAME", "test")
MLFLOW_TRACKING_INSECURE_TLS ="true" #disable TSL
MLFLOW_DISABLE_UV_ENV_DETECTION ="true" #to avoid uv to be detected
S3_BUCKET_ARTIFACT_TRAINING="s3://projet-bdc-alptis-g2/Artifacts_model_training"
MLFLOW_S3_IGNORE_TLS="true"
MLFLOW_S3_ENDPOINT_URL="https://minio-simple.lab.groupe-genes.fr"
MLFLOW_TRACKING_URI= os.getenv("MLFLOW_TRACKING_URI", "https://projet-bdc-data-mlflow.lab.groupe-genes.fr")

#-----------------------------------------------
#---------------DATA PROCESSING--------------
#--------------------------------------------

TARGET_COLUMN='target_resiliation_6mois'
KEY_COLUMN = "client_code"

REFERENCE_DATES = {
    "training":  pd.Timestamp("2024-05-31"),
    "validation": pd.Timestamp("2025-05-31"),  
    "evaluation":  pd.Timestamp("2025-05-31")
}

FILES_TO_DROP= ["impaye"] #drop all the columns that begin with that
COLUMNS_TO_DROP = ["client_code_postal", #car on ajoute le revenu médian commune à la place
                    "client_structure_familiale", "client_nom_banque", "client_nps_date_reponse_n_moins1", "client_nps_date_reponse_n", 'client_nps_date_reponse_n_moins1_jours', 'client_nps_date_reponse_n_jours'] #nom_banque car comme valeurs manquantes, compliqué à target-encodé puis normalisé 

#ajouter scores nps dans columns_to_drop et ajouter date_début_effet_garantie ?
COLUMNS_TO_PROCESSED_WITH_NLP= ['interaction_historique_mail',"client_nps_verbatim_n", "client_nps_verbatim_n_moins1"]


#----FEATURES ENGINEERING----------------

#categorical encoding
ANCIENNETE_COURTIER_COLUMN = 'courtier_anciennete_annees'
ANCIENNETE_BINS = [-1, 2, 7, 12, 20, 40, 200] #cf graph of Overview alptis (-1 )
ANCIENNETE_LABELS= ["new", "recent", "stable", "old", "very old", "ancient"]

AGE_COLUMN= "client_age_souscripteur"
AGE_BINS = [0, 18, 25, 35, 50, 60, 80, 120]  #based on younger and older person in portefeuille (one person has 16 yo...)
AGE_LABELS = ['child', 'young', 'adult', 'mature','middle aged', 'senior', 'senior plus'] 

RECLA_DELAIS_COURT = 3 #(<3 jours: court)
RECLA_DELAIS_LONG= 15 #(>15 jours: long, 3-15: moyen)

#external data 
NEW_COLUMN_REVENU_INSEE = "client_revenu_commune_2021"
REVENU_MEDIAN_FRANCE_2021 = 23160

#--------ENCODING-------------------------------------

HIGH_CARDINALITY=10
EXCEPT_HIGH_CARDINALITY = ["courtier_segmentation_interne"] #variable to one-hot encode despite high cardinality
#to avoid discrepancies test-training because not same categories
FIXED_CATEGORIES = {
    "courtier_segmentation_interne": ['Groupements','Sommeil','Challenger','Potentiel','CMA','Nouveau','Miltis','VADISTE','Filiale','Non catégorisé','Opportuniste','Dilemme','Inactif à relancer','Potentiel Agent','Partenariats','VIP', 'Alptis', 'Petit Producteur'], }

COLUMNS_ORDINAL = {
    "age_categories": AGE_LABELS,           # ['young', 'adult', 'mature', ...]
    "courtier_anciennete_categories": ANCIENNETE_LABELS,
    'client_frequence_paiement': ['SEM','MS','TRIM','AN'], #valeurs de frequence paiement ordonnées
}

#variables that counts days between values of columns and REFERENCE_DATES (becomes: {variable}_jours/mois when created)
TIMESTAMP_COLUMNS_DAYS = ["client_nps_date_reponse_n_moins1", "client_nps_date_reponse_n"]
TIMESTAMP_COLUMNS_MONTHS= [ "client_date_debut_effet_garantie", "dernier_paiement_consommation"]

#----------ENVIRONNEMENT SETTING--------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# To load locally or from S3
LOAD_FROM_S3 = True

#If LOAD_FROM_S3=False, put dataset in following dir: 
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data")


# S3 config
S3_ENDPOINT = "minio-simple.lab.groupe-genes.fr"
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")
S3_BUCKET = "projet-bdc-data"
S3_DATA_PROCESSED_BUCKET= "projet-bdc-alptis-g2" #where the data processed (validation/training) is saved after pre-processing 
S3_VERIFY_SSL = False


# Dataset file configurations

TRAINING_FILES = {
    "consommations": "apprentissage_consommations_2023_11.csv",
    "impayes": "apprentissage_impayes_2023_11.csv",
    "interactions": "apprentissage_interactions_2023_11.csv",
    "portefeuille": "apprentissage_portefeuille_2023_11.csv",
    "reclamations": "apprentissage_reclamations_2023_11.csv",
}

# Evaluation (2024-11)
EVALUATION_FILES = {
    "consommations": "evaluation_consommations_2024_11.csv",
    "impayes": "evaluation_impayes_2024_11.csv",
    "interactions": "evaluation_interactions_2024_11.csv",
    "portefeuille": "evaluation_portefeuille_2024_11.csv",
    "reclamations": "evaluation_reclamations_2024_11.csv",
}

#Validation files (2024-11)
VALIDATION_FILES = {
    "consommations": "validation_consommations_2024_11.csv",
    "impayes": "validation_impayes_2024_11.csv",
    "interactions": "validation_interactions_2024_11.csv",
    "portefeuille": "validation_portefeuille_2024_11.csv",
    "reclamations": "validation_reclamations_2024_11.csv",
}



# Map sets to their file configurations
DATASET_MAPPING = {
    "training": TRAINING_FILES,
    "evaluation": EVALUATION_FILES,
    "validation": VALIDATION_FILES,  
}
