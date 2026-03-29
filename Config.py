"""Configuration file for data paths and dataset specifications"""

import os
from dotenv import load_dotenv
import pandas as pd

#-----------------------------------------------
#---------------DATA PROCESSING--------------
#--------------------------------------------

TARGET_COLUMN='target_resiliation_6mois'
KEY_COLUMN = "client_code"

REFERENCE_DATES = {
    "training":  pd.Timestamp("2023-05-31"),
    "validation": pd.Timestamp("2024-05-31"),  
    "evaluation":  pd.Timestamp("2024-05-31")
}

FILES_TO_DROP= ["impaye"] #drop all the columns that begin with that
COLUMNS_TO_DROP = ["client_structure_familiale", "client_nom_banque", "client_nps_date_reponse_n_moins1", "client_nps_date_reponse_n", 'client_nps_date_reponse_n_moins1_jours', 'client_nps_date_reponse_n_jours'] #nom_banque car comme valeurs manquantes, compliqué à target-encodé puis normalisé 
#ajouter scores nps dans columns_to_drop et ajouter date_début_effet_garantie ?
COLUMNS_TO_PROCESSED_WITH_NLP= ['interaction_historique_mail',"client_nps_verbatim_n", "client_nps_verbatim_n_moins1"]


#----FEATURES ENGINEERING----------------

#categorical encoding
ANCIENNETE_COURTIER_COLUMN = 'courtier_anciennete_annees'
ANCIENNETE_BINS = [0, 2, 7, 12, 20, 40] #cf graph of Overview alptis 
ANCIENNETE_LABELS= ["new", "recent", "stable", "old", "very old"]

AGE_COLUMN= "client_age_souscripteur"
AGE_BINS = [18, 25, 35, 50, 60, 80, 120]  #based on younger and older person in portefeuille
AGE_LABELS = ['young', 'adult', 'mature','middle aged', 'senior', 'senior plus'] 

RECLA_DELAIS_COURT = 3 #(<3 jours: court)
RECLA_DELAIS_LONG= 15 #(>15 jours: long, 3-15: moyen)

#external data 
NEW_COLUMN_REVENU_INSEE = "client_revenu_commune_2021"
REVENU_MEDIAN_FRANCE_2021 = 23160

#--------ENCODING-------------------------------------

HIGH_CARDINALITY=10
EXCEPT_HIGH_CARDINALITY = ["courtier_segmentation_interne"] #variable to one-hot encode despite high cardinality
NO_ACTION_HIGH_CARDINALITY = ["client_code_postal", "client_departement"] #No target encoding

COLUMNS_ORDINAL = {
    "age_categories": AGE_LABELS,           # ['young', 'adult', 'mature', ...]
    "courtier_anciennete_categories": ANCIENNETE_LABELS,
    'client_frequence_paiement': ['SEM','MS','TRIM','AN'], #valeurs de frequence paiement ordonnées
}

#variables that counts days between values of columns and REFERENCE_DATES (becomes: {variable}_jours/mois when created)
TIMESTAMP_COLUMNS_DAYS = ["client_nps_date_reponse_n_moins1", "client_nps_date_reponse_n"]
TIMESTAMP_COLUMNS_MONTHS= [ "client_date_debut_effet_garantie", "dernier_paiement_consommation"]

#----------ENVIRONNEMENT SETTING--------------------------

# Load environment variables
load_dotenv(override=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# To load locally or from S3
LOAD_FROM_S3 = True

#If LOAD_FROM_S3=False, put dataset in following dir: 
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data")


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


# Cols to be transformed with external data
COLUMNS_EXTERNAL_TRANSFORM = ['client_cotisations_annualisees_n_moins2', 'client_cotisations_annualisees_n_moins1', 'client_cotisations_annualisees_n', 'client_cotisations_annualisees_n_plus1']