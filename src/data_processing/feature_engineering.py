"""File where the features engineering on the variables from
'portefeuille' file is conducted"""

import pandas as pd
import numpy as np
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from Config import (
    AGE_COLUMN,
    AGE_BINS,
    AGE_LABELS,
    ANCIENNETE_COURTIER_COLUMN,
    ANCIENNETE_BINS,
    ANCIENNETE_LABELS,
    TIMESTAMP_COLUMNS_DAYS, 
    TIMESTAMP_COLUMNS_MONTHS
)

def feature_engineering(
    df: pd.DataFrame,
    ref_date: pd.Timestamp, 
    optional_fill_missing_values: bool,
    date_days_columns= TIMESTAMP_COLUMNS_DAYS,
    date_months_columns=  TIMESTAMP_COLUMNS_MONTHS,
    age_column=AGE_COLUMN,
    anciennete_courtier_column=ANCIENNETE_COURTIER_COLUMN,
    age_labels=AGE_LABELS,
    age_bins=AGE_BINS,
    anciennete_bins=ANCIENNETE_BINS,
    anciennete_labels=ANCIENNETE_LABELS,
)-> pd.DataFrame:

    """Creation of new columns based on existing columns in merged file
    Args:
      df (DataFrame):  merged dataframe
    Returns:
      DataFrame: merged dataframe with new features created
    """
    #1- Creation of a categorical variables on age and anciennete 
    df["age_categories"] = pd.cut(df[age_column], bins=age_bins, labels=age_labels)
    df["courtier_anciennete_categories"] = pd.cut(df[anciennete_courtier_column], bins= anciennete_bins, labels=anciennete_labels)


    #2- Create "dernier_paiement_consommation": latest date de consommation enregistrée dans le fichier consommation
    df["dernier_paiement_consommation"] = df["annee_mois_paiement"].apply(
        lambda dates: max(pd.to_datetime(dates, format="%Y-%m")) if isinstance(dates, list) else pd.NaT)
    df.drop(columns=["annee_mois_paiement"], inplace=True) #drop old columns

    #3-Creation of variables on TimeStamp columns: count the days between ref_date and the time
    for col in date_days_columns:
        df[f"{col}_jours"] = (
            ref_date - pd.to_datetime(df[col], format='%Y-%m-%d')
        ).dt.days
        df.fillna({f"{col}_jours": -1}, inplace=True) #fill missing values with '-1': indicates that no nps were left
        df.drop(columns=col, inplace=True) #drop old columns

    #4-Creation of variables on TimeStamp columns: count the months between ref_date and the time
    for col in date_months_columns:
        df[f"{col}_mois"] = (
            (ref_date.year - pd.to_datetime(df[col], format='%Y-%m-%d').dt.year) * 12 +
            (ref_date.month - pd.to_datetime(df[col], format='%Y-%m-%d').dt.month))
        df.fillna({f"{col}_mois": -1}, inplace=True) #fill missing values with '-1': indicates that no paiement occured
        df.drop(columns=col, inplace=True) #drop old columns
    
    #5- Creation of client_cotisation_rate_n_nplus_1
    df["client_cotisations_taux_croissance_n_plus1_n"]= (df["client_cotisations_annualisees_n_plus1"]- df["client_cotisations_annualisees_n"])/df["client_cotisations_annualisees_n"]
    df["client_cotisations_taux_croissance_n_n_moins1"]= (df["client_cotisations_annualisees_n"] - df["client_cotisations_annualisees_n_moins1"])/ df["client_cotisations_annualisees_n_moins1"]
    
    #Sanitize from inf/-inf created
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if optional_fill_missing_values:
        #Optional filling of new missing values with -1
        df["client_cotisations_taux_croissance_n_n_moins1"]=df["client_cotisations_taux_croissance_n_n_moins1"].fillna(-1)
        df["client_cotisations_taux_croissance_n_plus1_n"]=df["client_cotisations_taux_croissance_n_plus1_n"].fillna(-1)
    
    #6 - Creation of boolean variable for future clipping "client_toujours_engagé": indicates whether or not the client is still engaged given the tenure of its contract
    df["client_toujours_engage"] = df["client_date_debut_effet_garantie_mois"] <= 11.5 #11.5 because 6 months for target retrieval (end of may- end of november)

    return df
    
