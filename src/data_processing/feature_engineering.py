"""File where the features engineering is conducted"""

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
)

def feature_engineering(
    df: pd.DataFrame,
    age_column=AGE_COLUMN,
    anciennete_courtier_column=ANCIENNETE_COURTIER_COLUMN,
    age_labels=AGE_LABELS,
    age_bins=AGE_BINS,
    anciennete_bins=ANCIENNETE_BINS,
    anciennete_labels=ANCIENNETE_LABELS
)-> pd.DataFrame:

    """Creation of new columns based on existing columns in merged file
    Args:
      df (DataFrame):  merged dataframe
    Returns:
      DataFrame: merged dataframe with new features created
    """
    # Creation of a categorical variables and one-hot encoding
    df["age_categories"] = pd.cut(df[age_column], bins=age_bins, labels=age_labels)
    df["age_categories"] = df["age_categories"].map({label: i for i, label in enumerate(age_labels)}) #one-hot encoding

    df["courtier_anciennete_categories"] = pd.cut(df[anciennete_courtier_column], bins= anciennete_bins, labels=anciennete_labels)
    df["courtier_anciennete_categories"] = df["courtier_anciennete_categories"].map({label: i for i, label in enumerate(anciennete_labels)}) #one-hot-encoding

    return df

