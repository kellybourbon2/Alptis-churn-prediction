"""Here are the files to create qualitative features from the textual data
coming from nps reviews and mail from interaction"""

import pandas as pd
import numpy as np

def create_nps_features(df, encoding=None):
    """
    Create key features for churn prediction based on NPS scores and verbatims of year n.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing:
        - 'client_nps_note_reco_n'
        - 'client_nps_verbatim_n'

    encoding : str or None, default=None
        Encoding method for NPS category:
        - None      : keep only categorical labels (default)
        - "ordinal" : 0 (detractor), 1 (passive), 2 (promoter)
        - "onehot"  : creates dummy variables

    Returns
    -------
    pandas.DataFrame
        DataFrame enriched with:
        - 'client_nps_category_n'
        - 'client_nps_price_mention_n'
        - 'client_nps_churn_mention_n'
    """
    # 1. Create a variable of NPS category (passive, promotor, detractor)
    def nps_category(score):
        if pd.isna(score):
            return None
        elif score <= 6:
            return "detractor"
        elif score <= 8:
            return "passive"
        else:
            return "promoter"

    df["client_nps_category_n"] = df["client_nps_note_reco_n"].apply(nps_category)

    # 2. Boolean variable on price mention in NPS verbatim
    keywords_price = [
        "prix", "tarif", "tarifs", "cotisation", "cotisations",
        "augmentation", "augmentations", "cher", "coût", "cout"
    ]

    def mention_price(text):
        if pd.isna(text):
            return 0
        text = str(text).lower()
        return int(any(word in text for word in keywords_price))

    df["client_nps_price_mention_n"] = df["client_nps_verbatim_n"].apply(mention_price)

    # 3. Boolean variable on churn signal in verbatim
    keywords_resiliation = [
        "résiliation", "resiliation", "résilier", "resilier", "résilié", "resilie",
        "quitter", "partir", "je pars", "je vais partir", "je pense partir",
        "changer d'assurance", "aller ailleurs", "annuler", "mettre fin", "stop",
        "résilier mon contrat", "je quitte", "je change",
        "ria", "rad"
    ]

    def mention_cancel(text):
        if pd.isna(text):
            return 0
        text = str(text).lower()
        return int(any(word in text for word in keywords_resiliation))

    df["client_nps_churn_mention_n"] = df["client_nps_verbatim_n"].apply(mention_cancel)

    return df
