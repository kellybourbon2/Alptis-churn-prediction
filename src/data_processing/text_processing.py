"""Here are the files to create qualitative features from the textual data
coming from nps reviews and mail from interaction"""

import pandas as pd
import numpy as np

def create_nps_features(df):
    """
    Create key features for churn prediction based on NPS scores and verbatims of year n.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing:
        - 'client_nps_note_reco_n'
        - 'client_nps_verbatim_n'

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

def create_mails_features(df):
    """
    Create key features for churn prediction based on mails from texte of mails
    in interaction file.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing:
        - 'interaction_texte_mail'

    Returns
    -------
    pandas.DataFrame
        DataFrame enriched with:
        - 'interaction_mail_churn_mention'
        -'interaction_mail_price_mention'
        - 'interaction_mail_cancel_churn_mention'
    """
    
    #1- Boolean variable on price mention in mails
    keywords_price = [
        "prix", "tarif", "tarifs", "cotisation", "cotisations",
        "augmentation", "augmentations", "cher", "coût", "cout"
    ]

    def mention_price(text):
        if pd.isna(text):
            return 0
        text = str(text).lower()
        return int(any(word in text for word in keywords_price))

    df["interaction_mail_price_mention"] = df["interaction_texte_mail"].apply(mention_price)

    # 2- Boolean variable on churn signal in verbatim
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

    df["interaction_mail_churn_mention"] = df["interaction_texte_mail"].apply(mention_cancel)

    #-3 Boolean indicate if has cancel resiliation

    #Select only motif Resiliation 
    mask= (df["interaction_motif"]=="Résiliation")

    keywords_cancel_resiliation = ["annuler", "annulation", "annule", "désiste"]

    def mention_cancel_resiliation(text):
        if pd.isna(text):
            return 0
        text = str(text).lower()
        return int(any(word in text for word in keywords_cancel_resiliation))

    df["interaction_mail_cancel_churn_mention"] = df[mask]["interaction_texte_mail"].apply(mention_cancel_resiliation)
    df.fillna({"interaction_mail_cancel_churn_mention":0}, inplace=True)

    return df

