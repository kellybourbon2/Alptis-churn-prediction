"""Orchestrate the complete data preprocessing pipeline"""

from .aggregation import aggregate_reclamations, aggregate_consommations, aggregate_impayes, aggregate_interaction
from .cleaning import portefeuille_cleaning


def preprocessing_pipeline(df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes):
    """Complete preprocessing pipeline: clean, aggregate, merge, and fill missing values.
    
    Args:
        df_portefeuille: Portfolio DataFrame
        df_consommations: Raw consumption DataFrame
        df_reclamations: Raw reclamations DataFrame
        df_interactions: Raw interactions DataFrame
        df_impayes: Raw overdue payments DataFrame
    
    Returns:
        Cleaned and fully merged DataFrame with one row per client
    """
    
    # Step 1: Clean portfolio
    df_portefeuille = portefeuille_cleaning(df_portefeuille)

    # Step 2: Aggregate secondary files to client level
    df_consommations = aggregate_consommations(df_consommations)
    df_reclamations = aggregate_reclamations(df_reclamations)
    df_impayes = aggregate_impayes(df_impayes)
    df_interactions = aggregate_interaction(df_interactions)

    # Step 3: Merge all on client_code
    df = df_portefeuille.copy()
    for other_df in [df_reclamations, df_consommations, df_impayes, df_interactions]:
        df = df.merge(other_df, how="left", on="client_code")

    # Step 4: Fill missing values by feature type
    _fill_reclamations_na(df)
    _fill_consumption_na(df)
    _fill_interactions_na(df)
    _fill_overdue_na(df)

    return df


def _fill_reclamations_na(df):
    """Fill NaN with 0 for reclamation count/indicator columns"""
    exclude_cols = {"recla_canal_entrant_principale", "recla_canal_entrant_secondaire", "recla_date_reception"}
    recla_cols = [c for c in df.columns if c.startswith("recla") and c not in exclude_cols]
    df[recla_cols] = df[recla_cols].fillna(0)


def _fill_consumption_na(df):
    """Fill NaN with 0 for consumption amount columns"""
    conso_cols = [c for c in df.columns if c.startswith(("frais", "remb", "nb_decomptes"))]
    df[conso_cols] = df[conso_cols].fillna(0)


def _fill_interactions_na(df):
    """Fill NaN with 0 for interaction count columns (except email history)"""
    exclude_cols = {"interaction_historique_mail"}
    inter_cols = [c for c in df.columns if c.startswith("interaction") and c not in exclude_cols]
    df[inter_cols] = df[inter_cols].fillna(0)


def _fill_overdue_na(df):
    """Fill NaN with 0 for overdue payment columns"""
    impaye_cols = [c for c in df.columns if c.startswith("impaye")]
    df[impaye_cols] = df[impaye_cols].fillna(0)
