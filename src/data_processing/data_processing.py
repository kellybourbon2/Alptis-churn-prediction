import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import logging
from typing import Literal

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_processing.data_load import data_loading
from src.data_processing.aggregation import (
    aggregate_consommations,
    aggregate_reclamations,
    aggregate_impayes,
    aggregate_interaction,
)
from src.data_processing.cleaning import portefeuille_cleaning
from src.data_processing.feature_engineering import feature_engineering


from Config import (
    TARGET_COLUMN,
    REFERENCE_DATES,
    HIGH_CARDINALITY,
    EXCEPT_HIGH_CARDINALITY,
    FILES_TO_DROP,
    COLUMNS_TO_DROP,
    COLUMNS_TO_PROCESSED_WITH_NLP,
    COLUMNS_ORDINAL
)

def get_reference_date(mode: Literal["training", "validation", "evaluation"]) -> pd.Timestamp:
    """retrieve reference date based on the set loaded (training, validation, evaluation)"""
    if mode not in REFERENCE_DATES:
        raise ValueError(f"Mode '{mode}' inconnu. Choix possibles : {list(REFERENCE_DATES.keys())}")
    return REFERENCE_DATES[mode]

class DataProcessor:
    """Data Processing following that order:
         1. manual preprocessing: cleaning, aggregating, feature engineering and drop unnecessary columns
         2. automatic encoding of categorical variables not treated manually
         3. Normalization of numerical variables (non categoricals)
    """

    def __init__(self, mode: Literal["training", "validation", "evaluation"] = "training"):
        self.mode                     = mode
        self.ref_date                 = get_reference_date(mode)
        self.scaler                   = None
        self.target_col               = TARGET_COLUMN
        self.high_cardinality         = HIGH_CARDINALITY
        self.exception_columns        = list(EXCEPT_HIGH_CARDINALITY)
        self.files_to_drop            = tuple(FILES_TO_DROP) if isinstance(FILES_TO_DROP, list) else FILES_TO_DROP
        self.columns_to_drop          = list(COLUMNS_TO_DROP) + list(COLUMNS_TO_PROCESSED_WITH_NLP)
        self.cols_to_exclude_encoding = (
            {TARGET_COLUMN}
            | set(EXCEPT_HIGH_CARDINALITY)
            | set(COLUMNS_TO_PROCESSED_WITH_NLP)
            | set(COLUMNS_TO_DROP)
            | set(COLUMNS_ORDINAL)
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, optional_normalisation=True) -> pd.DataFrame:
        """Full pipeline: load, manual preprocess, automatic encode and optionally normalize
                Args: 
                 optional_normalisation (bool): whether or not normalization applied on numerical values
        """
        # Load
        df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes = data_loading(self.mode)

        # Process
        df_processed = self.manual_preprocessing(df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes)
        df_encoded   = self.data_encoding(df_processed)
        if optional_normalisation:
            df_ready = self.data_normalization(df_encoded)
            return df_ready
        else: 
            return df_encoded

    # ------------------------------------------------------------------
    # Step 1 : Manual preprocessing
    # ------------------------------------------------------------------

    def manual_preprocessing(
        self,
        df_portefeuille,
        df_consommations,
        df_reclamations,
        df_interactions,
        df_impayes,
    ) -> pd.DataFrame:
        """Complete manual preprocessing pipeline: clean, aggregate, merge, fill NAs and drop unnecessary columns.

        Args:
            df_portefeuille:  Portfolio DataFrame
            df_consommations: Raw consumption DataFrame
            df_reclamations:  Raw reclamations DataFrame
            df_interactions:  Raw interactions DataFrame
            df_impayes:       Raw overdue payments DataFrame

        Returns:
            Cleaned and fully merged DataFrame with one row per client
        """
        # Step 1: Clean portfolio
        df_portefeuille = portefeuille_cleaning(df_portefeuille)

        # Step 2: Aggregate secondary files to client level
        df_consommations = aggregate_consommations(df_consommations)
        df_reclamations  = aggregate_reclamations(df_reclamations, ref_date=self.ref_date)
        df_impayes       = aggregate_impayes(df_impayes)
        df_interactions  = aggregate_interaction(df_interactions)

        # Step 3: Merge all on client_code
        df = df_portefeuille.copy()
        for other_df in [df_reclamations, df_consommations, df_impayes, df_interactions]:
            df = df.merge(other_df, how="left", on="client_code")

        # Step 4: Feature engineering
        df = feature_engineering(df)

        # Step 5: Fill missing values by feature type
        self._fill_reclamations_na(df)
        self._fill_consumption_na(df)
        self._fill_interactions_na(df)
        self._fill_overdue_na(df)

        # Step 6: Drop useless columns
        cols_to_drop = [
            col for col in df.columns
            if col.startswith(self.files_to_drop) or col in self.columns_to_drop
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        logging.info(f"Dropped columns: {cols_to_drop}")

        return df

    # ------------------------------------------------------------------
    # Step 2 : Encoding
    # ------------------------------------------------------------------

    def data_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns not treated manually:
        - Drop non-encodable (list/dict/timestamp) first to avoid select_dtypes crash
        - One-hot for low cardinality  (nunique <= high_cardinality)
        - Target encoding for high cardinality (nunique > high_cardinality)
        """
        df = df.copy()

        # Drop non-encodable columns (list, dict, timestamp) — MUST be before select_dtypes
        cols_to_drop = [
            col for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (list, dict, pd.Timestamp))).any()
        ]
        df.drop(columns=cols_to_drop, inplace=True)
        logging.info(f"Dropped (non-encodable): {cols_to_drop}")

        # Encode categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in self.cols_to_exclude_encoding]

        for col in cat_cols:
            if df[col].nunique() <= self.high_cardinality:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=col), dummies], axis=1)
            else:
                means = df.groupby(col)[self.target_col].mean()
                df[col] = df[col].map(means)

        logging.info(f"Encoded (categorical): {cat_cols}")
        return df

    # ------------------------------------------------------------------
    # Step 3 : Normalization
    # ------------------------------------------------------------------

    def data_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize continuous numeric columns (nunique > high_cardinality).
        Stores scaler as attribute for reuse on test/inference data.
        """
        df = df.copy()

        num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        cols_to_normalize = [
            col for col in num_cols
            if col not in self.cols_to_exclude_encoding
            and df[col].nunique() > self.high_cardinality
        ]

        self.scaler = StandardScaler()
        df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        logging.info(f"Normalized: {cols_to_normalize}")

        return df

    # UTILS -------------

    def _fill_reclamations_na(self, df: pd.DataFrame) -> None:
        exclude_cols = {
            "recla_canal_entrant_principale",
            "recla_canal_entrant_secondaire",
            "recla_recence_jours",
        }
        recla_cols = [c for c in df.columns if c.startswith("recla") and c not in exclude_cols]
        df[recla_cols] = df[recla_cols].fillna(0)

    def _fill_consumption_na(self, df: pd.DataFrame) -> None:
        conso_cols = [c for c in df.columns if c.startswith(("frais", "remb", "nb_decomptes"))]
        df[conso_cols] = df[conso_cols].fillna(0)

    def _fill_interactions_na(self, df: pd.DataFrame) -> None:
        exclude_cols = {"interaction_historique_mail"}
        inter_cols = [c for c in df.columns if c.startswith("interaction") and c not in exclude_cols]
        df[inter_cols] = df[inter_cols].fillna(0)

    def _fill_overdue_na(self, df: pd.DataFrame) -> None:
        impaye_cols = [c for c in df.columns if c.startswith("impaye")]
        df[impaye_cols] = df[impaye_cols].fillna(0)

# ----------------------------------------------------------------------

if __name__ == "__main__":
    processor = DataProcessor(mode="training")
    df_ready  = processor.run()
    print(df_ready.head(10))