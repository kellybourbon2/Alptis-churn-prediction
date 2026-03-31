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
    COLUMNS_ORDINAL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_reference_date(
    mode: Literal["training", "validation", "evaluation"]
) -> pd.Timestamp:
    """Retrieve reference date based on the loaded dataset split."""
    if mode not in REFERENCE_DATES:
        raise ValueError(
            f"Mode '{mode}' inconnu. Choix possibles : {list(REFERENCE_DATES.keys())}"
        )
    return REFERENCE_DATES[mode]


class DataProcessor:
    """Data processing pipeline:
    1. manual preprocessing: cleaning, aggregation, feature engineering, drop useless columns
    2. categorical encoding
    3. normalization of numeric columns
    """

    def __init__(
        self,
        mode: Literal["training", "validation", "evaluation"] = "training",
    ):
        self.mode = mode
        self.ref_date = get_reference_date(mode)
        self.scaler = None
        self.target_col = TARGET_COLUMN
        self.high_cardinality = HIGH_CARDINALITY
        self.except_high_cardinality = EXCEPT_HIGH_CARDINALITY
        self.files_to_drop = tuple(FILES_TO_DROP)
        self.columns_to_drop = list(COLUMNS_TO_DROP) + list(COLUMNS_TO_PROCESSED_WITH_NLP)
        self.cols_to_exclude_encoding = (
            {TARGET_COLUMN}
            | set(COLUMNS_TO_PROCESSED_WITH_NLP)
            | set(COLUMNS_TO_DROP)
        )
        self.ordinal_columns = COLUMNS_ORDINAL

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, optional_normalisation: bool = True) -> pd.DataFrame:
        """Full pipeline: load, preprocess, encode, and optionally normalize."""
        (
            df_portefeuille,
            df_consommations,
            df_reclamations,
            df_interactions,
            df_impayes,
        ) = data_loading(self.mode)

        df_processed = self.manual_preprocessing(
            df_portefeuille,
            df_consommations,
            df_reclamations,
            df_interactions,
            df_impayes,
        )
        df_encoded = self.data_encoding(df_processed)

        if optional_normalisation:
            logger.info("Normalisation applied on numerical values")
            df_ready = self.data_normalization(df_encoded)
            return df_ready

        return df_encoded

    # ------------------------------------------------------------------
    # Step 1 : Manual preprocessing
    # ------------------------------------------------------------------

    def manual_preprocessing(
        self,
        df_portefeuille: pd.DataFrame,
        df_consommations: pd.DataFrame,
        df_reclamations: pd.DataFrame,
        df_interactions: pd.DataFrame,
        df_impayes: pd.DataFrame,
    ) -> pd.DataFrame:
        """Complete manual preprocessing pipeline."""

        # Step 1: Clean portfolio
        df_portefeuille = portefeuille_cleaning(df_portefeuille)

        # Step 2: Aggregate secondary files to client level
        df_consommations = aggregate_consommations(df_consommations)
        df_reclamations = aggregate_reclamations(
            df_reclamations, ref_date=self.ref_date
        )
        df_impayes = aggregate_impayes(df_impayes)
        df_interactions = aggregate_interaction(df_interactions)

        # Step 3: Merge all on client_code
        df = df_portefeuille.copy()
        for other_df in [
            df_reclamations,
            df_consommations,
            df_impayes,
            df_interactions,
        ]:
            df = df.merge(other_df, how="left", on="client_code")

        # Step 4: Feature engineering
        df = feature_engineering(df, ref_date=self.ref_date)

        # Step 5: Fill missing values
        self._fill_reclamations_na(df)
        self._fill_consumption_na(df)
        self._fill_interactions_na(df)
        self._fill_overdue_na(df)

        # Step 6: Drop useless columns
        cols_to_drop = [
            col
            for col in df.columns
            if col.startswith(self.files_to_drop) or col in self.columns_to_drop
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"Dropped columns from dataset: {cols_to_drop}")

        return df

    # ------------------------------------------------------------------
    # Step 2 : Encoding
    # ------------------------------------------------------------------

    def data_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns not treated manually."""
        df = df.copy()

        # Drop non-encodable columns
        cols_to_drop = [
            col
            for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (list, dict, pd.Timestamp))).any()
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        if cols_to_drop:
            logger.warning(f"Unencodable columns needed to be dropped: {cols_to_drop}")

        # Select categorical columns to encode
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in self.cols_to_exclude_encoding]

        one_hot_cols = []
        target_encoded_cols = []
        ordinal_encoded_cols = []

        for col in cat_cols:
            # Ordinal encoding
            if col in self.ordinal_columns:
                ordinal_mapping = {
                    label: i for i, label in enumerate(self.ordinal_columns[col])
                }
                df[col] = df[col].map(ordinal_mapping).astype(float)
                ordinal_encoded_cols.append(col)

            # One-hot encoding
            elif (
                df[col].nunique() <= self.high_cardinality
                or col in self.except_high_cardinality
            ):
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=col), dummies], axis=1)
                one_hot_cols.append(col)

            # Target encoding
            else:
                means = df.groupby(col)[self.target_col].mean()
                df[col] = df[col].map(means)
                target_encoded_cols.append(col)

        logger.info(f"Encoded with one-hot encoding: {one_hot_cols}")
        logger.info(f"Encoded with target encoding: {target_encoded_cols}")
        logger.info(f"Encoded with ordinal encoding: {ordinal_encoded_cols}")

        return df

    # ------------------------------------------------------------------
    # Step 3 : Normalization
    # ------------------------------------------------------------------

    def data_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize continuous numeric columns and remove quasi-constant columns."""
        df = df.copy()

        num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

        cols_to_normalize = [
            col
            for col in num_cols
            if col not in self.cols_to_exclude_encoding
            and df[col].nunique() > self.high_cardinality
            and col not in self.except_high_cardinality
        ]

        self.scaler = StandardScaler()
        df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        logger.info(f"Normalized: {cols_to_normalize}")

        # Remove quasi-constant columns, but keep target and client_code
        exclude_cols = [self.target_col, "client_code"]

        quasi_cols = [
            col
            for col in df.columns
            if col not in exclude_cols
            and df[col].value_counts(normalize=True, dropna=False).iloc[0] > 0.95
        ]

        logger.info(f"Quasi-constant columns dropped: {quasi_cols}")
        df = df.drop(columns=quasi_cols, errors="ignore")

        return df

    # ------------------------------------------------------------------
    # Utils: fill NaN values in aggregated files
    # ------------------------------------------------------------------

    def _fill_reclamations_na(self, df: pd.DataFrame) -> None:
        exclude_cols = {
            "recla_canal_entrant_principale",
            "recla_canal_entrant_secondaire",
            "recla_latests_jours",
        }
        recla_cols = [
            c for c in df.columns if c.startswith("recla") and c not in exclude_cols
        ]
        df[recla_cols] = df[recla_cols].fillna(0)

    def _fill_consumption_na(self, df: pd.DataFrame) -> None:
        conso_cols = [
            c
            for c in df.columns
            if c.startswith(("frais", "remb", "nb", "reste_a_charge"))
        ]
        df[conso_cols] = df[conso_cols].fillna(0)

    def _fill_interactions_na(self, df: pd.DataFrame) -> None:
        exclude_cols = {"interaction_historique_mail"}
        inter_cols = [
            c for c in df.columns if c.startswith("interaction") and c not in exclude_cols
        ]
        df[inter_cols] = df[inter_cols].fillna(0)

    def _fill_overdue_na(self, df: pd.DataFrame) -> None:
        impaye_cols = [c for c in df.columns if c.startswith("impaye")]
        df[impaye_cols] = df[impaye_cols].fillna(0)


if __name__ == "__main__":
    processor = DataProcessor(mode="training")
    df_train = processor.run()
    print(df_train.head())
    print(df_train.shape)