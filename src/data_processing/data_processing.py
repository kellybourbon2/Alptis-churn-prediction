"""File where the whole data processing is conducted"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from src.data_processing.data_external import add_revenu_insee


from Config import (
    TARGET_COLUMN,
    REFERENCE_DATES,
    HIGH_CARDINALITY,
    EXCEPT_HIGH_CARDINALITY,
    FIXED_CATEGORIES,          
    FILES_TO_DROP,
    COLUMNS_TO_DROP,
    COLUMNS_TO_PROCESSED_WITH_NLP,
    COLUMNS_ORDINAL,
    NEW_COLUMN_REVENU_INSEE,
    REVENU_MEDIAN_FRANCE_2021,
    KEY_COLUMN
)

def get_reference_date(mode: Literal["training", "validation", "evaluation"]) -> pd.Timestamp:
    """retrieve reference date based on the set loaded (training, validation, evaluation)"""
    if mode not in REFERENCE_DATES:
        raise ValueError(f"Mode '{mode}' inconnu. Choix possibles : {list(REFERENCE_DATES.keys())}")
    return REFERENCE_DATES[mode]

class DataProcessor:
    """Data Processing following that order:
         1. manual preprocessing: cleaning, aggregating, feature engineering, external data and drop unnecessary columns
         2. automatic encoding of categorical variables (ordinal, target or one-hot given config)
         3. Normalization of numerical variables (non categoricals)
    """

    def __init__(self, mode: Literal["training", "validation", "evaluation"] = "training"):
        self.mode                     = mode
        self.ref_date                 = get_reference_date(mode)
        self.scaler                   = None
        self.encoded_columns          = None
        self.target_encoding_maps     = {}
        self.normalized_columns       = None
        self.global_means             = {}  # mean of target on training dataset (use to fill new values when target-encoding)
        self.target_col               = TARGET_COLUMN
        self.high_cardinality         = HIGH_CARDINALITY
        self.except_high_cardinality  = EXCEPT_HIGH_CARDINALITY
        self.fixed_categories         = FIXED_CATEGORIES  # ← {col: [categories]} for columns with fixed vocab
        self.fixed_categories_fitted  = {}                # ← populated at fit time, reused at transform time
        self.files_to_drop            = tuple(FILES_TO_DROP)
        self.columns_to_drop          = list(COLUMNS_TO_DROP) + list(COLUMNS_TO_PROCESSED_WITH_NLP)
        self.cols_to_exclude_encoding = (
            {TARGET_COLUMN}
            | set(COLUMNS_TO_PROCESSED_WITH_NLP)
            | set(COLUMNS_TO_DROP)
        )
        self.ordinal_orders = COLUMNS_ORDINAL  # from config
        self.ordinal_maps   = {}               # saved during training to be reused in test/val
        self.key_column     = KEY_COLUMN

    # ------------------------------------------------------------------
    # Pipelines run()
    # ------------------------------------------------------------------
    def run(self, optional_encoding=True, optional_normalisation=True) -> pd.DataFrame:
        """Pipeline TRAIN : fit_transform everything"""
        df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes = data_loading(self.mode)
        df_processed = self.manual_preprocessing(df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes)
        if optional_encoding:
            df_processed = self.data_encoding(df_processed)
        if optional_normalisation:
            logger.info(f"Columns that are normalized: {self.normalized_columns}")
            df_processed = self.data_normalization(df_processed)
        return df_processed

    def run_transform(self, optional_encoding=True, optional_normalisation=True) -> pd.DataFrame:
        """Pipeline TEST/VAL : transform only, without re-fitting"""
        df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes = data_loading(self.mode)
        df_processed = self.manual_preprocessing(df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes)
        if optional_encoding:
            df_processed = self.data_encoding_transform(df_processed)
        if optional_normalisation:
            df_processed = self.data_normalization_transform(df_processed)
        return df_processed

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
        """Complete manual preprocessing pipeline:
        1. clean
        2. aggregate
        3. merge on key ()
        4. features engineering
        5. add external data
        6: fill NAs
        7: drop unnecessary columns.

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

        # Step 3: Merge all on key column (client_code)
        df = df_portefeuille.copy()
        for other_df in [df_reclamations, df_consommations, df_impayes, df_interactions]:
            df = df.merge(other_df, how="left", on=self.key_column)

        # Step 4: Feature engineering
        df = feature_engineering(df, ref_date=self.ref_date)

        # Step 5: Add external data
        df = add_revenu_insee(df, path_insee_commune="data_external/revenu_median_communes.xlsx", new_column_revenu_name=NEW_COLUMN_REVENU_INSEE, revenu_median_fr=REVENU_MEDIAN_FRANCE_2021)

        # Step 6: Fill missing values by feature type
        self._fill_reclamations_na(df)
        self._fill_consumption_na(df)
        self._fill_interactions_na(df)
        self._fill_overdue_na(df)

        # Step 7: Drop useless columns
        cols_to_drop = [
            col for col in df.columns
            if col.startswith(self.files_to_drop) or col in self.columns_to_drop
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"Dropped columns from dataset: {cols_to_drop}")

        return df

    # ------------------------------------------------------------------
    # Step 2 : Encoding
    # ------------------------------------------------------------------

    def _encode_fixed_category_col(self, df: pd.DataFrame, col: str, categories: list) -> pd.DataFrame:
        """One-hot encode a column using a fixed, predetermined list of categories.
        - Unknown values (present in data but not in categories) are silently ignored (→ all zeros row)
        - Missing categories (in list but absent from data) are added as zero columns
        This guarantees identical output columns regardless of which values appear in the data.
        """
        df[col] = pd.Categorical(df[col], categories=categories)
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        return pd.concat([df.drop(columns=col), dummies], axis=1)

    # ------------------------------------------------------------------
    # Encoding : fit_transform (call only on TRAIN dataset)
    # ------------------------------------------------------------------
    def data_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform (call on train dataset only):
        Encode categorical columns not treated manually:
        - Drop non-encodable (list/dict/timestamp) first to avoid select_dtypes crash
        - Fixed-category one-hot for columns listed in FIXED_CATEGORIES config
          (vocabulary locked at config time → identical columns on train and test)
        - One-hot for low cardinality (nunique <= high_cardinality, except for the variable in self.except_high_cardinality)
        - Target encoding for high cardinality (nunique > high_cardinality + variable in except_high_cardinality)
        - Ordinal encoding for variables with values that can be ordered
        """
        df = df.copy()
        df = self._drop_unencodable(df)
        cat_cols = self._get_cat_cols(df)
        one_hot_col, target_col, fixed_col = [], [], []

        for col in cat_cols:

            if col in self.ordinal_orders:
                ordinal_mapping = {label: i for i, label in enumerate(self.ordinal_orders[col])}
                self.ordinal_maps[col] = ordinal_mapping  # save mapping for test/val
                df[col] = df[col].map(ordinal_mapping).astype(float)

            elif col in self.fixed_categories:
                # Fixed-vocab one-hot: categories locked in config, unknown values → all zeros
                # Fit: record the exact category list that was used (from config, not from data)
                categories = self.fixed_categories[col]
                self.fixed_categories_fitted[col] = categories  # save for transform
                df = self._encode_fixed_category_col(df, col, categories)
                fixed_col.append(col)

            elif (df[col].nunique() <= self.high_cardinality) or (col in self.except_high_cardinality):
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=col), dummies], axis=1)
                one_hot_col.append(col)

            else:  # target-encoding
                self.global_means[col] = df[self.target_col].mean()  # save global mean of target on whole training dataset (to fill new values btw training-validation/training-evaluation)
                means = df.groupby(col)[self.target_col].mean()       # fit on train
                self.target_encoding_maps[col] = means                # save for later (train and validation)
                df[col] = df[col].map(means)
                target_col.append(col)

        # Save columns after encoding (important for reuse on test/val dataset)
        self.encoded_columns = df.columns.tolist()

        logger.info(f"Columns that are Fixed-category one-hot encoded: {fixed_col}")
        logger.info(f"Columns that are One-hot encoded: {one_hot_col}")
        logger.info(f"Columns that are Target encoded: {target_col}")
        logger.info(f"Columns that are Ordinal encoded: {list(self.ordinal_maps.items())}")

        return df

    # ------------------------------------------------------------------
    # Encoding : transform only (sur le TEST/VALIDATION)
    # ------------------------------------------------------------------
    def data_encoding_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform only using the encoder fitted on train (to call on test/val):
        - Fixed-category columns: use the exact category list saved at fit time
        - Target-encoded columns: apply training means, fill unseen values with global mean
        - One-hot columns: encode then reindex to match training schema (missing → 0, extra → dropped)
        """
        df = df.copy()
        df = self._drop_unencodable(df)
        cat_cols = self._get_cat_cols(df)

        for col in cat_cols:

            if col in self.ordinal_maps:
                df[col] = df[col].map(self.ordinal_maps[col]).astype(float)

            elif col in self.fixed_categories_fitted:
                # Fixed-vocab one-hot: reuse exact category list from training — unknown values → all zeros
                categories = self.fixed_categories_fitted[col]
                df = self._encode_fixed_category_col(df, col, categories)

            elif col in self.target_encoding_maps:
                global_mean = self.global_means[col]
                df[col] = df[col].map(self.target_encoding_maps[col])  # encode with target-mean computed on training dataset
                df[col] = df[col].fillna(global_mean)                  # fill missing values (new values in validation/evaluation with global_means)

            elif (df[col].nunique() <= self.high_cardinality) or (col in self.except_high_cardinality):
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=col), dummies], axis=1)

        # Realign columns on train dataset (missing columns → 0, unknown columns → drop)
        expected = [c for c in self.encoded_columns]
        df = df.reindex(columns=expected, fill_value=0)

        return df

    # ------------------------------------------------------------------
    # Normalization : fit_transform (train) vs transform (test/validation)
    # ------------------------------------------------------------------
    def data_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform (train only):
        normalize columns that are numericals, non binary and not one-hot encoded"""
        df = df.copy()
        cols_to_normalize = self._get_cols_to_normalize(df)
        self.scaler = StandardScaler()
        df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        self.normalized_columns = cols_to_normalize  # ← save to use again in test/val
        return df

    def data_normalization_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform only (test/val):
        Normalize function fitted on train dataset:
        normalize columns that are numericals, non binary and not one-hot encoded
        """
        df = df.copy()
        df[self.normalized_columns] = self.scaler.transform(df[self.normalized_columns])
        return df

    def add_external_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation on some columns based on external data
        Normalize data based on a commune-specific value
        """
        df = df.copy()

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _drop_unencodable(self, df):
        cols_to_drop = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (list, dict, pd.Timestamp))).any()]
        return df.drop(columns=cols_to_drop)

    def _get_cat_cols(self, df):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        return [c for c in cat_cols if c not in self.cols_to_exclude_encoding]

    def _get_cols_to_normalize(self, df):
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        return [col for col in num_cols if col not in self.cols_to_exclude_encoding and df[col].nunique() > self.high_cardinality and col not in self.except_high_cardinality]

    # Functions that replace the NaN values with 0 in aggregated files
    def _fill_reclamations_na(self, df: pd.DataFrame) -> None:
        exclude_cols = {
            "recla_canal_entrant_principale",
            "recla_canal_entrant_secondaire",
            "recla_latests_jours",
        }
        recla_cols = [c for c in df.columns if c.startswith("recla") and c not in exclude_cols]
        df[recla_cols] = df[recla_cols].fillna(0)

    def _fill_consumption_na(self, df: pd.DataFrame) -> None:
        conso_cols = [c for c in df.columns if c.startswith(("frais", "remb", "nb", "reste_a_charge"))]
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

    # Training
    train_processor = DataProcessor(mode="training")
    df_train = train_processor.run()

    # Test:
    test_processor = DataProcessor(mode="evaluation")

    # Reuse the same pre-processor fitted on training dataset
    # (so no data-leakage with target-encoding, normalization, ...)
    test_processor.global_means             = train_processor.global_means            # save global target % to fill new values for target-encoded columns
    test_processor.target_encoding_maps     = train_processor.target_encoding_maps
    test_processor.ordinal_maps             = train_processor.ordinal_maps
    test_processor.fixed_categories_fitted  = train_processor.fixed_categories_fitted  # ← fixed vocab locked at training time
    test_processor.encoded_columns          = train_processor.encoded_columns
    test_processor.scaler                   = train_processor.scaler
    test_processor.normalized_columns       = train_processor.normalized_columns

    df_test = test_processor.run_transform()
    df_test.head(5)