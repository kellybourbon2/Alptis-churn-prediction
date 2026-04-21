"""File where the whole data processing is conducted"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
from pandas.errors import PerformanceWarning

#Silent warning
warnings.filterwarnings("ignore", category=PerformanceWarning)
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
from src.data_processing.text_processing import create_nps_features, create_mails_features
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
    KEY_COLUMN,
    COLUMNS_TO_KEEP,
    TRY_FEW_COLUMNS
)


def get_reference_date(mode: Literal["training", "validation", "evaluation"]) -> pd.Timestamp:
    """Retrieve reference date based on the set loaded (training, validation, evaluation)."""
    if mode not in REFERENCE_DATES:
        raise ValueError(f"Mode '{mode}' inconnu. Choix possibles : {list(REFERENCE_DATES.keys())}")
    return REFERENCE_DATES[mode]


class DataProcessor:
    """Data Processing following that order:
         1. Manual preprocessing: cleaning, aggregating, feature engineering,
            external data, missing indicators, imputation, drop unnecessary columns.
         2. Automatic encoding of categorical variables
            (ordinal, target or one-hot given config).
         3. Normalization of numerical variables (non-categoricals).

    Args:
        mode:                   'training', 'validation', or 'evaluation'.
        add_missing_indicators: If True, creates binary `<col>_was_missing` columns
                                BEFORE imputation so the model can learn from
                                the missingness pattern itself. Fitted on train,
                                reused (same column list) on test/val.
    """

    def __init__(
        self,
        mode: Literal["training", "validation", "evaluation"] = "training",
        add_missing_indicators: bool = False,
    ):
        self.mode                     = mode
        self.ref_date                 = get_reference_date(mode)
        self.scaler                   = None
        self.encoded_columns          = None
        self.target_encoding_maps     = {}
        self.normalized_columns       = None
        self.global_means             = {}
        self.target_col               = TARGET_COLUMN
        self.high_cardinality         = HIGH_CARDINALITY
        self.except_high_cardinality  = EXCEPT_HIGH_CARDINALITY
        self.fixed_categories         = FIXED_CATEGORIES
        self.fixed_categories_fitted  = {}
        self.files_to_drop            = tuple(FILES_TO_DROP)
        self.columns_to_drop          = list(COLUMNS_TO_DROP) + list(COLUMNS_TO_PROCESSED_WITH_NLP)
        self.cols_to_exclude_encoding = (
            {TARGET_COLUMN}
            | set(COLUMNS_TO_PROCESSED_WITH_NLP)
            | set(COLUMNS_TO_DROP)
        )
        self.ordinal_orders           = COLUMNS_ORDINAL
        self.ordinal_maps             = {}

        # Missing indicators
        self.add_missing_indicators   = add_missing_indicators
        self.missing_indicator_cols   = []   # fitted on train, reused on test/val

        self.key_column               = KEY_COLUMN

    # ------------------------------------------------------------------
    # Pipelines run()
    # ------------------------------------------------------------------

    def run(
        self,
        optional_encoding: bool = True,
        optional_normalisation: bool = True,
        optional_fill_missing_values: bool = True,
    ) -> pd.DataFrame:
        """Pipeline TRAIN: fit_transform everything."""
        df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes = data_loading(self.mode)

        df_processed = self.manual_preprocessing(
            df_portefeuille,
            df_consommations,
            df_reclamations,
            df_interactions,
            df_impayes,
            optional_fill_missing_values=optional_fill_missing_values,
        )

        if optional_encoding:
            df_processed = self.data_encoding(df_processed)

        if optional_normalisation:
            logger.info(f"Columns that are normalized: {self.normalized_columns}")
            df_processed = self.data_normalization(df_processed)

        return df_processed

    def run_transform(
        self,
        optional_encoding: bool = True,
        optional_normalisation: bool = True,
        optional_fill_missing_values: bool = True,
    ) -> pd.DataFrame:
        """Pipeline TEST/VAL: transform only, without re-fitting."""
        df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes = data_loading(self.mode)

        df_processed = self.manual_preprocessing(
            df_portefeuille,
            df_consommations,
            df_reclamations,
            df_interactions,
            df_impayes,
            optional_fill_missing_values=optional_fill_missing_values,
        )

        if optional_encoding:
            df_processed = self.data_encoding_transform(df_processed)

        if optional_normalisation:
            df_processed = self.data_normalization_transform(df_processed)

        return df_processed

    # ------------------------------------------------------------------
    # Step 1: Manual preprocessing
    # ------------------------------------------------------------------

    def manual_preprocessing(
        self,
        df_portefeuille,
        df_consommations,
        df_reclamations,
        df_interactions,
        df_impayes,
        optional_fill_missing_values: bool,
    ) -> pd.DataFrame:
        """Complete manual preprocessing pipeline:
            1. Clean portfolio
            2. Aggregate secondary files to client level
            3. Merge on key column
            4. Feature engineering
            5. NPS / text features
            6. Add external data
            7. Fill NaN values by feature type
            8. Drop unnecessary columns
        """
        # Step 1: Clean portfolio
        df_portefeuille = portefeuille_cleaning(
            df_portefeuille,
            optional_fill_missing_values=optional_fill_missing_values,
        )

        #Step 1.5: Create booleans on mails of interactions before agregation
        df_interactions = create_mails_features(df_interactions)

        # Step 2: Aggregate secondary files to client level
        df_consommations = aggregate_consommations(df_consommations)
        df_reclamations  = aggregate_reclamations(df_reclamations, ref_date=self.ref_date)
        df_impayes       = aggregate_impayes(df_impayes)
        df_interactions  = aggregate_interaction(df_interactions)
        
        
        #Step 2.5: retrieve missing columns
        if self.add_missing_indicators:
            if self.mode == "training":
                self._fit_missing_indicators(df_portefeuille)
            self._apply_missing_indicators(df_portefeuille)

        # Step 3: Merge all on key column (client_code)
        df = df_portefeuille.copy()
        for other_df in [df_reclamations, df_consommations, df_impayes, df_interactions]:
            df = df.merge(other_df, how="left", on=self.key_column)

        # Step 4: Feature engineering
        df = feature_engineering(
            df,
            ref_date=self.ref_date,
            optional_fill_missing_values=optional_fill_missing_values,
        )

        # Step 5: Add features from textual data (NPS review and mails from interactions)
        df = create_nps_features(df)

        # Step 6: Add external data
        df = add_revenu_insee(
            df,
            new_column_revenu_name=NEW_COLUMN_REVENU_INSEE,
            revenu_median_fr=REVENU_MEDIAN_FRANCE_2021,
        )
        
        # Step 7: Fill missing values by feature type after aggregation
        self._fill_reclamations_na(df)
        self._fill_consumption_na(df)
        # Optional for interactions and impayes since NaN can reflect non-happening events (e.g. dates)
        self._fill_impaye_na(df, optional_fill_missing_values)
        self._fill_interactions_na(df, optional_fill_missing_values)

        # Step 8: Drop useless columns
        if TRY_FEW_COLUMNS:
            cols_to_drop = [
                col for col in df.columns
                if col not in COLUMNS_TO_KEEP
                and col not in ["client_code", "target_resiliation_6mois"]
            ]
        else:
            cols_to_drop = [
                col for col in df.columns
                if (col.startswith(self.files_to_drop) or col in self.columns_to_drop)
                and col not in ["client_code", "target_resiliation_6mois"]
            ]

        df = df.drop(columns=cols_to_drop, errors="ignore")
        logger.info(f"Dropped columns from dataset: {cols_to_drop}")

        return df

    # ------------------------------------------------------------------
    # Step 2: Encoding
    # ------------------------------------------------------------------

    def _encode_fixed_category_col(self, df: pd.DataFrame, col: str, categories: list) -> pd.DataFrame:
        """One-hot encode a column using a fixed, predetermined list of categories.
        - Unknown values (present in data but not in categories) → all-zero row.
        - Missing categories (in list but absent from data) → added as zero columns.
        Guarantees identical output columns regardless of which values appear in data.
        """
        df[col] = pd.Categorical(df[col], categories=categories)
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        return pd.concat([df.drop(columns=col), dummies], axis=1)

    def data_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform (call on train dataset only).
        Encode categorical columns:
        - Drop non-encodable (list/dict/timestamp) first.
        - Fixed-category one-hot for columns listed in FIXED_CATEGORIES.
        - Ordinal encoding for ordered variables.
        - One-hot for low cardinality (nunique <= high_cardinality).
        - Target encoding for high cardinality.
        """
        df = df.copy()
        df = self._drop_unencodable(df)
        cat_cols = self._get_cat_cols(df)
        one_hot_col, target_col, fixed_col = [], [], []

        for col in cat_cols:

            if col in self.ordinal_orders:
                ordinal_mapping = {label: i for i, label in enumerate(self.ordinal_orders[col])}
                self.ordinal_maps[col] = ordinal_mapping
                df[col] = df[col].map(ordinal_mapping).astype(float)

            elif col in self.fixed_categories:
                categories = self.fixed_categories[col]
                self.fixed_categories_fitted[col] = categories
                df = self._encode_fixed_category_col(df, col, categories)
                fixed_col.append(col)

            elif (df[col].nunique() <= self.high_cardinality) or (col in self.except_high_cardinality):
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=col), dummies], axis=1)
                one_hot_col.append(col)

            else:  # target-encoding
                self.global_means[col] = df[self.target_col].mean()
                means = df.groupby(col)[self.target_col].mean()
                self.target_encoding_maps[col] = means
                df[col] = df[col].map(means)
                target_col.append(col)

        self.encoded_columns = df.columns.tolist()

        logger.info(f"Columns that are Fixed-category one-hot encoded: {fixed_col}")
        logger.info(f"Columns that are One-hot encoded: {one_hot_col}")
        logger.info(f"Columns that are Target encoded: {target_col}")
        logger.info(f"Columns that are Ordinal encoded: {list(self.ordinal_maps.items())}")

        return df

    def data_encoding_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform only using encoder fitted on train (call on test/val).
        - Fixed-category columns: reuse exact category list from training.
        - Target-encoded columns: apply training means, fill unseen with global mean.
        - One-hot columns: encode then reindex to match training schema.
        """
        df = df.copy()
        df = self._drop_unencodable(df)
        cat_cols = self._get_cat_cols(df)

        for col in cat_cols:

            if col in self.ordinal_maps:
                df[col] = df[col].map(self.ordinal_maps[col]).astype(float)

            elif col in self.fixed_categories_fitted:
                categories = self.fixed_categories_fitted[col]
                df = self._encode_fixed_category_col(df, col, categories)

            elif col in self.target_encoding_maps:
                global_mean = self.global_means[col]
                df[col] = df[col].map(self.target_encoding_maps[col])
                df[col] = df[col].fillna(global_mean)

            elif (df[col].nunique() <= self.high_cardinality) or (col in self.except_high_cardinality):
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=col), dummies], axis=1)

        # Realign columns on train schema (missing → 0, unknown → drop)
        expected = [c for c in self.encoded_columns]
        df = df.reindex(columns=expected, fill_value=0)

        return df

    # ------------------------------------------------------------------
    # Step 3: Normalization
    # ------------------------------------------------------------------

    def data_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit + transform (train only): normalize non-binary numerical columns."""
        df = df.copy()
        cols_to_normalize = self._get_cols_to_normalize(df)
        self.scaler = StandardScaler()
        df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        self.normalized_columns = cols_to_normalize
        return df

    def data_normalization_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform only (test/val): apply scaler fitted on train."""
        df = df.copy()
        df[self.normalized_columns] = self.scaler.transform(df[self.normalized_columns])
        return df

    # ------------------------------------------------------------------
    # Missing indicators (fit on train, apply on train + test/val)
    # ------------------------------------------------------------------

    def _fit_missing_indicators(self, df: pd.DataFrame) -> None:
        """Detect which columns contain NaN on the training set.
        Saves the resulting indicator column names for reuse on test/val.
        Must be called BEFORE any imputation step.
        """
        cols_with_nan = [col for col in df.columns if df[col].isnull().any()]
        self.missing_indicator_cols = [f"{col}_was_missing" for col in cols_with_nan]
        logger.info(
            f"Missing indicators fitted on {len(cols_with_nan)} columns: {cols_with_nan}"
        )

    def _apply_missing_indicators(self, df: pd.DataFrame) -> None:
        """Create binary `<col>_was_missing` columns in-place.
        On train: uses the list just fitted by _fit_missing_indicators.
        On test/val: reuses the list transferred from the train processor,
                     defaulting to 0 if the original column was dropped.
        Must be called BEFORE any imputation step.
        """
        for indicator_col in self.missing_indicator_cols:
            original_col = indicator_col.replace("_was_missing", "")
            if original_col in df.columns:
                df[indicator_col] = df[original_col].isnull().astype(int)
            else:
                # Column was dropped upstream → no missing signal → fill with 0
                df[indicator_col] = 0

        logger.info(f"Applied {len(self.missing_indicator_cols)} missing indicator columns.")

    # ------------------------------------------------------------------
    # NaN filling helpers
    # ------------------------------------------------------------------

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

    def _fill_interactions_na(self, df: pd.DataFrame, optional_fill_missing_values: bool) -> None:
        inter_cols = [
            c for c in df.columns
            if c.startswith("interaction") and c not in ["derniere_interaction_date_mois"]
        ]
        df[inter_cols] = df[inter_cols].fillna(0)

        if optional_fill_missing_values:
            # Optional: -1 signals a non-happening event (no interaction date recorded)
            df["derniere_interaction_date_mois"] = df["derniere_interaction_date_mois"].fillna(-1)

    def _fill_impaye_na(self, df: pd.DataFrame, optional_fill_missing_values: bool) -> None:
        impaye_cols = [c for c in df.columns if c.startswith("impaye")]
        if optional_fill_missing_values:
            # -1 signals a non-happening event (no overdue payment recorded)
            df[impaye_cols] = df[impaye_cols].fillna(-1)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _drop_unencodable(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            col for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (list, dict, pd.Timestamp))).any()
        ]
        return df.drop(columns=cols_to_drop)

    def _get_cat_cols(self, df: pd.DataFrame) -> list:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        return [c for c in cat_cols if c not in self.cols_to_exclude_encoding]

    def _get_cols_to_normalize(self, df: pd.DataFrame) -> list:
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        return [
            col for col in num_cols
            if col not in self.cols_to_exclude_encoding
            and df[col].nunique() > self.high_cardinality
            and col not in self.except_high_cardinality
        ]


# ----------------------------------------------------------------------

if __name__ == "__main__":

    # Training
    train_processor = DataProcessor(mode="training", add_missing_indicators=True)
    df_train = train_processor.run()

    # Test / Evaluation:
    # Reuse everything fitted on training (no data leakage).
    test_processor = DataProcessor(mode="evaluation", add_missing_indicators=True)

    test_processor.global_means             = train_processor.global_means
    test_processor.target_encoding_maps     = train_processor.target_encoding_maps
    test_processor.ordinal_maps             = train_processor.ordinal_maps
    test_processor.fixed_categories_fitted  = train_processor.fixed_categories_fitted
    test_processor.encoded_columns          = train_processor.encoded_columns
    test_processor.scaler                   = train_processor.scaler
    test_processor.normalized_columns       = train_processor.normalized_columns

    # Transfer the fitted missing-indicator column list so test uses the same columns as train
    test_processor.missing_indicator_cols   = train_processor.missing_indicator_cols

    df_test = test_processor.run_transform()
    df_test.head(5)