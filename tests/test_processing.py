# tests/test_data_processing.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) #so src is visible

import pytest
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def processor():
    from src.data_processing.data_processing import DataProcessor
    return DataProcessor(mode="training")

@pytest.fixture(scope="session")
def df_processed(processor):
    from src.data_processing.data_load import data_loading
    df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes = data_loading("training")
    return processor.manual_preprocessing(df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes)

@pytest.fixture(scope="session")
def df_encoded(processor, df_processed):
    return processor.data_encoding(df_processed)

@pytest.fixture(scope="session")
def df_ready(processor, df_encoded):
    return processor.data_normalization(df_encoded)

@pytest.fixture(scope="session")
def target_col():
    from Config import TARGET_COLUMN
    return TARGET_COLUMN

@pytest.fixture(scope="session")
def high_cardinality():
    from Config import HIGH_CARDINALITY
    return HIGH_CARDINALITY

@pytest.fixture(scope="session")
def cols_normalized(df_ready, target_col, high_cardinality):
    num_cols = df_ready.select_dtypes(include=["float64", "int64"]).columns.tolist()
    return [
        col for col in num_cols
        if col != target_col
        and df_ready[col].nunique() > high_cardinality
    ]


# ------------------------------------------------------------------
# 1. Preprocessing tests
# ------------------------------------------------------------------

class TestPreprocessing:

    def test_one_row_per_client(self, df_processed):
        """No duplicate client_code after preprocessing"""
        assert df_processed["client_code"].duplicated().sum() == 0, \
            "Duplicate client_code found after preprocessing"

    def test_no_list_or_dict_columns(self, df_processed):
        """No list/dict columns remaining after preprocessing"""
        bad_cols = [
            col for col in df_processed.columns
            if df_processed[col].apply(lambda x: isinstance(x, (list, dict))).any()
        ]
        assert bad_cols == [], f"List/dict columns still present: {bad_cols}"

    def test_target_not_null(self, df_processed, target_col):
        """Target column has no NaN"""
        assert df_processed[target_col].isna().sum() == 0, \
            f"Target column '{target_col}' contains NaN"

    def test_ordinal_client_genre(self, df_processed):
        """client_genre values are in {0, 1, 2} or NaN"""
        if "client_genre" in df_processed.columns:
            valid = df_processed["client_genre"].dropna().isin([0, 1, 2]).all()
            assert valid, "client_genre contains values outside {0, 1, 2}"

    def test_ordinal_nb_enfants(self, df_processed):
        """client_nb_enfants values are in range [1, 7] or NaN"""
        if "client_nb_enfants" in df_processed.columns:
            vals = df_processed["client_nb_enfants"].dropna()
            assert vals.between(1, 7).all(), \
                f"client_nb_enfants out of range [1,7]: {vals.unique()}"

    def test_no_timestamp_columns(self, df_processed):
        """No raw Timestamp columns remaining"""
        ts_cols = [
            col for col in df_processed.columns
            if df_processed[col].apply(lambda x: isinstance(x, pd.Timestamp)).any()
        ]
        assert ts_cols == [], f"Timestamp columns still present: {ts_cols}"

    def test_shape_consistency(self, df_processed, df_ready):
        """Number of rows unchanged through the pipeline"""
        assert df_processed.shape[0] == df_ready.shape[0], \
            f"Row count changed: {df_processed.shape[0]} -> {df_ready.shape[0]}"


# ------------------------------------------------------------------
# 2. Encoding tests
# ------------------------------------------------------------------

class TestEncoding:

    def test_no_object_columns_after_encoding(self, df_encoded, target_col):
        """No object/category columns remaining after encoding (except target if string)"""
        obj_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
        obj_cols = [c for c in obj_cols if c != target_col]
        assert obj_cols == [], f"Object/category columns still present after encoding: {obj_cols}"

    def test_no_nan_introduced_by_encoding(self, df_processed, df_encoded):
        """Encoding should not introduce new NaN on non-NaN columns"""
        for col in df_processed.columns:
            if col in df_encoded.columns:
                before = df_processed[col].isna().sum()
                after  = df_encoded[col].isna().sum()
                assert after <= before + 1, \
                    f"{col}: NaN increased from {before} to {after} after encoding"

    def test_nlp_columns_not_encoded(self, df_encoded):
        """NLP columns should not be present after encoding (dropped or kept raw)"""
        from Config import COLUMNS_TO_PROCESSED_WITH_NLP
        for col in COLUMNS_TO_PROCESSED_WITH_NLP:
            assert col not in df_encoded.columns, \
                f"NLP column '{col}' should not be in encoded df"

    def test_binary_columns_valid(self, df_encoded):
        """All binary columns contain only 0 and 1"""
        binary_cols = [
            col for col in df_encoded.columns
            if df_encoded[col].dropna().isin([0, 1]).all()
            and df_encoded[col].nunique() <= 2
        ]
        for col in binary_cols:
            assert set(df_encoded[col].dropna().unique()).issubset({0, 1}), \
                f"{col}: binary column contains values outside {{0, 1}}"


# ------------------------------------------------------------------
# 3. Normalization tests
# ------------------------------------------------------------------

class TestNormalization:

    def test_normalized_mean_near_zero(self, df_ready, cols_normalized):
        """Normalized columns should have mean ~0"""
        for col in cols_normalized:
            mean = df_ready[col].mean()
            assert abs(mean) < 0.01, f"{col}: mean={mean:.4f}, expected ~0"

    def test_normalized_std_near_one(self, df_ready, cols_normalized):
        """Normalized columns should have std ~1"""
        for col in cols_normalized:
            std = df_ready[col].std()
            assert abs(std - 1) < 0.01, f"{col}: std={std:.4f}, expected ~1"

    def test_no_nan_after_normalization(self, df_ready, cols_normalized):
        """Normalization should not introduce NaN"""
        for col in cols_normalized:
            assert df_ready[col].isna().sum() == 0, \
                f"{col}: contains NaN after normalization"

    def test_binary_columns_untouched(self, df_ready):
        """Binary columns should not be normalized"""
        binary_cols = [
            col for col in df_ready.columns
            if df_ready[col].dropna().isin([0, 1]).all()
            and df_ready[col].nunique() <= 2
        ]
        for col in binary_cols:
            assert set(df_ready[col].dropna().unique()).issubset({0, 1}), \
                f"{col}: binary column was altered by normalization"

    def test_target_column_untouched(self, df_ready, target_col):
        """Target column should not be normalized"""
        assert set(df_ready[target_col].unique()).issubset({0, 1}), \
            f"Target column '{target_col}' was normalized"


# ------------------------------------------------------------------
# 4. Final dataset tests
# ------------------------------------------------------------------

class TestFinalDataset:

    def test_no_constant_columns(self, df_ready):
        """No constant columns (variance = 0)"""
        constant_cols = [
            col for col in df_ready.columns
            if df_ready[col].nunique() <= 1
        ]
        assert constant_cols == [], f"Constant columns found: {constant_cols}"

    def test_no_quasi_constant_columns(self, df_ready, threshold=0.95):
        """No quasi-constant columns (>95% same value)"""
        quasi_cols = [
            col for col in df_ready.columns
            if df_ready[col].value_counts(normalize=True).iloc[0] > threshold
        ]
        assert quasi_cols == [], f"Quasi-constant columns (>{threshold*100}%): {quasi_cols}"

    def test_target_balance(self, df_ready, target_col, min_ratio=0.05):
        """Target column should not be extremely imbalanced (minority class > 5%)"""
        ratio = df_ready[target_col].value_counts(normalize=True).min()
        assert ratio > min_ratio, \
            f"Target extremely imbalanced: minority class = {ratio:.2%}"

    def test_no_duplicate_columns(self, df_ready):
        """No duplicate column names"""
        assert len(df_ready.columns) == len(set(df_ready.columns)), \
            "Duplicate column names found"

    def test_no_infinite_values(self, df_ready):
        """No infinite values in numeric columns"""
        num_cols = df_ready.select_dtypes(include=[np.number]).columns
        inf_cols = [col for col in num_cols if np.isinf(df_ready[col]).any()]
        assert inf_cols == [], f"Infinite values found in: {inf_cols}"

    def test_client_code_present(self, df_ready):
        """client_code column should be present"""
        assert "client_code" in df_ready.columns, "client_code column missing from final dataset"