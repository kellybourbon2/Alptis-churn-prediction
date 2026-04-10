"""Load raw and processsed data - supports both local files and S3 storage for loading of raw files"""

import os
import pandas as pd
from typing import Literal
import sys
import logging
import s3fs

# Add parent directories to path to import Config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import Config


def _load_from_local(set: str, files: dict) -> dict:
    """Load datasets from local data_raw directory"""
    data_frames = {}
    for key, filename in files.items():
        filepath = os.path.join(Config.DATA_RAW_DIR, set, filename)
        try:
            data_frames[key] = pd.read_csv(filepath, sep=";", index_col=False,low_memory=False )
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
    return data_frames


def _load_from_s3(set: str, files: dict) -> dict:
    """Load datasets from S3 storage"""
    
    # Set up S3 connection
    fs = s3fs.S3FileSystem(
        endpoint_url='https://' + Config.S3_ENDPOINT,
        key=Config.S3_ACCESS_KEY,
        secret=Config.S3_SECRET_KEY,
        token=Config.S3_SESSION_TOKEN,
        client_kwargs={'verify': Config.S3_VERIFY_SSL}
    )
    
    data_frames = {}
    for key, filename in files.items():
        # S3 path: bucket/alptis/set/filename
        s3_path = f"{Config.S3_BUCKET}/alptis/{set}/{filename}"
        try:
            with fs.open(s3_path, 'rb') as f:
                data_frames[key] = pd.read_csv(f, sep=";", index_col=False, low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found on S3: {s3_path}")
    return data_frames


def data_loading(set: Literal["training", "validation", "evaluation"]) -> tuple[pd.DataFrame, ...]:
    """Load the datasets from either local folder or S3 based on Config.LOAD_FROM_S3
    
    Args: 
        set: the set to load ("training", "validation", or "evaluation")
    
    Returns: 
        a tuple of pandas DataFrames in this order:
        (portefeuille, consommations, reclamations, interactions, impayes)
    """
    # Validate set parameter
    if set not in Config.DATASET_MAPPING:
        raise ValueError(f"Invalid set parameter: '{set}'. Must be one of {list(Config.DATASET_MAPPING.keys())}")
    
    # Get file configuration for this set
    files = Config.DATASET_MAPPING[set]
    
    # Load from appropriate source
    if Config.LOAD_FROM_S3:
        data_frames = _load_from_s3(set, files)
    else:
        data_frames = _load_from_local(set, files)
    
    # Return in consistent order: 
    # (portefeuille, consommations, reclamations, interactions, impayes)
    return (
        data_frames["portefeuille"], 
        data_frames["consommations"],
        data_frames["reclamations"],
        data_frames["interactions"], 
        data_frames["impayes"])

def load_data_processed_from_S3(dataset: str):
    """
    Load a processed dataset from S3 storage.
    Args:
        dataset: Name of the dataset to load.
                 Expected values: 'X_train', 'X_test', 'y_train', 'y_test'.
    Returns:
        pd.DataFrame: The requested dataset.
    """
    # Set up S3 connection
    fs = s3fs.S3FileSystem(
        endpoint_url='https://' + Config.S3_ENDPOINT,
        key=Config.S3_ACCESS_KEY,
        secret=Config.S3_SECRET_KEY,
        token=Config.S3_SESSION_TOKEN,
        client_kwargs={'verify': Config.S3_VERIFY_SSL}
    ) 
    s3_path = f"{Config.S3_DATA_PROCESSED_BUCKET}/{dataset}.parquet"
    try:
        with fs.open(s3_path, 'rb') as f:
            file = pd.read_parquet(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found on S3: {s3_path}")
    return file

def save_data_processed_parquet_to_s3(df: pd.DataFrame, dataset: str) -> None:
    """
    Save a DataFrame as parquet to S3 storage.
    Args:
        df:      DataFrame to save.
        dataset: Dataset name used as filename (e.g. 'X_train', 'y_test').
    """
    fs = s3fs.S3FileSystem(
        endpoint_url='https://' + Config.S3_ENDPOINT,
        key=Config.S3_ACCESS_KEY,
        secret=Config.S3_SECRET_KEY,
        token=Config.S3_SESSION_TOKEN,
        client_kwargs={'verify': Config.S3_VERIFY_SSL}
    )
    s3_path = f"{Config.S3_DATA_PROCESSED_BUCKET}/{dataset}.parquet"
    try:
        with fs.open(s3_path, 'wb') as f:
            df.to_parquet(f, index=False)
        logging.info(f"{dataset} saved to S3: {s3_path}")
    except Exception as e:
        logging.error(f"Failed to save {dataset} to S3: {e}")
        raise
